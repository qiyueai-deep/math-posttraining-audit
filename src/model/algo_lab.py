"""后训练算法实验台: 可解析梯度下的 SFT -> DPO 全流程 (零第三方依赖)。

这一层回答「大模型算法岗作品集」最要紧的问题: 在没有 GPU 的机器上,
能不能真的把 DPO 的目标函数、梯度、参考模型锚点、beta、KL 和长度偏置
跑起来, 而不是只在文档里贴一段公式。

做法是把策略写成候选集上的 log-linear 模型

    log pi_theta(y|x) = theta . phi(x,y) - log Z_theta(x),
    Z_theta(x) = sum_{y in Y(x)} exp(theta . phi(x,y)),

phi 含 6 个表面特征和 1 个由严格验证器给出的正确性特征。因为成对比较里
log Z 抵消, DPO 的隐式 reward 差和梯度都有闭式解:

    Delta_i = (theta - theta_ref) . (phi_i^+ - phi_i^-),
    L_i     = -log sigmoid(beta * Delta_i),
    grad_i  = -beta * (1 - sigmoid(beta * Delta_i)) * (phi_i^+ - phi_i^-).

KL(pi_theta || pi_ref) 在候选集上可以精确求和, 不需要 Monte Carlo。
这正好和真实 LLM 层形成对照: 那里 Z 不可算, 只能采样估计。

诚实边界: 这里训练的是一个 7 维解析策略, 不是 Qwen3-0.6B 的权重。
所有 algo_lab 记录的 note 都写明这一点。results.jsonl 里的
model_stage 记录仍然是 pending_model_run, 两者绝不互相冒充。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from src.core.preferences import GROUPS, load as load_pref, wrong_answer
from src.core.verifier import CORRECT, VERIFIERS

FEATURES = ("length", "marker", "latex", "digits", "punct", "newline", "correctness")
DIM = len(FEATURES)
QUALITY = FEATURES.index("correctness")
LENGTH = FEATURES.index("length")
NOTE = ("analytic 7-dim log-linear policy on CPU; mechanism validation, "
        "NOT Qwen3/LoRA weights; model_stage metrics remain pending")


def phi(text: str, gold: str) -> list[float]:
    """回答的特征。correctness 由严格三态验证器给出, UNVERIFIABLE 记 0。"""
    v = VERIFIERS["strict_poly"].verify(text, gold)
    return [
        len(text) / 100.0,
        1.0 if "最终答案：" in text else 0.0,
        text.count("\\") / 10.0,
        sum(ch.isdigit() for ch in text) / 10.0,
        sum(ch in "，。；：、,.;:!?！？" for ch in text) / 10.0,
        float(text.count("\n")),
        1.0 if v.verdict == CORRECT else 0.0,
    ]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def log_sigmoid(z: float) -> float:
    if z >= 0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    z = sum(e)
    return [x / z for x in e]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def kl(theta: list[float], ref: list[float], cands: list[list[float]]) -> float:
    """候选集上的精确 KL(pi_theta || pi_ref), 不是 MC 估计。"""
    p = softmax([dot(theta, c) for c in cands])
    q = softmax([dot(ref, c) for c in cands])
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def dpo_loss(theta: list[float], ref: list[float], diffs: list[list[float]],
             beta: float) -> float:
    return -sum(log_sigmoid(beta * dot([a - b for a, b in zip(theta, ref)], d))
                for d in diffs) / len(diffs)


def dpo_grad(theta: list[float], ref: list[float], diffs: list[list[float]],
             beta: float) -> list[float]:
    g = [0.0] * DIM
    gap = [a - b for a, b in zip(theta, ref)]
    for d in diffs:
        t = beta * dot(gap, d)
        w = -beta * (1.0 - sigmoid(t)) / len(diffs)
        for j in range(DIM):
            g[j] += w * d[j]
    return g


# --------------------------------------------------------------- 数据构造

def build_pairs(group: str, data_dir: Path) -> list[list[float]]:
    """每对偏好数据 -> phi(chosen) - phi(rejected)。"""
    rows = load_pref(group, data_dir)
    out = []
    for r in rows:
        a = phi(r["chosen"], r["answer"])
        b = phi(r["rejected"], r["answer"])
        out.append([x - y for x, y in zip(a, b)])
    return out


def build_probe(rows: list[dict], n: int, seed: int) -> list[dict]:
    """2x2 平衡探针: 正确/错误 x 短/长, 用来分离长度偏好与质量偏好。

    短对、短错等长; 长对、长错等长。这样选中长回答只能来自长度特征,
    选中正确回答只能来自 correctness 特征。
    """
    rng = random.Random(seed)
    pool = [r for r in rows if r["split"] == "test"
            and wrong_answer(r["answer"]) is not None]
    rng.shuffle(pool)
    pool = pool[:n]
    probes = []
    for r in pool:
        gold, chosen = r["answer"], r["chosen"]
        w = wrong_answer(gold)
        short_ok = f"直接得到最终答案：{gold}"
        short_bad = f"直接得到最终答案：{w}"
        long_bad = chosen.replace(gold, w)
        if long_bad == chosen or len(long_bad) != len(chosen):
            continue
        texts = [short_ok, chosen, short_bad, long_bad]
        feats = [phi(t, gold) for t in texts]
        if feats[0][QUALITY] != 1.0 or feats[1][QUALITY] != 1.0:
            raise AssertionError("probe correct candidate not verified as correct")
        if feats[2][QUALITY] != 0.0 or feats[3][QUALITY] != 0.0:
            raise AssertionError("probe wrong candidate verified as correct")
        # reward hacking 探针: 短的正确回答 vs 明显更长的错误回答。
        # 只有长度权重压过正确性权重时才会选错者, 直接量化长度偏置。
        padded_bad = long_bad + " 补充说明：以上各步均已反复核对，确认无误。" * 6
        hack_feats = [feats[0], phi(padded_bad, gold)]
        if hack_feats[1][QUALITY] != 0.0:
            raise AssertionError("padded wrong answer verified as correct")
        probes.append({"id": r["id"], "gold": gold, "feats": feats,
                       "hack_feats": hack_feats,
                       "short": (0, 2), "long": (1, 3),
                       "correct": (0, 1), "wrong": (2, 3)})
    return probes


def probe_scores(theta: list[float], probes: list[dict]) -> dict:
    """用 softmax 概率而非硬 argmax 打分, 避免平局时按索引取第一个造成虚高。"""
    acc = longs = within = hack = 0.0
    for p in probes:
        logits = [dot(theta, f) for f in p["feats"]]
        probs = softmax(logits)
        acc += probs[0] + probs[1]
        longs += probs[1] + probs[3]
        # 同一长度层内, 正确回答排在错误回答前面的概率
        for ok, bad in ((0, 2), (1, 3)):
            within += sigmoid(logits[ok] - logits[bad])
        hp = softmax([dot(theta, f) for f in p["hack_feats"]])
        hack += hp[1]
    n = len(probes)
    return {"accuracy": acc / n, "long_rate": longs / n,
            "correct_within_length": within / (2 * n),
            "hack_rate": hack / n, "n": n}


# --------------------------------------------------------------- 训练

def train_sft(rows: list[dict], seed: int, steps: int = 120,
              lr: float = 0.03) -> tuple[list[float], dict]:
    """在 chosen 上做 log-linear MLE, 梯度 = phi(chosen) - E_pi[phi]。"""
    rng = random.Random(seed + 1)
    train = [r for r in rows if r["split"] == "train" and r["subset"] == "train_id"]
    rng.shuffle(train)
    samples = []
    for i, r in enumerate(train):
        gold = r["answer"]
        w = wrong_answer(gold)
        distractors = ["我无法确定。",
                       f"直接得到最终答案：{w if w else gold}",
                       train[(i + 37) % len(train)]["chosen"]]
        cands = [phi(r["chosen"], gold)] + [phi(t, gold) for t in distractors]
        samples.append(cands)
    theta = [0.0] * DIM
    batch = samples[::4]
    hist = []
    for step in range(steps):
        # NLL 梯度 = E_pi[phi] - phi(chosen); 这里 g 取其相反数, 故用 += (梯度下降)
        g = [0.0] * DIM
        for cands in batch:
            p = softmax([dot(theta, c) for c in cands])
            for j in range(DIM):
                g[j] += (cands[0][j] - sum(pk * c[j] for pk, c in zip(p, cands))) / len(batch)
        for j in range(DIM):
            theta[j] += lr * g[j]
        if step % 40 == 0 or step == steps - 1:
            hist.append({"step": step, "theta_quality": theta[QUALITY]})
    return theta, {"n_train": len(samples), "steps": steps, "lr": lr,
                   "final_theta": list(theta), "history": hist}


def train_dpo(theta_init: list[float], ref: list[float], diffs: list[list[float]],
              beta: float, probes: list[dict], steps: int = 400,
              lr: float = 0.08) -> dict:
    theta = list(theta_init)
    curve = []
    for step in range(steps):
        g = dpo_grad(theta, ref, diffs, beta)
        for j in range(DIM):
            theta[j] -= lr * g[j]
        if step % 20 == 0 or step == steps - 1:
            gap = [a - b for a, b in zip(theta, ref)]
            margin = beta * sum(dot(gap, d) for d in diffs) / len(diffs)
            ps = probe_scores(theta, probes)
            curve.append({"step": step, "loss": dpo_loss(theta, ref, diffs, beta),
                          "reward_margin": margin,
                          "reward_accuracy": sum(dot(gap, d) > 0 for d in diffs) / len(diffs),
                          "probe_accuracy": ps["accuracy"],
                          "probe_hack_rate": ps["hack_rate"],
                          "theta_length": theta[LENGTH],
                          "theta_quality": theta[QUALITY]})
    gap = [a - b for a, b in zip(theta, ref)]
    margin = beta * sum(dot(gap, d) for d in diffs) / len(diffs)
    ps = probe_scores(theta, probes)
    return {
        "theta": list(theta), "curve": curve,
        "loss": dpo_loss(theta, ref, diffs, beta),
        "reward_margin": margin,
        "reward_accuracy": sum(dot(gap, d) > 0 for d in diffs) / len(diffs),
        "margin_over_beta": margin / beta,
        "kl": sum(kl(theta, ref, p["feats"]) for p in probes) / len(probes),
        "theta_length": theta[LENGTH], "theta_quality": theta[QUALITY],
        **{f"probe_{k}": v for k, v in ps.items()},
    }


# --------------------------------------------------------------- 实验入口

def run(store, root: Path, rows: list[dict], seed: int,
        n_probe: int = 120, steps: int = 300) -> dict:
    data_dir = root / "data"
    chk = self_test()
    store.record("algo_lab", "dpo_gradient_max_abs_error",
                 chk["gradient_max_abs_error"], unit="abs",
                 note=NOTE + "; analytic gradient vs central finite difference")
    store.record("algo_lab", "gradient_check_passed", 1 if chk["ok"] else 0,
                 unit="bool", note=NOTE)
    if not chk["ok"]:
        raise AssertionError(f"DPO gradient check failed: {chk}")
    probes = build_probe(rows, n_probe, seed)
    if not probes:
        raise AssertionError("no valid 2x2 probes built")
    theta_sft, sft_info = train_sft(rows, seed)
    theta_base = [0.0] * DIM
    base_scores = probe_scores(theta_base, probes)
    sft_scores = probe_scores(theta_sft, probes)

    store.record("algo_lab", "n_probe_items", len(probes), unit="items", seed=seed,
                 note=NOTE)
    store.record("algo_lab", "n_feature_dims", DIM, unit="dims", note=NOTE)
    store.record("algo_lab", "sft_train_items", sft_info["n_train"], unit="items",
                 note=NOTE)
    store.record("algo_lab", "probe_accuracy_base", base_scores["accuracy"],
                 unit="rate", n=len(probes), seed=seed,
                 note=NOTE + "; theta=0 reference policy")
    store.record("algo_lab", "probe_accuracy_sft", sft_scores["accuracy"],
                 unit="rate", n=len(probes), seed=seed, note=NOTE)
    store.record("algo_lab", "sft_gain_over_base",
                 sft_scores["accuracy"] - base_scores["accuracy"],
                 unit="abs", n=len(probes), note=NOTE)
    store.record("algo_lab", "sft_theta_quality", theta_sft[QUALITY],
                 unit="coef", note=NOTE)

    group_runs = {}
    for g in GROUPS:
        diffs = build_pairs(g, data_dir)
        if not diffs:
            raise AssertionError(f"empty preference group: {g}")
        res = train_dpo(list(theta_sft), list(theta_sft), diffs,
                        beta=0.1, probes=probes, steps=steps)
        group_runs[g] = res
        store.record("algo_lab", f"n_pairs_{g}", len(diffs), unit="pairs", note=NOTE)
        for key in ("loss", "reward_margin", "reward_accuracy", "kl",
                    "theta_length", "theta_quality", "probe_accuracy",
                    "probe_long_rate", "probe_correct_within_length", "probe_hack_rate"):
            store.record("algo_lab", f"{key}_{g}", res[key], unit="rate" if "accuracy" in key
                         or "rate" in key else "value", n=len(probes), seed=seed, note=NOTE)
        store.record("algo_lab", f"dpo_gain_over_sft_{g}",
                     res["probe_accuracy"] - sft_scores["accuracy"],
                     unit="abs", n=len(probes), note=NOTE)
        dq = res["theta_quality"] - theta_sft[QUALITY]
        dl = res["theta_length"] - theta_sft[LENGTH]
        store.record("algo_lab", f"theta_quality_delta_{g}", dq, unit="coef",
                     note=NOTE + "; DPO gradient spent on true quality")
        store.record("algo_lab", f"theta_length_delta_{g}", dl, unit="coef",
                     note=NOTE + "; DPO gradient spent on length shortcut")
        store.record("algo_lab", f"quality_per_length_{g}",
                     abs(dq) / (abs(dl) + 0.01), unit="ratio",
                     note=NOTE + "; |d quality| / (|d length| + 0.01 smoothing)")

    # beta 扫描。分开两件事:
    #  (a) 固定策略时 margin/beta = mean Delta, 与 beta 无关, 是恒等式, 不是发现;
    #  (b) 真正训练时 beta 会改变优化轨迹, 此时要看精确 KL 与下游准确率。
    diffs = build_pairs("deconfounded", data_dir)
    fixed = [x + 0.05 for x in theta_sft]
    gap_fixed = [a - b for a, b in zip(fixed, theta_sft)]
    mean_delta = sum(dot(gap_fixed, d) for d in diffs) / len(diffs)
    ratios_fixed = [(beta * mean_delta) / beta for beta in (0.05, 0.1, 0.3, 0.5)]
    cv_fixed = (statistics.pstdev(ratios_fixed) / statistics.fmean(ratios_fixed)
                if statistics.fmean(ratios_fixed) else 0.0)
    store.record("algo_lab", "margin_over_beta_cv_fixed_theta", cv_fixed, unit="cv",
                 note=NOTE + "; fixed theta: margin = beta * mean Delta is an identity")
    store.record("algo_lab", "fixed_theta_mean_delta", mean_delta, unit="logratio",
                 note=NOTE)

    sweep = {}
    for beta in (0.05, 0.1, 0.3, 0.5):
        r = train_dpo(list(theta_sft), list(theta_sft), diffs, beta,
                      probes, steps=steps)
        sweep[str(beta)] = {"margin": r["reward_margin"],
                            "margin_over_beta": r["margin_over_beta"],
                            "kl": r["kl"], "probe_accuracy": r["probe_accuracy"]}
        tag = f"beta_{str(beta).replace('.', 'p')}"
        store.record("algo_lab", f"{tag}_kl", r["kl"], unit="nats",
                     n=len(probes), note=NOTE)
        store.record("algo_lab", f"{tag}_probe_accuracy", r["probe_accuracy"],
                     unit="rate", n=len(probes), note=NOTE)
        store.record("algo_lab", f"{tag}_reward_margin", r["reward_margin"],
                     unit="value", note=NOTE)
    ratios = [v["margin_over_beta"] for v in sweep.values()]
    cv = (statistics.pstdev(ratios) / statistics.fmean(ratios)
          if statistics.fmean(ratios) else float("nan"))
    store.record("algo_lab", "margin_over_beta_cv_after_training", cv, unit="cv",
                 note=NOTE + "; beta changes the optimization path, unlike case (a)")
    store.record("algo_lab", "beta_kl_range",
                 max(v["kl"] for v in sweep.values()) - min(v["kl"] for v in sweep.values()),
                 unit="nats", note=NOTE)
    store.record("algo_lab", "beta_probe_accuracy_range",
                 max(v["probe_accuracy"] for v in sweep.values())
                 - min(v["probe_accuracy"] for v in sweep.values()),
                 unit="abs", note=NOTE)

    # ref 锚点对照: 正确做法 ref=SFT, 旧项目缺陷 ref=base(TRL ref_model=None)
    correct = group_runs["deconfounded"]
    wrong = train_dpo(list(theta_sft), theta_base, diffs, beta=0.1,
                      probes=probes, steps=steps)
    store.record("algo_lab", "wrong_anchor_kl", wrong["kl"], unit="nats",
                 n=len(probes), note=NOTE + "; ref=base while policy starts from SFT")
    store.record("algo_lab", "anchor_kl_gap", abs(wrong["kl"] - correct["kl"]),
                 unit="nats", note=NOTE)
    store.record("algo_lab", "anchor_probe_accuracy_gap",
                 abs(wrong["probe_accuracy"] - correct["probe_accuracy"]),
                 unit="abs", n=len(probes), note=NOTE)
    store.record("algo_lab", "anchor_margin_gap",
                 abs(wrong["reward_margin"] - correct["reward_margin"]),
                 unit="value", note=NOTE)

    # 训练曲线存成可画图的数据, 并生成一张纯 SVG (不依赖 matplotlib)
    curves = {"deconfounded": group_runs["deconfounded"]["curve"],
              "shortcut": group_runs["shortcut"]["curve"],
              "length_reversed": group_runs["length_reversed"]["curve"]}
    (root / "reports" / "algo_lab_curves.json").write_text(
        json.dumps({"note": NOTE, "curves": curves, "beta_sweep": sweep},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "reports" / "algo_lab_curve.svg").write_text(
        render_curve_svg(curves), encoding="utf-8")

    return {
        "NOTE": NOTE, "n_probe": len(probes), "sft": sft_info,
        "base_scores": base_scores, "sft_scores": sft_scores,
        "groups": {g: {k: v for k, v in r.items() if k not in ("theta", "curve")}
                   for g, r in group_runs.items()},
        "beta_sweep": sweep, "margin_over_beta_cv_after_training": cv,
        "margin_over_beta_cv_fixed_theta": cv_fixed,
        "anchor": {"correct_kl": correct["kl"], "wrong_kl": wrong["kl"],
                   "correct_probe": correct["probe_accuracy"],
                   "wrong_probe": wrong["probe_accuracy"]},
    }


def render_curve_svg(curves: dict) -> str:
    """把 reward margin 与下游 probe 准确率画在同一张图上, 纯标准库。"""
    W, H, PAD = 720, 320, 48
    main = curves["shortcut"]
    steps = [c["step"] for c in main]
    smax = max(steps) or 1
    marg = [c["reward_margin"] for c in main]
    mmax = max(abs(x) for x in marg) or 1.0

    def X(s):
        return PAD + (W - 2 * PAD) * s / smax

    def Y(v, scale):
        return H - PAD - (H - 2 * PAD) * (v / scale)

    def path(vals, scale):
        return " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v, scale):.1f}"
                        for i, (s, v) in enumerate(zip(steps, vals)))

    acc = [c["probe_hack_rate"] for c in main]
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 320" width="720" height="320">',
        '<rect width="720" height="320" fill="#ffffff"/>',
        '<text x="48" y="26" font-size="15" fill="#111827" font-family="sans-serif">'
        'shortcut 偏好组：reward margin 上升，长度偏置劫持率同时上升</text>',
        f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#9ca3af"/>',
        f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#9ca3af"/>',
        f'<path d="{path(marg, mmax)}" fill="none" stroke="#dc2626" stroke-width="2.5"/>',
        f'<path d="{path(acc, 1.0)}" fill="none" stroke="#2563eb" stroke-width="2.5"/>',
        f'<text x="{W-PAD-180}" y="{PAD+16}" font-size="12" fill="#dc2626">reward margin</text>',
        f'<text x="{W-PAD-180}" y="{PAD+34}" font-size="12" fill="#2563eb">hack rate</text>',
        f'<text x="{PAD}" y="{H-16}" font-size="12" fill="#374151">step 0</text>',
        f'<text x="{W-PAD-44}" y="{H-16}" font-size="12" fill="#374151">step {smax}</text>',
        '<text x="48" y="300" font-size="11" fill="#6b7280">'
        '数据来源 reports/algo_lab_curves.json，由 src/model/algo_lab.py 真实训练产生</text>',
        '</svg>', '',
    ])


def self_test() -> dict:
    """有限差分验证 DPO 梯度, 并检查 KL 的基本性质。"""
    rng = random.Random(7)
    diffs = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(9)]
    theta = [rng.uniform(-0.4, 0.4) for _ in range(DIM)]
    ref = [rng.uniform(-0.4, 0.4) for _ in range(DIM)]
    beta = 0.2
    ana = dpo_grad(theta, ref, diffs, beta)
    eps = 1e-6
    num = []
    for j in range(DIM):
        a, b = list(theta), list(theta)
        a[j] += eps
        b[j] -= eps
        num.append((dpo_loss(a, ref, diffs, beta) - dpo_loss(b, ref, diffs, beta)) / (2 * eps))
    max_err = max(abs(x - y) for x, y in zip(ana, num))
    cands = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(5)]
    return {"gradient_max_abs_error": max_err,
            "kl_self": kl(theta, theta, cands),
            "kl_nonnegative": kl(theta, ref, cands) >= -1e-12,
            "ok": max_err < 1e-6 and abs(kl(theta, theta, cands)) < 1e-12
            and kl(theta, ref, cands) >= -1e-12}


if __name__ == "__main__":
    from src.core.results import ResultStore

    ap = argparse.ArgumentParser(description="后训练算法实验台（零依赖，可右键运行）")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=str(_PYCHARM_ROOT))
    a = ap.parse_args()
    if a.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
    else:
        r = Path(a.root)
        rows = [json.loads(l) for name in ("train", "valid", "test")
                for l in (r / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        st = ResultStore(r)
        out = run(st, r, rows, seed=20251109)
        print(json.dumps({k: v for k, v in out.items() if k in
                          ("n_probe", "sft_scores", "groups",
                           "margin_over_beta_cv_fixed_theta",
                           "margin_over_beta_cv_after_training", "anchor")},
                         ensure_ascii=False, indent=2)[:4000])
        print("\n注意: 这是解析策略上的机制验证, 未写入 results.jsonl 时需运行 run_research。")
