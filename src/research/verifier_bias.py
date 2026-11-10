"""子贡献 3：验证器误判校正。

自动验证器本身是一个有假阳性和假阴性的测量工具。旧实现把
`gold in pred` 的子串匹配当判对, 且把不可解析样本静默回退成字符串匹配,
观测准确率完全不可信。

这里:
  1. 用生成器构造真值已知的 calibration 与 survey 集合(不是人工猜测)。
  2. 估计 FPR/FNR 及其 Wilson 区间, 用 Rogan-Gladen 型矩估计还原真实准确率,
     并给 bootstrap 区间和覆盖检验。
  3. 用两个判定路径不同的 verifier 做 latent class EM, 在假定无金标准时
     也能估计真实准确率, 并做条件独立假设的敏感性分析。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from src.core.preferences import wrong_answer
from src.core.verifier import CORRECT, UNVERIFIABLE, VERIFIERS, binary
from src.research.cluster_inference import wilson_ci


def _sample_rows(rows: list[dict], rng: random.Random, n: int) -> list[dict]:
    pool = [r for r in rows if wrong_answer(r["answer"]) is not None]
    rng.shuffle(pool)
    return pool[:n]


def equivalent_form(ans: str) -> str | None:
    """数学上等价、字面不同的写法。严格解析器判对, 浮点/子串验证器会分歧。

    这是让三个 verifier 产生真实判定差异的关键, 否则潜类别模型会退化成
    「数正例」而完全不可识别。
    """
    if re.fullmatch(r"-?\d+/\d+", ans):
        n, d = ans.split("/")
        return f"{int(n) * 2}/{int(d) * 2}"
    if re.fullmatch(r"-?\d+", ans):
        return f"{ans}/1"
    if "," in ans:
        return ",".join(reversed(ans.split(",")))
    return None


def make_calibration(rows: list[dict], seed: int, n_per_class: int = 150) -> list[dict]:
    """真值已知: 1 = 输出确实给出正确最终答案, 0 = 否。"""
    rng = random.Random(seed + 1)
    pool = _sample_rows(rows, rng, n_per_class)
    out = []
    for i, r in enumerate(pool):
        gold, chosen = r["answer"], r["chosen"]
        w = wrong_answer(gold)
        bad = chosen.replace(gold, w) if w else chosen
        eq = equivalent_form(gold)
        good_text = chosen.replace(gold, eq) if (eq and i % 3 == 0) else chosen
        out.append({"text": good_text, "gold": gold, "truth": 1, "kind": "correct"})
        out.append({"text": bad, "gold": gold, "truth": 0, "kind": "wrong_answer"})
        out.append({"text": r["rationale"], "gold": gold, "truth": 0,
                    "kind": "missing_marker"})
        out.append({"text": "我无法确定。", "gold": gold, "truth": 0,
                    "kind": "refusal"})
        # 正文里出现正确答案但没有按协议给出最终答案。真值为 0,
        # 宽松子串验证器会判对, 这正是旧项目 expectation 题假阳性的来源。
        out.append({"text": r["rationale"] + f"（注意：这里出现了 {gold}，"
                            "但它不是按协议给出的最终答案。）",
                    "gold": gold, "truth": 0, "kind": "answer_leak_in_body"})
    return out


def make_survey(rows: list[dict], seed: int, n: int = 400,
                target_pi: float = 0.70) -> list[dict]:
    """survey 的真实 pi 由构造确定, 用于检验校正估计量是否无偏。"""
    rng = random.Random(seed + 2)
    pool = _sample_rows(rows, rng, max(n, 1))
    out = []
    for i, r in enumerate(pool[:n]):
        w = wrong_answer(r["answer"])
        if rng.random() < target_pi:
            eq = equivalent_form(r["answer"])
            text = r["chosen"].replace(r["answer"], eq) if (eq and i % 3 == 0) else r["chosen"]
            out.append({"text": text, "gold": r["answer"], "truth": 1})
        else:
            kind = i % 4
            if kind == 0 and w:
                text = r["chosen"].replace(r["answer"], w)
            elif kind == 1:
                text = r["rationale"]
            elif kind == 2:
                text = "我无法确定。"
            else:
                text = r["rationale"] + f"（注意：这里出现了 {r['answer']}，"
                "但它不是按协议给出的最终答案。）"
            out.append({"text": text, "gold": r["answer"], "truth": 0})
    return out


def _judge(record: dict, verifier_name: str) -> str:
    return VERIFIERS[verifier_name].verify(record["text"], record["gold"]).verdict


def estimate_error_rates(calibration: list[dict], verifier_name: str) -> dict:
    tp = fp = tn = fn = 0
    n_unver = 0
    for r in calibration:
        v = _judge(r, verifier_name)
        n_unver += int(v == UNVERIFIABLE)
        if r["truth"] == 1:
            if v == CORRECT:
                tp += 1
            else:
                fn += 1
        else:
            if v == CORRECT:
                fp += 1
            else:
                tn += 1
    n0, n1 = fp + tn, tp + fn
    fpr = fp / n0 if n0 else 0.0
    fnr = fn / n1 if n1 else 0.0
    coverage = 1 - n_unver / len(calibration)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": fpr, "fnr": fnr,
        "fpr_ci": wilson_ci(fp, n0), "fnr_ci": wilson_ci(fn, n1),
        "coverage": coverage, "n": len(calibration),
    }


def observed_accuracy(survey: list[dict], verifier_name: str) -> float:
    return sum(_judge(r, verifier_name) == CORRECT for r in survey) / len(survey)


def corrected_accuracy(p_hat: float, fpr: float, fnr: float) -> float:
    denom = 1.0 - fpr - fnr
    if denom <= 0:
        return float("nan")
    return max(0.0, min(1.0, (p_hat - fpr) / denom))


def bootstrap_corrected(calibration: list[dict], survey: list[dict],
                        verifier_name: str, B: int = 2000,
                        seed: int = 20251109) -> dict:
    """重采样的是已判定的标记, 不是重新调用解析器, 保证 B 次可在秒级完成。"""
    cal_flag = [int(_judge(r, verifier_name) == CORRECT) for r in calibration]
    cal_truth = [r["truth"] for r in calibration]
    sur_flag = [int(_judge(r, verifier_name) == CORRECT) for r in survey]
    nc, ns = len(calibration), len(survey)
    rng = random.Random(seed + 3)
    vals = []
    for _ in range(B):
        tp = fn = fp = tn = 0
        for _ in range(nc):
            i = rng.randrange(nc)
            if cal_truth[i] == 1:
                if cal_flag[i]: tp += 1
                else: fn += 1
            else:
                if cal_flag[i]: fp += 1
                else: tn += 1
        n0, n1 = fp + tn, tp + fn
        fpr = fp / n0 if n0 else 0.0
        fnr = fn / n1 if n1 else 0.0
        p = sum(sur_flag[rng.randrange(ns)] for _ in range(ns)) / ns
        v = corrected_accuracy(p, fpr, fnr)
        if math.isfinite(v):
            vals.append(v)
    vals.sort()
    return {"lo": vals[int(0.025 * len(vals))],
            "hi": vals[min(len(vals) - 1, int(0.975 * len(vals)))],
            "B": len(vals)}


def latent_class_em(judgements: list[list[int]], restarts: int = 8,
                    iters: int = 300, seed: int = 20251109) -> dict:
    """J 个 verifier 条件独立时的二类别潜变量 EM。

    pi = P(真值正确), se_j = P(verifier j 判 1 | 真值 1),
    sp_j = P(verifier j 判 0 | 真值 0)。

    自由度核算: 2^J-1 个独立模式频率, 1+2J 个参数。
    J=2 时 3 < 5, 模型欠识别, EM 只能收敛到边界解, 估计不可信。
    J=3 时 7 = 7, 恰好识别。因此本项目一律使用三个判定路径不同的 verifier。
    """
    from itertools import product

    J = len(judgements)
    n = len(judgements[0])
    rng = random.Random(seed + 4)
    patt = Counter(tuple(jl[i] for jl in judgements) for i in range(n))
    keys = list(product((0, 1), repeat=J))
    counts = [patt.get(k, 0) for k in keys]
    N = sum(counts)
    n_free = 2 ** J - 1
    n_params = 1 + 2 * J
    if N == 0:
        return {"pi": float("nan"), "loglik": float("nan"), "n": 0}

    def _clip(x):
        return min(0.999, max(0.001, x))

    def loglik(pi, se, sp):
        ll = 0.0
        for pat, c in zip(keys, counts):
            p1 = math.prod(se[j] if pat[j] else 1 - se[j] for j in range(J))
            p0 = math.prod((1 - sp[j]) if pat[j] else sp[j] for j in range(J))
            ll += c * math.log(max(pi * p1 + (1 - pi) * p0, 1e-300))
        return ll

    best = None
    for _ in range(restarts):
        pi = rng.uniform(0.3, 0.8)
        se = [rng.uniform(0.55, 0.98) for _ in range(J)]
        sp = [rng.uniform(0.55, 0.98) for _ in range(J)]
        for _ in range(iters):
            wsum = n0 = 0.0
            se_num = [0.0] * J
            sp_num = [0.0] * J
            for pat, c in zip(keys, counts):
                p1 = math.prod(se[j] if pat[j] else 1 - se[j] for j in range(J))
                p0 = math.prod((1 - sp[j]) if pat[j] else sp[j] for j in range(J))
                w = pi * p1 / max(pi * p1 + (1 - pi) * p0, 1e-300)
                wt, w0t = c * w, c * (1 - w)
                wsum += wt
                n0 += w0t
                for j in range(J):
                    if pat[j]:
                        se_num[j] += wt
                    else:
                        sp_num[j] += w0t
            pi = _clip(wsum / N)
            se = [_clip(se_num[j] / wsum if wsum else se[j]) for j in range(J)]
            sp = [_clip(sp_num[j] / n0 if n0 else sp[j]) for j in range(J)]
        ll = loglik(pi, se, sp)
        if best is None or ll > best["loglik"]:
            best = {"pi": pi, "se": se, "sp": sp, "loglik": ll, "n": N,
                    "J": J, "n_free_stats": n_free, "n_params": n_params,
                    "identified": n_free >= n_params}
    return best


def run(store, rows: list[dict], seed: int, B: int = 2000) -> dict:
    cal = make_calibration(rows, seed)
    survey = make_survey(rows, seed)
    pi_true = sum(r["truth"] for r in survey) / len(survey)
    out = {"pi_true": pi_true, "verifiers": {}}

    for name in VERIFIERS:
        er = estimate_error_rates(cal, name)
        p_hat = observed_accuracy(survey, name)
        pi_hat = corrected_accuracy(p_hat, er["fpr"], er["fnr"])
        bs = bootstrap_corrected(cal, survey, name, B=B, seed=seed)
        naive_bias = p_hat - pi_true
        corrected_bias = pi_hat - pi_true
        out["verifiers"][name] = {
            "error_rates": er, "observed_accuracy": p_hat,
            "corrected_accuracy": pi_hat, "bootstrap_ci": (bs["lo"], bs["hi"]),
            "naive_bias": naive_bias, "corrected_bias": corrected_bias,
            "ci_covers_truth": bs["lo"] <= pi_true <= bs["hi"],
        }
        store.record("verifier_bias", f"fpr_{name}", er["fpr"], unit="rate",
                     n=er["n"], ci_low=er["fpr_ci"][0], ci_high=er["fpr_ci"][1],
                     seed=seed)
        store.record("verifier_bias", f"fnr_{name}", er["fnr"], unit="rate",
                     n=er["n"], ci_low=er["fnr_ci"][0], ci_high=er["fnr_ci"][1],
                     seed=seed)
        store.record("verifier_bias", f"coverage_{name}", er["coverage"],
                     unit="rate", n=er["n"])
        store.record("verifier_bias", f"observed_accuracy_{name}", p_hat,
                     unit="rate", n=len(survey), seed=seed)
        store.record("verifier_bias", f"corrected_accuracy_{name}", pi_hat,
                     unit="rate", n=len(survey), seed=seed,
                     ci_low=bs["lo"], ci_high=bs["hi"])
        store.record("verifier_bias", f"naive_bias_{name}", naive_bias, unit="bias",
                     n=len(survey))
        store.record("verifier_bias", f"corrected_bias_{name}", corrected_bias,
                     unit="bias", n=len(survey))
        store.record("verifier_bias", f"ci_covers_truth_{name}",
                     int(bs["lo"] <= pi_true <= bs["hi"]), unit="bool")

    store.record("verifier_bias", "survey_pi_true", pi_true, unit="rate",
                 n=len(survey), note="known by construction, used to validate the estimator")
    # 简历用的是被校正的有偏验证器, 这样校正前后的差异才有意义
    primary = "legacy_substring"
    store.record("verifier_bias", "observed_accuracy",
                 out["verifiers"][primary]["observed_accuracy"],
                 unit="rate", n=len(survey),
                 note="biased legacy-style verifier, alias for resume")
    store.record("verifier_bias", "corrected_accuracy",
                 out["verifiers"][primary]["corrected_accuracy"],
                 unit="rate", n=len(survey),
                 note="bias-corrected estimate, alias for resume")
    per = out["verifiers"][primary]["error_rates"]
    store.record("verifier_bias", "correction_denominator",
                 1.0 - per["fpr"] - per["fnr"], unit="abs", n=len(survey),
                 note="1 - FPR - FNR; a small denominator amplifies sampling error")

    # 三 verifier 潜类别估计: J=3 时 2^J-1=7 个自由频率恰好识别 1+2J=7 个参数,
    # 而 J=2 只有 3 个自由频率对应 5 个参数, 属欠识别, 不能用。
    raw = [[binary(VERIFIERS[name].verify(r["text"], r["gold"])) for r in survey]
           for name in ("strict_poly", "numeric_float", primary)]
    keep = [i for i in range(len(survey))
            if all(jl[i] is not None for jl in raw)]
    jls = [[jl[i] for i in keep] for jl in raw]
    em = latent_class_em(jls, seed=seed)
    pi_true_subset = (sum(survey[i]["truth"] for i in keep) / len(keep)
                      if keep else float("nan"))
    pi_true_full = pi_true
    out["latent_class"] = {
        "n_used": len(keep), "coverage": len(keep) / len(survey),
        "pi_hat": em["pi"], "pi_true_subset": pi_true_subset,
        "pi_true_full": pi_true_full,
        "abs_error": abs(em["pi"] - pi_true_subset),
        "selection_bias": pi_true_subset - pi_true_full,
        "se": em["se"], "sp": em["sp"],
        "n_free_stats": em["n_free_stats"], "n_params": em["n_params"],
        "identified": em["identified"],
    }
    lc = out["latent_class"]
    store.record("verifier_bias", "latent_class_pi", em["pi"], unit="rate",
                 n=len(keep), seed=seed, note="3-verifier EM without gold labels")
    store.record("verifier_bias", "latent_class_abs_error", lc["abs_error"],
                 unit="abs", n=len(keep))
    store.record("verifier_bias", "latent_class_coverage", lc["coverage"],
                 unit="rate", n=len(survey))
    store.record("verifier_bias", "latent_class_selection_bias", lc["selection_bias"],
                 unit="bias", n=len(keep),
                 note="UNVERIFIABLE is missing-not-at-random; dropping it biases the subset")
    store.record("verifier_bias", "latent_class_identified", int(lc["identified"]),
                 unit="bool", note="2^J-1 free cell frequencies vs 1+2J parameters")

    # 条件独立假设敏感性: 以概率 eps 把第三个 verifier 替换成第一个, 制造误差相关
    sens = {}
    for eps in (0.0, 0.2, 0.5):
        rng = random.Random(seed + 5)
        j3c = [jls[0][i] if rng.random() < eps else jls[2][i]
               for i in range(len(keep))]
        e = latent_class_em([jls[0], jls[1], j3c], restarts=4, seed=seed)
        sens[str(eps)] = e["pi"]
        store.record("verifier_bias", f"latent_class_pi_dependence_{eps}", e["pi"],
                     unit="rate", n=len(keep),
                     note="sensitivity to conditional-independence violation")
    out["dependence_sensitivity"] = sens
    out["dependence_sensitivity_spread"] = max(sens.values()) - min(sens.values())
    store.record("verifier_bias", "latent_class_dependence_spread",
                 out["dependence_sensitivity_spread"], unit="abs", n=len(keep))
    return out
