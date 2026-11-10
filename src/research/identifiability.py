"""子贡献 1：偏好信号的可识别性。

形式化陈述与证明见 THEORY.md。核心是：若存在表面特征 S 使偏好标签 Y 被 S
确定性解释，且真实质量 Q 与 Y 在给定 S 后独立，则 DPO 损失在 Q 方向上不可识别，
reward accuracy=1.0 只能说明模型学到了 S。

这里给出三样可复现的东西：
  1. 只用表面特征(长度/格式/LaTeX/数字/标点)的分类器, 交叉验证 AUC。
  2. shortcut_index = 2*AUC-1, 以及去混淆后的残差可识别性。
  3. DPO 损失在随机重排 Q 下不变的数值佐证, 以及 beta 恒等式的量化证据。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import json
import math
import random
import re
from pathlib import Path

SURFACE_KEYS = ("len_diff", "format_diff", "latex_diff", "digit_diff",
                "punct_diff", "newline_diff")


def surface_features(chosen: str, rejected: str) -> dict[str, float]:
    def cnt(s, pat):
        return len(re.findall(pat, s))
    return {
        "len_diff": len(chosen) - len(rejected),
        "format_diff": int("最终答案" in chosen) - int("最终答案" in rejected),
        "latex_diff": cnt(chosen, r"\\(?:frac|boxed|sqrt)") - cnt(rejected, r"\\(?:frac|boxed|sqrt)"),
        "digit_diff": cnt(chosen, r"\d") - cnt(rejected, r"\d"),
        "punct_diff": cnt(chosen, r"[，。；、]") - cnt(rejected, r"[，。；、]"),
        "newline_diff": chosen.count("\n") - rejected.count("\n"),
    }


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def auc_score(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U, 用平均秩正确处理 ties。"""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r1 = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (r1 - pos * (pos + 1) / 2.0) / (pos * neg)


def _standardize(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    d = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(d)]
    stds = []
    for j in range(d):
        v = sum((row[j] - means[j]) ** 2 for row in X) / len(X)
        stds.append(math.sqrt(v) or 1.0)
    return ([[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X],
            means, stds)


def _fit_logistic(X: list[list[float]], y: list[int], iters: int = 200,
                  lr: float = 0.3, l2: float = 1e-3) -> tuple[list[float], float]:
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = sum(wj * xj for wj, xj in zip(w, xi)) + b
            e = _sigmoid(z) - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def cv_auc(pairs: list[dict], seed: int = 20251109, folds: int = 5) -> dict:
    """每个偏好对生成正负两个视角, 按 pair 分组做交叉验证, 防止同一对跨折泄漏。"""
    X, y, group = [], [], []
    for i, p in enumerate(pairs):
        f = [surface_features(p["chosen"], p["rejected"])[k] for k in SURFACE_KEYS]
        X.append(f)
        y.append(1)
        group.append(i)
        X.append([-v for v in f])
        y.append(0)
        group.append(i)
    Xs, _, _ = _standardize(X)
    idx = list(range(len(y)))
    rng = random.Random(seed)
    pair_ids = list(range(len(pairs)))
    rng.shuffle(pair_ids)
    fold_of = {}
    for k, pid in enumerate(pair_ids):
        fold_of[pid] = k % folds
    oof = [None] * len(y)
    for k in range(folds):
        tr = [i for i in idx if fold_of[group[i]] != k]
        te = [i for i in idx if fold_of[group[i]] == k]
        if not tr or not te:
            continue
        w, b = _fit_logistic([Xs[i] for i in tr], [y[i] for i in tr])
        for i in te:
            oof[i] = sum(wj * xj for wj, xj in zip(w, Xs[i])) + b
    keep = [i for i in idx if oof[i] is not None]
    auc = auc_score([oof[i] for i in keep], [y[i] for i in keep])
    single = {}
    for j, key in enumerate(SURFACE_KEYS):
        s = [X[i][j] for i in keep]
        single[key] = auc_score(s, [y[i] for i in keep])
    return {
        "cv_auc": auc,
        "shortcut_index": max(0.0, 2 * auc - 1),
        "residual_identifiability": 1 - max(0.0, 2 * auc - 1),
        "best_single_feature": max(single, key=single.get),
        "best_single_auc": max(single.values()),
        "single_feature_auc": single,
        "n_pairs": len(pairs),
    }


def dpo_loss_invariance(beta: float = 0.1, n: int = 200, seed: int = 20251109) -> dict:
    """数值佐证: 当 log-ratio 差只由表面特征 S 决定时, DPO 损失不含 Q。

    对同一批 (S, delta) 随机重排真实质量 Q 的分配, 损失应当逐位不变。
    同时给出反例: 若 delta 也依赖 Q, 损失会随 Q 变化。
    """
    rng = random.Random(seed)
    S = [rng.choice([-1.0, 1.0]) for _ in range(n)]
    delta = [2.0 * s for s in S]

    def loss(d):
        return sum(-math.log(_sigmoid(beta * v)) for v in d) / n

    base = loss(delta)
    q = [rng.choice([0, 1]) for _ in range(n)]
    changes = []
    for _ in range(50):
        rng.shuffle(q)
        changes.append(abs(loss(delta) - base))
    coupled = [2.0 * S[i] + 3.0 * q[i] for i in range(n)]
    coupled_changes = []
    for _ in range(50):
        rng.shuffle(q)
        d = [2.0 * S[i] + 3.0 * q[i] for i in range(n)]
        coupled_changes.append(abs(loss(d) - loss(coupled)))
    return {
        "max_abs_change_under_Q_permute": max(changes),
        "max_abs_change_when_delta_depends_on_Q": max(coupled_changes),
        "beta": beta, "n": n,
    }


def beta_identity(legacy_root: Path) -> dict | None:
    """量化旧 beta 扫描是恒等式而非发现: margin/beta 的变异系数应接近 0。"""
    p = legacy_root / "reports" / "beta_sweep.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    ratios = [r["mean_reward_margin_tail"] / r["beta"] for r in rows]
    losses = [r["teacher_forced"]["teacher_forced_loss"] for r in rows
              if "teacher_forced" in r]
    mean = sum(ratios) / len(ratios)
    var = sum((v - mean) ** 2 for v in ratios) / len(ratios)
    cv = math.sqrt(var) / mean if mean else 0.0
    return {
        "margin_over_beta_cv": cv,
        "margin_over_beta_mean": mean,
        "test_loss_range": max(losses) - min(losses) if losses else None,
        "n_betas": len(rows),
    }


def run(store, groups: dict[str, list[dict]], legacy_root: Path, seed: int) -> dict:
    out = {"groups": {}, "invariance": None, "beta_identity": None, "legacy": None}
    for name, pairs in groups.items():
        r = cv_auc(pairs, seed)
        r["true_math_error_rate"] = (sum(p["contains_true_math_error"] for p in pairs)
                                     / len(pairs) if pairs else 0.0)
        r["mean_length_diff"] = (sum(p["length_diff"] for p in pairs) / len(pairs)
                                 if pairs else 0.0)
        out["groups"][name] = r
        store.record("identifiability", f"surface_auc_{name}", r["cv_auc"],
                     unit="AUC", n=r["n_pairs"], seed=seed,
                     note="5-fold pair-grouped CV, surface features only",
                     source="preferences_" + name + ".jsonl")
        store.record("identifiability", f"shortcut_index_{name}", r["shortcut_index"],
                     unit="SI", n=r["n_pairs"], seed=seed)
        store.record("identifiability", f"true_error_rate_{name}",
                     r["true_math_error_rate"], unit="rate", n=len(pairs), seed=seed)

    inv = dpo_loss_invariance(seed=seed)
    out["invariance"] = inv
    store.record("identifiability", "max_abs_change_under_Q_permute",
                 inv["max_abs_change_under_Q_permute"], unit="abs",
                 n=inv["n"], seed=seed,
                 note="DPO loss invariance when the log-ratio depends only on S")
    store.record("identifiability", "max_abs_change_coupled_Q",
                 inv["max_abs_change_when_delta_depends_on_Q"], unit="abs",
                 n=inv["n"], seed=seed)

    bi = beta_identity(legacy_root)
    out["beta_identity"] = bi
    if bi:
        store.record("identifiability", "margin_over_beta_cv", bi["margin_over_beta_cv"],
                     unit="CV", n=bi["n_betas"],
                     note="~0 means reported margin growth is pure beta scaling")
        store.record("identifiability", "beta_sweep_test_loss_range",
                     bi["test_loss_range"], unit="nats", n=bi["n_betas"])

    # 旧项目 240 条偏好对也跑同一套诊断, 作为真实案例
    lp = legacy_root / "data" / "preferences.jsonl"
    if lp.exists():
        legacy_pairs = []
        for l in lp.read_text(encoding="utf-8").splitlines():
            d = json.loads(l)
            legacy_pairs.append({"chosen": d["chosen"], "rejected": d["rejected"],
                                 "length_diff": d["chosen_chars"] - d["rejected_chars"],
                                 "contains_true_math_error": False})
        lr = cv_auc(legacy_pairs, seed)
        out["legacy"] = lr
        store.record("identifiability", "surface_auc_legacy", lr["cv_auc"],
                     unit="AUC", n=lr["n_pairs"], seed=seed,
                     source=str(lp))
    return out
