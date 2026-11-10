"""子贡献 2：伪重复与配对推断。

旧项目的 "40 条独立测试集" 实际是 2 道题各重复 20 次(仅编号不同)。
把 20 个伪重复当 iid 会严重低估方差。这里以题目(或题型族)为聚类单位,
给出簇内相关 ICC、设计效应、有效样本量、聚类 bootstrap 区间、配对 McNemar
精确检验、Holm 校正与功效曲线。

所有计算基于真实逐条预测记录, 不假设模型结果。
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
import statistics
from collections import defaultdict
from pathlib import Path

_ND = statistics.NormalDist()


def icc_anova(clusters: dict[str, list[int]]) -> float:
    """单因素随机效应 ANOVA 估计簇内相关, 退化情形(组内方差为 0)也能处理。"""
    ks = [v for v in clusters.values() if v]
    k = len(ks)
    if k < 2:
        return 0.0
    n = sum(len(v) for v in ks)
    grand = sum(sum(v) for v in ks) / n
    msb = sum(len(v) * (statistics.mean(v) - grand) ** 2 for v in ks) / (k - 1)
    within = sum(sum((x - statistics.mean(v)) ** 2 for x in v) for v in ks)
    df_w = n - k
    msw = within / df_w if df_w > 0 else 0.0
    m0 = (n - sum(len(v) ** 2 for v in ks) / n) / (k - 1) if k > 1 else 1.0
    denom = msb + (m0 - 1) * msw
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (msb - msw) / denom))


def design_effect(rho: float, mean_cluster_size: float) -> float:
    return 1.0 + (mean_cluster_size - 1.0) * rho


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = _ND.inv_cdf(1 - alpha / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def iid_paired_ci(diffs: list[float], alpha: float = 0.05) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"))
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n)
    z = _ND.inv_cdf(1 - alpha / 2)
    return (m - z * se, m + z * se)


def cluster_bootstrap_ci(clusters: dict[str, list[tuple[int, int]]], stat,
                         B: int = 10000, seed: int = 20251109,
                         alpha: float = 0.05) -> dict:
    keys = list(clusters)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        picked = [keys[rng.randrange(len(keys))] for _ in keys]
        sample = [row for key in picked for row in clusters[key]]
        vals.append(stat(sample))
    vals.sort()
    lo = vals[int((alpha / 2) * B)]
    hi = vals[min(B - 1, int((1 - alpha / 2) * B))]
    return {"lo": lo, "hi": hi, "B": B,
            "point": stat([row for v in clusters.values() for row in v]),
            "n_clusters": len(keys)}


def mcnemar_exact(b: int, c: int) -> float:
    """精确二项检验, b/c 为两个方向的不一致配对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def required_clusters(delta: float, sigma_cluster: float,
                      alpha: float = 0.05, power: float = 0.8) -> int:
    """给定簇均值的标准差, 求检出 delta 所需的最少独立簇数。"""
    if delta <= 0 or sigma_cluster <= 0:
        return 0
    z_a = _ND.inv_cdf(1 - alpha / 2)
    z_b = _ND.inv_cdf(power)
    return math.ceil(((z_a + z_b) * sigma_cluster / delta) ** 2)


def power_at(K: int, delta: float, sigma_cluster: float, alpha: float = 0.05) -> float:
    if K <= 0 or sigma_cluster <= 0:
        return 0.0
    se = sigma_cluster / math.sqrt(K)
    z_a = _ND.inv_cdf(1 - alpha / 2)
    return _ND.cdf(delta / se - z_a) + _ND.cdf(-delta / se - z_a)


def _paired_from_legacy(items: dict, model_a: str, model_b: str,
                        field: str = "symbolic") -> tuple[dict, list[float], int, int]:
    a = {x["id"]: x for x in items[model_a]}
    b = {x["id"]: x for x in items[model_b]}
    clusters: dict[str, list[tuple[int, int]]] = defaultdict(list)
    diffs, discord_b, discord_c = [], 0, 0
    for key in sorted(set(a) & set(b)):
        va, vb = int(a[key][field]), int(b[key][field])
        clusters[a[key]["type"]].append((va, vb))
        diffs.append(vb - va)
        if va == 1 and vb == 0:
            discord_c += 1
        elif va == 0 and vb == 1:
            discord_b += 1
    return dict(clusters), diffs, discord_b, discord_c


def run_from_predictions(store, predictions: dict[str, list[dict]], seed: int,
                         B: int = 10000) -> dict:
    """模型阶段完成后, 对真实逐条预测做以题型族为聚类单位的配对推断。

    与 run() 的区别: 这里不读旧项目文件, 输入是 evaluate.py 写出的
    predictions_*.jsonl。verdict=unverifiable 单独保留, 不并入 correct。
    """
    names = [n for n in ("base", "sft", "dpo") if n in predictions]
    result = {}
    stat = lambda rows: (sum(b - a for a, b in rows) / len(rows)) if rows else 0.0
    pvals = []
    for ma, mb in ((names[i], names[i + 1]) for i in range(len(names) - 1)):
        a = {x["id"]: x for x in predictions[ma]}
        b = {x["id"]: x for x in predictions[mb]}
        clusters: dict[str, list[tuple[int, int]]] = defaultdict(list)
        diffs, db, dc = [], 0, 0
        for key in sorted(set(a) & set(b)):
            va = int(a[key]["verdict"] == "correct")
            vb = int(b[key]["verdict"] == "correct")
            clusters[a[key]["family"]].append((va, vb))
            diffs.append(vb - va)
            if va == 1 and vb == 0:
                dc += 1
            elif va == 0 and vb == 1:
                db += 1
        if not diffs:
            continue
        by_cluster = {k: [x[1] for x in v] for k, v in clusters.items()}
        rho = icc_anova(by_cluster)
        m = len(diffs) / len(clusters)
        deff = design_effect(rho, m)
        bs = cluster_bootstrap_ci(clusters, stat, B=B, seed=seed)
        iid = iid_paired_ci(diffs)
        p = mcnemar_exact(db, dc)
        pvals.append(p)
        key = f"{ma}_to_{mb}"
        result[key] = {
            "n": len(diffs), "n_clusters": len(clusters), "icc": rho,
            "design_effect": deff, "effective_n": len(diffs) / deff,
            "iid_ci": iid, "cluster_ci": (bs["lo"], bs["hi"]),
            "point": bs["point"], "mcnemar_p": p,
        }
        store.record("model_stage", f"cluster_ci_low_{key}", bs["lo"], unit="diff",
                     n=len(clusters), seed=seed)
        store.record("model_stage", f"cluster_ci_high_{key}", bs["hi"], unit="diff",
                     n=len(clusters), seed=seed)
        store.record("model_stage", f"effective_n_{key}", len(diffs) / deff,
                     unit="rows", n=len(diffs))
        store.record("model_stage", f"mcnemar_p_{key}", p, unit="p", n=len(diffs))
    if pvals:
        adjusted = holm(pvals)
        for i, key in enumerate(result):
            result[key]["holm_p"] = adjusted[i]
            store.record("model_stage", f"holm_p_{key}", adjusted[i], unit="p")
    return result


def run(store, legacy_root: Path, new_rows: list[dict], seed: int,
        B: int = 10000) -> dict:
    result = {"legacy": None, "new_design": None}
    p = legacy_root / "reports" / "symbolic_full_40.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data["items"]
        clusters, diffs, db, dc = _paired_from_legacy(items, "base", "sft")
        n = len(diffs)
        K = len(clusters)
        m = n / K
        by_cluster = {k: [x[1] for x in v] for k, v in clusters.items()}
        rho = icc_anova(by_cluster)
        deff = design_effect(rho, m)
        eff_n = n / deff
        stat = lambda rows: (sum(b - a for a, b in rows) / len(rows)) if rows else 0.0
        bs = cluster_bootstrap_ci(clusters, stat, B=B, seed=seed)
        iid = iid_paired_ci(diffs)
        p_mcnemar = mcnemar_exact(db, dc)
        # 三个两两比较做 Holm 校正
        ps = [p_mcnemar]
        for ma, mb in (("sft", "dpo"), ("base", "dpo")):
            _, _, b2, c2 = _paired_from_legacy(items, ma, mb)
            ps.append(mcnemar_exact(b2, c2))
        adjusted = holm(ps)
        cluster_means = [statistics.mean([b - a for a, b in v]) for v in clusters.values()]
        sigma_cluster = statistics.stdev(cluster_means) if len(cluster_means) > 1 else 0.0
        need_K = required_clusters(0.05, sigma_cluster) if sigma_cluster > 0 else 0
        result["legacy"] = {
            "n": n, "n_clusters": K, "mean_cluster_size": m, "icc": rho,
            "design_effect": deff, "effective_n": eff_n,
            "iid_ci": iid, "cluster_ci": (bs["lo"], bs["hi"]),
            "point_difference": bs["point"], "mcnemar_p": p_mcnemar,
            "holm_adjusted_p": adjusted, "sigma_cluster": sigma_cluster,
            "clusters_needed_for_5pp": need_K,
            "power_curve": {str(k): power_at(k, 0.05, sigma_cluster)
                            for k in (2, 5, 10, 20, 50, 100)},
        }
        r = result["legacy"]
        store.record("cluster_inference", "legacy_n", n, unit="rows")
        store.record("cluster_inference", "n_clusters", K, unit="clusters")
        store.record("cluster_inference", "icc", rho, unit="rho", n=n)
        store.record("cluster_inference", "design_effect", deff, unit="deff", n=n)
        store.record("cluster_inference", "effective_n", eff_n, unit="rows", n=n)
        store.record("cluster_inference", "iid_ci_low", iid[0], unit="diff", n=n)
        store.record("cluster_inference", "iid_ci_high", iid[1], unit="diff", n=n)
        store.record("cluster_inference", "cluster_ci_low", bs["lo"], unit="diff",
                     n=K, seed=seed, ci_low=bs["lo"], ci_high=bs["hi"])
        store.record("cluster_inference", "cluster_ci_high", bs["hi"], unit="diff",
                     n=K, seed=seed)
        store.record("cluster_inference", "point_difference_sft_minus_base",
                     bs["point"], unit="diff", n=n)
        store.record("cluster_inference", "mcnemar_p_base_sft", p_mcnemar, unit="p", n=n)
        store.record("cluster_inference", "holm_adjusted_min_p", min(adjusted), unit="p")
        store.record("cluster_inference", "clusters_needed_for_5pp", need_K,
                     unit="clusters", note="80% power, two-sided alpha=0.05")
        for k, v in r["power_curve"].items():
            store.record("cluster_inference", f"power_at_K{k}", v, unit="power")

    # 新数据集的设计层面情景分析: 用实测 rho 做保守情景, 明确标注为 scenario
    fam = defaultdict(int)
    for row in new_rows:
        if row["split"] == "test":
            fam[row["family"]] += 1
    if fam:
        sizes = list(fam.values())
        n_new, K_new = sum(sizes), len(sizes)
        m_new = n_new / K_new
        scen = {}
        for label, rho in (("independence", 0.0), ("moderate", 0.3),
                           ("legacy_measured", result["legacy"]["icc"] if result["legacy"] else 1.0)):
            d = design_effect(rho, m_new)
            scen[label] = {"rho": rho, "deff": d, "effective_n": n_new / d}
            store.record("cluster_inference", f"new_effective_n_{label}", n_new / d,
                         unit="rows", n=n_new, note="design-level scenario, not a model result")
        result["new_design"] = {"n_test": n_new, "n_family_clusters": K_new,
                                "mean_cluster_size": m_new, "scenarios": scen}
        store.record("cluster_inference", "new_n_test", n_new, unit="rows")
        store.record("cluster_inference", "new_n_family_clusters", K_new, unit="clusters")
    return result
