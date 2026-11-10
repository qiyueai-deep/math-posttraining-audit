"""参数化中文数学题生成器。

修复旧项目的致命问题 1: 旧数据只有 8 个手写模板, "变式" 只是追加编号,
train 80 条实际 4 道题, test 40 条实际 2 道题。

这里每道题有唯一的参数组, 答案由 oracle 独立计算, 并用第二条计算路径
交叉验证(cross_check), 生成器和判卷器不共用同一段代码。变体是真实的参数变化,
不是编号。测试集显式分出四个泛化子集: 未见参数 / 同义改写 / 反事实 / 未见题型。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import hashlib
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

from src.core.expressions import values_equal


def fmt(q: Fraction) -> str:
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _sf(v: int) -> str:
    return f"+{v}" if v >= 0 else str(v)


@dataclass(frozen=True)
class Family:
    name: str
    sample: Callable[[random.Random, bool], dict]
    oracle: Callable[[dict], str]
    cross_check: Callable[[dict, str], bool]
    render: Callable[[dict, int], str]
    rationale: Callable[[dict, str], str]
    counterfactual: Callable[[dict], dict | None]


# ------------------------------------------------------------------ 题型定义

def _linear(rng, ood):
    hi = 40 if ood else 9
    a = rng.randint(2, 9 if not ood else 15)
    b = rng.randint(-hi, hi)
    k = rng.randint(-hi, hi)
    return {"a": a, "b": b, "k": k, "c": a * k + b}


FAMILIES: list[Family] = [
    Family(
        "linear_equation", _linear,
        lambda p: fmt(Fraction(p["c"] - p["b"], p["a"])),
        lambda p, ans: p["a"] * Fraction(ans) + p["b"] == p["c"],
        lambda p, s: (f"解方程：{p['a']}x{_sf(p['b'])}={p['c']}。" if s == 0 else
                      f"已知 {p['a']}x{_sf(p['b'])}={p['c']}，求 x 的值。"),
        lambda p, a: f"移项得 {p['a']}x={p['c']}-{p['b']}={p['c']-p['b']}，两边除以 {p['a']} 得 x={a}。",
        lambda p: {
            "params": dict(p, b=-p["b"], k=p["k"] + 1, c=p["a"] * (p["k"] + 1) - p["b"]),
            "answer": str(p["k"] + 1),
            "question": f"解方程：{p['a']}x{_sf(-p['b'])}={p['a']*(p['k']+1)-p['b']}。",
            "rationale": (f"移项得 {p['a']}x={p['a']*(p['k']+1)-p['b']}-({-p['b']})"
                          f"={p['a']*(p['k']+1)}，故 x={p['k']+1}。"),
        },
    ),
    Family(
        "fraction_sum",
        lambda rng, ood: {"a": rng.randint(1, 25 if ood else 9),
                          "b": rng.randint(2, 20 if ood else 12),
                          "c": rng.randint(1, 25 if ood else 9),
                          "d": rng.randint(2, 20 if ood else 12)},
        lambda p: fmt(Fraction(p["a"], p["b"]) + Fraction(p["c"], p["d"])),
        lambda p, ans: Fraction(ans) == Fraction(p["a"] * p["d"] + p["c"] * p["b"], p["b"] * p["d"]),
        lambda p, s: (f"计算：\\frac{{{p['a']}}}{{{p['b']}}}+\\frac{{{p['c']}}}{{{p['d']}}}。" if s == 0 else
                      f"求 \\frac{{{p['a']}}}{{{p['b']}}} 与 \\frac{{{p['c']}}}{{{p['d']}}} 之和。"),
        lambda p, a: f"通分得 {p['a']*p['d']}/{p['b']*p['d']}+{p['c']*p['b']}/{p['d']*p['b']}，相加化简为 {a}。",
        None,
    ),
    Family(
        "quadratic_min",
        lambda rng, ood: {"p": 2 * rng.randint(-12 if ood else -5, 12 if ood else 5),
                          "q": rng.randint(-40 if ood else -20, 40 if ood else 20)},
        lambda p: fmt(Fraction(p["q"]) - Fraction(p["p"] * p["p"], 4)),
        lambda p, ans: Fraction(ans) == (Fraction(-p["p"], 2) ** 2 + p["p"] * Fraction(-p["p"], 2) + p["q"]),
        lambda p, s: (f"若 f(x)=x^2{_sf(p['p'])}x{_sf(p['q'])}，求其最小值。" if s == 0 else
                      f"二次函数 f(x)=x^2{_sf(p['p'])}x{_sf(p['q'])} 的最小值是多少？"),
        lambda p, a: f"配方得 f(x)=(x{_sf(-p['p']//2)})^2{a}，故最小值为 {a}。",
        lambda p: {
            "params": dict(p),
            "answer": fmt(Fraction(p["q"])),
            "question": f"设 f(x)=x^2{_sf(p['p'])}x{_sf(p['q'])}，求 f(0) 的值。",
            "rationale": f"令 x=0，得 f(0)={fmt(Fraction(p['q']))}。",
            "check": lambda np, ans: Fraction(ans) == np["q"],
        },
    ),
    Family(
        "arithmetic_seq",
        lambda rng, ood: {"a1": rng.randint(-30 if ood else -9, 30 if ood else 9),
                          "d": rng.randint(-15 if ood else -6, 15 if ood else 6),
                          "n": rng.randint(20 if ood else 5, 40 if ood else 15)},
        lambda p: str(p["a1"] + (p["n"] - 1) * p["d"]),
        lambda p, ans: int(ans) == _iter_seq(p["a1"], p["d"], p["n"]),
        lambda p, s: (f"一个等差数列首项为 {p['a1']}，公差为 {p['d']}，求第 {p['n']} 项。" if s == 0 else
                      f"已知等差数列 a_1={p['a1']}、d={p['d']}，计算 a_{p['n']}。"),
        lambda p, a: f"由 a_n=a_1+(n-1)d 得 a_{p['n']}={p['a1']}+{p['n']-1}×{p['d']}={a}。",
        lambda p: {
            "params": dict(p),
            "answer": str(p["n"] * (2 * p["a1"] + (p["n"] - 1) * p["d"]) // 2),
            "question": (f"一个等差数列首项为 {p['a1']}，公差为 {p['d']}，"
                         f"求前 {p['n']} 项的和。"),
            "rationale": (f"由 S_n=n(2a_1+(n-1)d)/2 得 "
                          f"S_{p['n']}={p['n']}*(2*{p['a1']}+{p['n']-1}*{p['d']})/2"
                          f"={p['n']*(2*p['a1']+(p['n']-1)*p['d'])//2}。"),
            "check": lambda np, ans: int(ans) == sum(np["a1"] + i * np["d"] for i in range(np["n"])),
        },
    ),
    Family(
        "combination",
        lambda rng, ood: {"n": rng.randint(41 if ood else 5, 70 if ood else 40),
                          "k": rng.randint(2, 6)},
        lambda p: str(math.comb(p["n"], p["k"])),
        lambda p, ans: int(ans) == _comb_rec(p["n"], p["k"]),
        lambda p, s: (f"从 {p['n']} 个球中取 {p['k']} 个，有多少种取法？" if s == 0 else
                      f"计算组合数 C({p['n']},{p['k']})。"),
        lambda p, a: f"C({p['n']},{p['k']})={p['n']}!/({p['k']}!({p['n']}-{p['k']})!)={a}。",
        None,
    ),
    Family(
        "definite_integral",
        lambda rng, ood: {"k": rng.randint(1, 30 if ood else 12),
                          "m": rng.randint(1, 12 if ood else 6)},
        lambda p: fmt(Fraction(p["k"], p["m"] + 1)),
        lambda p, ans: abs(float(Fraction(ans)) - _quad(p["k"], p["m"])) < 2e-3,
        lambda p, s: (f"求积分 ∫_0^1 {p['k']}x^{p['m']} dx。" if s == 0 else
                      f"计算定积分 ∫_0^1 {p['k']}x^{p['m']} dx 的值。"),
        lambda p, a: f"原函数为 {fmt(Fraction(p['k'], p['m']+1))}x^{p['m']+1}，代入上下限得 {a}。",
        None,
    ),
    Family(
        "weighted_expectation",
        lambda rng, ood: {"a": rng.randint(-20 if ood else -8, 20 if ood else 8),
                          "b": rng.randint(-20 if ood else -8, 20 if ood else 8),
                          "pn": rng.randint(1, 9), "pd": rng.randint(10, 30 if ood else 16)},
        lambda p: fmt(_expect(p)),
        lambda p, ans: Fraction(ans) == _expect_enum(p),
        lambda p, s: (f"随机变量 X 取 {p['a']} 的概率为 {p['pn']}/{p['pd']}，"
                      f"取 {p['b']} 的概率为 {fmt(1-Fraction(p['pn'],p['pd']))}，求 E[X]。" if s == 0 else
                      f"X 服从两点分布：P(X={p['a']})={p['pn']}/{p['pd']}，"
                      f"P(X={p['b']})={fmt(1-Fraction(p['pn'],p['pd']))}，求数学期望。"),
        lambda p, a: f"E[X]={p['pn']}/{p['pd']}×{p['a']}+{fmt(1-Fraction(p['pn'],p['pd']))}×{p['b']}={a}。",
        None,
    ),
    Family(
        "eigen_diagonal",
        lambda rng, ood: {"d1": rng.randint(-25 if ood else -8, 25 if ood else 8),
                          "d2": rng.randint(-25 if ood else -8, 25 if ood else 8)},
        lambda p: f"{p['d1']},{p['d2']}",
        lambda p, ans: sorted(map(int, ans.split(","))) == sorted([p["d1"], p["d2"]]),
        lambda p, s: (f"矩阵 [[{p['d1']},0],[0,{p['d2']}]] 的特征值是什么？" if s == 0 else
                      f"求对角矩阵 diag({p['d1']},{p['d2']}) 的全部特征值。"),
        lambda p, a: f"对角矩阵的特征多项式为 (λ-{p['d1']})(λ-{p['d2']})，特征值为 {a}。",
        None,
    ),
    Family(
        "geometric_sum",
        lambda rng, ood: {"r": rng.randint(13 if ood else 2, 18 if ood else 12),
                          "n": rng.randint(11 if ood else 2, 20 if ood else 10)},
        lambda p: str(sum(p["r"] ** i for i in range(p["n"]))),
        lambda p, ans: Fraction(ans) == Fraction(p["r"] ** p["n"] - 1, p["r"] - 1),
        lambda p, s: (f"求等比数列 1+{p['r']}+...+{p['r']}^{p['n']-1} 的和。" if s == 0 else
                      f"计算 ∑_{{i=0}}^{{{p['n']-1}}} {p['r']}^i。"),
        lambda p, a: f"由等比数列求和公式 ({p['r']}^{p['n']}-1)/({p['r']}-1)={a}。",
        None,
    ),
    Family(
        "lcm_product",
        lambda rng, ood: {"a": rng.randint(2, 60 if ood else 24),
                          "b": rng.randint(2, 60 if ood else 24)},
        lambda p: str(p["a"] * p["b"] // math.gcd(p["a"], p["b"])),
        lambda p, ans: int(ans) == _lcm_brute(p["a"], p["b"]),
        lambda p, s: (f"求 {p['a']} 与 {p['b']} 的最小公倍数。" if s == 0 else
                      f"计算 lcm({p['a']},{p['b']})。"),
        lambda p, a: f"gcd({p['a']},{p['b']})={math.gcd(p['a'],p['b'])}，故 lcm=a×b/gcd={a}。",
        None,
    ),
    Family(
        "derivative_at",
        lambda rng, ood: {"p": rng.randint(-20 if ood else -9, 20 if ood else 9),
                          "q": rng.randint(-20, 20),
                          "t": rng.randint(-15 if ood else -6, 15 if ood else 6)},
        lambda p: str(2 * p["t"] + p["p"]),
        lambda p, ans: Fraction(ans) == _diff_quotient(p),
        lambda p, s: (f"设 P(x)=x^2{_sf(p['p'])}x{_sf(p['q'])}，求 P'({p['t']})。" if s == 0 else
                      f"多项式 P(x)=x^2{_sf(p['p'])}x{_sf(p['q'])} 在 x={p['t']} 处的导数值是多少？"),
        lambda p, a: f"P'(x)=2x{_sf(p['p'])}，代入 x={p['t']} 得 {a}。",
        None,
    ),
    Family(
        "hypergeometric",
        lambda rng, ood: {"r": rng.randint(3, 40 if ood else 15),
                          "b": rng.randint(2, 40 if ood else 15)},
        lambda p: fmt(Fraction(math.comb(p["r"], 2), math.comb(p["r"] + p["b"], 2))),
        lambda p, ans: Fraction(ans) == Fraction(p["r"], p["r"] + p["b"]) * Fraction(p["r"] - 1, p["r"] + p["b"] - 1),
        lambda p, s: (f"袋中有 {p['r']} 个红球、{p['b']} 个蓝球，无放回连取 2 个，求两个都是红球的概率。" if s == 0 else
                      f"{p['r']} 红 {p['b']} 蓝，不放回抽 2 球，P(全红) 是多少？"),
        lambda p, a: f"C({p['r']},2)/C({p['r']+p['b']},2)={a}。",
        None,
    ),
]


# ------------------------------------------------------------------ 独立交叉验证路径

def _iter_seq(a1, d, n):
    v = a1
    for _ in range(n - 1):
        v += d
    return v


def _comb_rec(n, k):
    # 帕斯卡三角递推, 与 math.comb 完全不同的计算路径
    row = [0] * (k + 1)
    row[0] = 1
    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            row[j] += row[j - 1]
    return row[k]


def _quad(k, m, N=4000):
    h = 1.0 / N
    return sum(k * ((i + 0.5) * h) ** m for i in range(N)) * h


def _expect(p):
    q = Fraction(p["pn"], p["pd"])
    return q * p["a"] + (1 - q) * p["b"]


def _expect_enum(p):
    q = Fraction(p["pn"], p["pd"])
    return sum(v * w for v, w in [(p["a"], q), (p["b"], 1 - q)])


def _lcm_brute(a, b):
    v = max(a, b)
    while v % a or v % b:
        v += 1
    return v


def _diff_quotient(p):
    h = Fraction(1, 1000)
    t = Fraction(p["t"])
    f = lambda x: x * x + p["p"] * x + p["q"]
    return (f(t + h) - f(t - h)) / (2 * h)


# ------------------------------------------------------------------ 生成与切分

N_PER_FAMILY = 36
N_EXTRA = 8
N_TRAIN_FAMILIES = 9  # 前 9 个族参与训练, 后 3 个族只在测试集出现


def _stable_id(family: str, params: dict, variant: str) -> str:
    blob = json.dumps([family, sorted(params.items()), variant], ensure_ascii=False, sort_keys=True)
    return f"{family}_{hashlib.sha1(blob.encode()).hexdigest()[:10]}_{variant}"


def _make(fam: Family, params: dict, style: int, subset: str, split: str,
          variant: str = "base", question: str | None = None,
          answer: str | None = None, rationale: str | None = None,
          check: Callable[[dict, str], bool] | None = None) -> dict:
    ans = answer if answer is not None else fam.oracle(params)
    validator = check or fam.cross_check
    if not validator(params, ans):
        raise AssertionError(f"oracle cross-check failed: {fam.name} {params} -> {ans}")
    q = question if question is not None else fam.render(params, style)
    rationale = rationale if rationale is not None else fam.rationale(params, ans)
    return {
        "id": _stable_id(fam.name, params, variant),
        "family": fam.name,
        "params": params,
        "variant": variant,
        "style": style,
        "question": q + " 请给出可核验的简洁推导，并以「最终答案：」结尾。",
        "answer": ans,
        "rationale": rationale,
        "chosen": rationale + " 最终答案：" + ans,
        "subset": subset,
        "split": split,
        "messages": [
            {"role": "system", "content": "你是严谨的中文数学助手。先给出简洁推导，再以「最终答案：」给出结果。"},
            {"role": "user", "content": q + " 请给出可核验的简洁推导，并以「最终答案：」结尾。"},
            {"role": "assistant", "content": rationale + " 最终答案：" + ans},
        ],
    }


def generate(seed: int = 20251109, n_per_family: int = N_PER_FAMILY,
             n_extra: int = N_EXTRA, n_train_families: int = N_TRAIN_FAMILIES) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for fi, fam in enumerate(FAMILIES):
        unseen = fi >= n_train_families
        params_seen: set[str] = set()

        def fresh(ood=False):
            for _ in range(500):
                p = fam.sample(rng, ood)
                key = json.dumps(p, sort_keys=True)
                if key not in params_seen:
                    params_seen.add(key)
                    return p
            raise RuntimeError(f"cannot sample distinct params for {fam.name}")

        base = [fresh(False) for _ in range(n_per_family)]
        for i, p in enumerate(base):
            if unseen:
                rows.append(_make(fam, p, i % 2, "test_unseen_family", "test",
                                  variant=f"unseen{i}"))
            else:
                r = i % 20
                split, subset = ("train", "train_id") if r < 14 else (
                    ("valid", "valid_id") if r < 17 else ("test", "test_id"))
                rows.append(_make(fam, p, 0, subset, split, variant=f"base{i}"))

        if unseen:
            continue
        for j in range(n_extra):
            p = fresh(True)
            rows.append(_make(fam, p, 1, "test_ood_param", "test", variant=f"ood{j}"))
            p = fresh(False)
            rows.append(_make(fam, p, 1, "test_paraphrase", "test", variant=f"para{j}"))
            if fam.counterfactual is not None:
                p = fresh(False)
                cf = fam.counterfactual(p)
                rows.append(_make(fam, cf["params"], 0, "test_counterfactual", "test",
                                  variant=f"cf{j}", question=cf["question"],
                                  answer=cf["answer"], rationale=cf["rationale"],
                                  check=cf.get("check")))
    return rows


def write_dataset(rows: list[dict], out_dir: str | Path, seed: int) -> dict:
    from src.core.splits import leakage_check, manifest
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    check = leakage_check(rows)
    if not check["ok"]:
        raise AssertionError(f"leakage check failed: {check}")
    for name in ("train", "valid", "test"):
        part = [r for r in rows if r["split"] == name]
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    meta = manifest(rows, out, seed)
    return {"check": check, "manifest": meta}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data")
    ap.add_argument("--seed", type=int, default=20251109)
    a = ap.parse_args()
    rows = generate(a.seed)
    result = write_dataset(rows, a.output_dir, a.seed)
    print(json.dumps({"n": len(rows), **result}, ensure_ascii=False, indent=2))
