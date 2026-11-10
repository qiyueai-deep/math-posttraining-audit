"""旧项目假象的真实复算审计。

所有结论都从旧项目的实际文件读出, 不引用上一轮对话里的记忆数字。
审计本身是子贡献的实证案例: 用修好的验证器和统计口径重新判一遍旧结果。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import json
import re
from collections import Counter
from pathlib import Path

from src.core.expressions import extract_answer
from src.core.verifier import CORRECT, VERIFIERS


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def audit_dataset(legacy_root: Path) -> dict:
    p = legacy_root / "data" / "test.jsonl"
    if not p.exists():
        return {}
    rows = _rows(p)
    qs = [re.sub(r"（题目变式 \d+）", "", r["question"]).strip() for r in rows]
    answers = Counter(r["answer"] for r in rows)
    return {
        "n_test_rows": len(rows),
        "n_unique_questions": len(set(qs)),
        "n_unique_answers": len(answers),
        "pseudo_replication_factor": len(rows) / max(1, len(set(qs))),
        "answer_counts": dict(answers),
    }


def audit_preferences(legacy_root: Path) -> dict:
    p = legacy_root / "data" / "preferences.jsonl"
    if not p.exists():
        return {}
    rows = _rows(p)
    claimed_wrong = [r for r in rows if r.get("preference_type") == "wrong_final_answer"]
    actually_wrong = 0
    unverifiable = 0
    for r in claimed_wrong:
        gold = extract_answer(r["chosen"])
        if gold is None:
            unverifiable += 1
            continue
        v = VERIFIERS["strict_poly"].verify(r["rejected"], gold)
        if v.verdict != CORRECT:
            actually_wrong += 1
    length_diffs = [r["chosen_chars"] - r["rejected_chars"] for r in rows]
    return {
        "n_pairs": len(rows),
        "n_claimed_wrong_final_answer": len(claimed_wrong),
        "n_actually_mathematically_wrong": actually_wrong,
        "n_unverifiable_gold": unverifiable,
        "true_error_rate_in_claimed_wrong": (actually_wrong / len(claimed_wrong)
                                             if claimed_wrong else 0.0),
        "length_diff_all_positive": all(d > 0 for d in length_diffs),
        "length_diff_min": min(length_diffs) if length_diffs else None,
        "length_diff_max": max(length_diffs) if length_diffs else None,
    }


def audit_generative(legacy_root: Path) -> dict:
    p = legacy_root / "reports" / "symbolic_full_40.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data["items"]
    agree = sum(1 for name in items for it in items[name]
                if it["exact"] == it["symbolic"])
    total = sum(len(v) for v in items.values())
    old_true = sum(1 for name in items for it in items[name] if it["symbolic"])
    new_correct = new_unverifiable = new_incorrect = 0
    false_positive = 0
    for name, rows in items.items():
        for it in rows:
            v = VERIFIERS["strict_poly"].verify(it["pred"], it["gold"])
            if v.verdict == CORRECT:
                new_correct += 1
            elif v.verdict == "unverifiable":
                new_unverifiable += 1
            else:
                new_incorrect += 1
            if it["symbolic"] and v.verdict != CORRECT:
                false_positive += 1
    return {
        "n_predictions": total,
        "exact_equals_symbolic": agree,
        "exact_equals_symbolic_rate": agree / total if total else 0.0,
        "old_reported_correct": old_true,
        "strict_verifier_correct": new_correct,
        "strict_verifier_unverifiable": new_unverifiable,
        "strict_verifier_incorrect": new_incorrect,
        "old_false_positives": false_positive,
        "old_false_positive_rate": false_positive / old_true if old_true else 0.0,
    }


def audit_metric_conflicts(legacy_root: Path) -> dict:
    """从三个独立报告里抽同一个 SFT+DPO 的 PPL, 量化互相矛盾。"""
    found = {}
    md = legacy_root / "reports" / "qwen3_real_metrics.md"
    if md.exists():
        for line in md.read_text(encoding="utf-8").splitlines():
            if "SFT+DPO" in line and "|" in line:
                cells = [c.strip() for c in line.split("|")]
                nums = [c for c in cells if re.fullmatch(r"\d+\.\d+", c)]
                if len(nums) >= 2:
                    found["qwen3_real_metrics.md"] = float(nums[1])
    tf = legacy_root / "reports" / "teacher_forced_metrics.json"
    if tf.exists():
        found["teacher_forced_metrics.json"] = json.loads(
            tf.read_text(encoding="utf-8"))["dpo"]["perplexity"]
    bs = legacy_root / "reports" / "beta_sweep.json"
    if bs.exists():
        rows = json.loads(bs.read_text(encoding="utf-8"))
        for r in rows:
            if abs(r["beta"] - 0.1) < 1e-9 and "teacher_forced" in r:
                found["beta_sweep.json(beta=0.1)"] = r["teacher_forced"]["perplexity"]
    vals = list(found.values())
    return {
        "sources": found,
        "n_distinct_values": len(set(round(v, 2) for v in vals)),
        "spread": max(vals) - min(vals) if vals else None,
    }


def audit_code_patterns(legacy_root: Path) -> dict:
    def read(p):
        f = legacy_root / p
        return f.read_text(encoding="utf-8") if f.exists() else ""
    tf = read("src/evaluate_teacher_forced.py")
    dpo = read("src/train_dpo.py")
    ev = read("src/evaluate.py")
    return {
        "teacher_forced_has_prompt_mask": "-100" in tf,
        "dpo_uses_ref_model_none": "ref_model=None" in dpo,
        "evaluator_uses_substring_match": "in norm(pred)" in ev,
        "evaluator_silent_fallback": "return exact(pred,ans)" in ev,
    }


def run(store, legacy_root: Path) -> dict:
    out = {
        "dataset": audit_dataset(legacy_root),
        "preferences": audit_preferences(legacy_root),
        "generative": audit_generative(legacy_root),
        "metrics": audit_metric_conflicts(legacy_root),
        "code": audit_code_patterns(legacy_root),
    }
    d, pr, g, m, c = (out["dataset"], out["preferences"], out["generative"],
                      out["metrics"], out["code"])
    if d:
        store.record("data_audit", "legacy_test_rows", d["n_test_rows"], unit="rows")
        store.record("data_audit", "legacy_unique_questions", d["n_unique_questions"],
                     unit="questions")
        store.record("data_audit", "legacy_pseudo_replication",
                     d["pseudo_replication_factor"], unit="x")
    if pr:
        store.record("data_audit", "legacy_claimed_wrong_pairs",
                     pr["n_claimed_wrong_final_answer"], unit="pairs")
        store.record("data_audit", "legacy_actually_wrong_pairs",
                     pr["n_actually_mathematically_wrong"], unit="pairs")
        store.record("data_audit", "legacy_length_diff_all_positive",
                     int(pr["length_diff_all_positive"]), unit="bool")
    if g:
        store.record("data_audit", "legacy_exact_equals_symbolic_rate",
                     g["exact_equals_symbolic_rate"], unit="rate", n=g["n_predictions"])
        store.record("legacy_audit", "old_reported_correct", g["old_reported_correct"],
                     unit="count", n=g["n_predictions"])
        store.record("legacy_audit", "strict_verifier_correct", g["strict_verifier_correct"],
                     unit="count", n=g["n_predictions"])
        store.record("legacy_audit", "strict_verifier_unverifiable",
                     g["strict_verifier_unverifiable"], unit="count", n=g["n_predictions"])
        store.record("legacy_audit", "old_false_positives", g["old_false_positives"],
                     unit="count", n=g["old_reported_correct"])
        store.record("legacy_audit", "old_false_positive_rate",
                     g["old_false_positive_rate"], unit="rate")
    if m and m["sources"]:
        store.record("legacy_audit", "ppl_sources_conflicting",
                     m["n_distinct_values"], unit="distinct values",
                     note="same SFT+DPO model reported with different PPL")
        store.record("legacy_audit", "ppl_spread", m["spread"], unit="PPL")
    if c:
        store.record("legacy_audit", "old_prompt_mask_present",
                     int(c["teacher_forced_has_prompt_mask"]), unit="bool")
        store.record("legacy_audit", "old_ref_model_none",
                     int(c["dpo_uses_ref_model_none"]), unit="bool")
        store.record("legacy_audit", "old_substring_matcher",
                     int(c["evaluator_uses_substring_match"]), unit="bool")
    return out
