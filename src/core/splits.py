"""按题目族与参数组切分, 并做语义级泄露检查。

修复旧项目的致命问题 1 与 8: 旧 validate_dataset.py 只做字符串去重,
因为 "题目变式 0001" 这类编号后缀不同, 它报告 cross_split_overlap=0,
实际上 train 与 test 是同样 8 道手写题。这里以 (题型族, 参数组, 变体) 为
唯一签名, 并额外做去标点题干去重, 任何一处失败都直接 raise。
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
import re
from collections import Counter, defaultdict
from pathlib import Path

SUBSET_DEFINITION = {
    "train_id": "训练族, 分布内参数",
    "valid_id": "训练族, 分布内参数, 仅用于早停与选参",
    "test_id": "训练族, 分布内参数, 独立留出",
    "test_ood_param": "训练族, 参数超出训练范围",
    "test_paraphrase": "训练族, 同义改写题面",
    "test_counterfactual": "训练族, 改动条件且答案随之改变",
    "test_unseen_family": "训练阶段完全未出现的题型族",
}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def signature(row: dict) -> tuple[str, str, str]:
    return (row["family"], row["variant"],
            json.dumps(row["params"], sort_keys=True, ensure_ascii=False))


def canonical_question(q: str) -> str:
    return re.sub(r"[\s，,。.、：:；;（）()？?！!]", "", q)


def leakage_check(rows: list[dict]) -> dict:
    sig = [signature(r) for r in rows]
    dup_sig = len(sig) - len(set(sig))
    by_split: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for r, s in zip(rows, sig):
        by_split[r["split"]].add(s)
    overlaps = {}
    for a in ("train", "valid", "test"):
        for b in ("train", "valid", "test"):
            if a < b:
                overlaps[f"{a}_{b}"] = len(by_split[a] & by_split[b])
    qnorm = [canonical_question(r["question"]) for r in rows]
    dup_q = len(qnorm) - len(set(qnorm))
    by_subset = Counter(r["subset"] for r in rows)
    unseen_in_train = [r["id"] for r in rows
                       if r["subset"] == "test_unseen_family" and r["split"] != "test"]
    train_families = sorted({r["family"] for r in rows if r["split"] == "train"})
    unseen_families = sorted({r["family"] for r in rows if r["subset"] == "test_unseen_family"})
    family_leak = sorted(set(train_families) & set(unseen_families))

    checks = {
        "duplicate_parameter_signatures": dup_sig,
        "duplicate_canonical_questions": dup_q,
        "cross_split_parameter_overlap": sum(overlaps.values()),
        "cross_split_pairwise": overlaps,
        "unseen_family_rows_outside_test": len(unseen_in_train),
        "family_in_both_train_and_unseen": family_leak,
        "empty_splits": [s for s in ("train", "valid", "test") if not by_split[s]],
    }
    ok = (dup_sig == 0 and dup_q == 0 and sum(overlaps.values()) == 0
          and not unseen_in_train and not family_leak and not checks["empty_splits"])
    return {
        "ok": ok,
        "n_rows": len(rows),
        "n_unique_signatures": len(set(sig)),
        "n_unique_questions": len(set(qnorm)),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_subset": dict(by_subset),
        "train_families": train_families,
        "unseen_families": unseen_families,
        "checks": checks,
    }


def manifest(rows: list[dict], out_dir: str | Path, seed: int) -> dict:
    out = Path(out_dir)
    check = leakage_check(rows)
    if not check["ok"]:
        raise AssertionError(f"leakage check failed: {check['checks']}")
    files = {}
    for name in ("train", "valid", "test"):
        p = out / f"{name}.jsonl"
        files[f"{name}.jsonl"] = {"sha256": file_sha256(p),
                                  "rows": sum(1 for _ in p.open(encoding="utf-8"))}
    meta = {
        "seed": seed,
        "generator": "src/core/data_gen.py",
        "split_rule": "group by (family, parameter tuple); 3 unseen families held out",
        "subset_definition": SUBSET_DEFINITION,
        "n_rows": len(rows),
        "n_families": len({r["family"] for r in rows}),
        "leakage": check,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return meta
