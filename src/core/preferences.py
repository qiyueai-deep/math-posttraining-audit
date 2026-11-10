"""去混淆偏好对构造。

修复旧项目的致命问题 4 与 7:
  旧 bad_answer() 用 ans.replace("4","5",1), 而 train 的答案是 21/10/1/1,2,
  替换全部无效 -> 240 条偏好对里没有任何一条包含真正的数学错误。
  同时 chosen 恒比 rejected 长 18-28 字符, reward accuracy=1.0 是长度线索的平凡解。
  旧 mine_errors.py 把 max_new_tokens=64 的解码截断当成数学错误, 且 rejected 更长,
  偏好方向被反转。

这里显式构造三组对照:
  shortcut        : 复刻旧缺陷, 答案其实正确, 只有长度/格式线索 (对照组)
  deconfounded    : 长度严格相等、格式完全相同, 只有数学内容错误 (处理组)
  length_reversed : rejected 更长但答案错误, 故意反转长度线索 (反事实组)
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

from src.core.expressions import ParseError, values_equal

GROUPS = ("shortcut", "deconfounded", "length_reversed")


def wrong_answer(ans: str) -> str | None:
    """生成与 ans 等长、可解析、且数学上不等的错误答案。"""
    if re.fullmatch(r"-?\d+", ans):
        d = int(ans[-1])
        cand = ans[:-1] + str((d + 1) % 10)
    elif "/" in ans and re.fullmatch(r"-?\d+/\d+", ans):
        head, den = ans.rsplit("/", 1)
        cand = head + "/" + den[:-1] + str((int(den[-1]) + 1) % 10)
    elif "," in ans:
        parts = ans.split(",")
        last = parts[-1]
        if not re.fullmatch(r"-?\d+", last):
            return None
        parts[-1] = last[:-1] + str((int(last[-1]) + 1) % 10)
        cand = ",".join(parts)
    else:
        return None
    if cand == ans:
        return None
    try:
        if values_equal(ans, cand):
            return None
    except ParseError:
        return None
    return cand


def _row(r: dict, group: str, rejected: str, wrong: str | None,
         true_error: bool, label_claim: str, idx: int) -> dict:
    return {
        "id": f"{r['id']}_{group}_{idx}",
        "prompt": r["question"],
        "chosen": r["chosen"],
        "rejected": rejected,
        "group": group,
        "family": r["family"],
        "answer": r["answer"],
        "wrong_answer": wrong,
        "chosen_chars": len(r["chosen"]),
        "rejected_chars": len(rejected),
        "length_diff": len(r["chosen"]) - len(rejected),
        "format_match": ("最终答案：" in r["chosen"]) == ("最终答案：" in rejected),
        "contains_true_math_error": true_error,
        "label_claim": label_claim,
    }


def build(rows: list[dict]) -> dict[str, list[dict]]:
    # 只用训练族的训练分片, 避免把测试题写进偏好数据
    train = [r for r in rows if r["split"] == "train" and r["subset"] == "train_id"]
    out: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for i, r in enumerate(train):
        ans, chosen, rationale = r["answer"], r["chosen"], r["rationale"]

        # 对照组: 完整复刻旧项目的三种 rejected, 全部不含真实数学错误
        out["shortcut"].append(_row(
            r, "shortcut", f"直接得到最终答案：{ans}", None, False,
            "wrong_final_answer", i))
        out["shortcut"].append(_row(
            r, "shortcut", f"答案是{ans}，无需推导。", None, False,
            "correct_final_incomplete_reasoning", i))
        out["shortcut"].append(_row(
            r, "shortcut", "我无法确定。", None, False, "refusal", i))

        w = wrong_answer(ans)
        if w is None:
            continue
        # 处理组: 推导里所有出现的答案同步替换为等长错误值, 格式与长度完全不变
        bad_rationale = rationale.replace(ans, w)
        deconf = bad_rationale + " 最终答案：" + w
        if len(deconf) == len(chosen) and "最终答案：" in deconf:
            out["deconfounded"].append(_row(
                r, "deconfounded", deconf, w, True, "wrong_final_answer", i))
        # 反事实组: rejected 更长, 但答案仍然是错的
        reversed_rejected = (rationale + " 补充说明：以上各步均已反复核对，确认无误。"
                             + " 最终答案：" + w)
        out["length_reversed"].append(_row(
            r, "length_reversed", reversed_rejected, w, True,
            "wrong_final_answer", i))
    return out


def diagnostics(groups: dict[str, list[dict]]) -> dict:
    out = {}
    for g, rows in groups.items():
        diffs = [r["length_diff"] for r in rows]
        out[g] = {
            "rows": len(rows),
            "true_math_error_rate": (sum(r["contains_true_math_error"] for r in rows) / len(rows)
                                     if rows else 0.0),
            "length_diff_min": min(diffs) if diffs else None,
            "length_diff_max": max(diffs) if diffs else None,
            "length_diff_mean": sum(diffs) / len(diffs) if diffs else None,
            "length_diff_sign_counts": dict(Counter(
                "chosen_longer" if d > 0 else ("rejected_longer" if d < 0 else "equal")
                for d in diffs)),
            "format_match_rate": (sum(r["format_match"] for r in rows) / len(rows)
                                  if rows else 0.0),
            "label_claim_counts": dict(Counter(r["label_claim"] for r in rows)),
        }
    return out


def write(groups: dict[str, list[dict]], out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g, rows in groups.items():
        with (out / f"preferences_{g}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    diag = diagnostics(groups)
    (out / "preferences_diagnostics.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    return diag


def load(group: str, data_dir: str | Path = "data") -> list[dict]:
    p = Path(data_dir) / f"preferences_{group}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
