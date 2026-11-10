"""统一生成协议下的严格生成式评测。

修复旧项目的致命问题 2、7、8:
  - 用三态 verifier 取代子串包含匹配, UNVERIFIABLE 单独统计。
  - 统一 enable_thinking=False、max_new_tokens>=256, 不再把 64 token 截断
    产生的「没有最终答案」当成数学错误。
  - 逐条 predictions 写盘, 供 cluster_inference 以题目族为聚类单位做配对推断。
  - 所有汇总数字通过 ResultStore 进入 results.jsonl, 不写独立报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.results import ResultStore, append_records, render_all  # noqa: E402
from src.core.verifier import CORRECT, VERIFIERS  # noqa: E402
from src.model.teacher_forced import _load, _apply_chat_template  # noqa: E402


def generate_all(path: str, test_file: str, base_model: str,
                 max_new_tokens: int = 256, limit: int | None = None) -> list[dict]:
    import torch
    from src.model import check_deps
    check_deps()
    tok, model = _load(path, base_model)
    rows = [json.loads(l) for l in Path(test_file).read_text(encoding="utf-8").splitlines()]
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        prompt = _apply_chat_template(tok, r["messages"][:2], add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        v = VERIFIERS["strict_poly"].verify(text, r["answer"])
        out.append({
            "id": r["id"], "family": r["family"], "subset": r["subset"],
            "gold": r["answer"], "pred": text, "verdict": v.verdict,
            "extracted": v.extracted, "reason": v.reason,
            "pred_chars": len(text),
        })
    return out


def summarize(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0}
    cnt = defaultdict(int)
    for x in items:
        cnt[x["verdict"]] += 1
    by_subset = defaultdict(list)
    for x in items:
        by_subset[x["subset"]].append(x)
    return {
        "n": n,
        "correct_rate": cnt[CORRECT] / n,
        "incorrect_rate": cnt["incorrect"] / n,
        "unverifiable_rate": cnt["unverifiable"] / n,
        "mean_chars": sum(x["pred_chars"] for x in items) / n,
        "by_subset": {
            k: {"n": len(v), "correct_rate": sum(x["verdict"] == CORRECT for x in v) / len(v),
                "unverifiable_rate": sum(x["verdict"] == "unverifiable" for x in v) / len(v)}
            for k, v in sorted(by_subset.items())
        },
    }


def run(base_model: str, sft_dir: str, dpo_dir: str, test_file: str,
        root: Path = ROOT, max_new_tokens: int = 256,
        tag: str = "", only: tuple[str, ...] | None = None) -> dict:
    from src.model import check_deps
    check_deps()
    models = {"base": base_model, "sft": sft_dir, "dpo": dpo_dir}
    suffix = f"_{tag}" if tag else ""
    predictions, summaries, records = {}, {}, []
    for name, path in models.items():
        if only and name not in only:
            continue
        if not path or not Path(path).exists():
            continue
        items = generate_all(path, test_file, base_model, max_new_tokens)
        predictions[name] = items
        p = root / "reports" / f"predictions_{name}{suffix}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
        s = summarize(items)
        summaries[name] = s
        m = lambda key: f"{key}_{name}{suffix}"
        records.append({"experiment": "model_stage", "metric": m("generative_accuracy"),
                        "value": s["correct_rate"], "unit": "rate", "n": s["n"],
                        "ci_low": None, "ci_high": None, "status": "measured",
                        "note": "strict verifier, assistant-only protocol", "source": str(p)})
        records.append({"experiment": "model_stage", "metric": m("unverifiable_rate"),
                        "value": s["unverifiable_rate"], "unit": "rate", "n": s["n"],
                        "ci_low": None, "ci_high": None, "status": "measured",
                        "note": "explicitly reported, never folded into incorrect",
                        "source": str(p)})
        for subset, v in s["by_subset"].items():
            records.append({"experiment": "model_stage",
                            "metric": m(f"generative_accuracy_{subset}"),
                            "value": v["correct_rate"], "unit": "rate", "n": v["n"],
                            "ci_low": None, "ci_high": None, "status": "measured",
                            "note": "generalization subset", "source": str(p)})
    if records:
        append_records(root, records)
        meta_path = root / "reports" / "experiment_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        render_all(root, meta)
    return {"predictions": predictions, "summaries": summaries,
            "n_records_written": len(records)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--sft_dir")
    ap.add_argument("--dpo_dir")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="comma separated subset of base,sft,dpo")
    a = ap.parse_args()
    if a.max_new_tokens < 256:
        raise SystemExit("max_new_tokens must be >= 256; 64-token truncation was a legacy bug")
    only = tuple(x.strip() for x in a.only.split(",") if x.strip()) or None
    out = run(a.base_model, a.sft_dir, a.dpo_dir, a.test_file,
              max_new_tokens=a.max_new_tokens, tag=a.tag, only=only)
    print(json.dumps(out["summaries"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
