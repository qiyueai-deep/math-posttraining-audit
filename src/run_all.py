"""全流程入口。

    python -m src.run_all --stage research   # 无需 torch, 本机可跑, 产出真实数字
    python -m src.run_all --stage algo       # 单独演示算法实验台
    python -m src.run_all --stage pilot      # 真实 Qwen3-0.6B 权重上的 CPU pilot
    python -m src.run_all --stage model     # 需要 GPU 环境
    python -m src.run_all --stage check     # 三轮检查
    python -m src.run_all --stage all

模型阶段在依赖缺失时会明确报错并停止, 绝不静默跳过, 也不写入任何伪造指标。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import dump as dump_config, load as load_config  # noqa: E402

GROUPS = ("deconfounded", "shortcut", "length_reversed")


def _run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def stage_research() -> None:
    _run("src.run_research")


def stage_algo() -> None:
    """现场演示算法实验台: 脚本方式运行, 不写单一事实源。"""
    subprocess.run([sys.executable, str(ROOT / "algo_lab_pycharm.py")],
                   cwd=ROOT, check=True)


def _dpo_config(group: str) -> str:
    c = load_config(ROOT / "configs" / "dpo.yaml")
    c["train_file"] = f"data/preferences_{group}.jsonl"
    c["output_dir"] = f"runs/dpo_{group}"
    p = ROOT / "runs" / "_tmp_configs" / f"dpo_{group}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    dump_config(c, p)
    return str(p.relative_to(ROOT))


def stage_model() -> None:
    from src.model import check_deps
    state = check_deps(raise_on_missing=False)
    missing = [k for k, v in state.items() if not v]
    if missing:
        raise SystemExit(
            "模型阶段依赖缺失: " + ", ".join(missing)
            + "\n请执行: pip install -r requirements.txt (需要 GPU 环境)"
            + "\n或者只运行 CPU 可跑的阶段: python -m src.run_all --stage research"
            + " / --stage algo"
            + "\n在此之前, reports/research_report.md 中所有模型指标保持 pending_model_run。")

    eval_cfg = load_config(ROOT / "configs" / "eval.yaml")
    base = str(eval_cfg["base_model"])
    test_file = str(eval_cfg["test_file"])

    _run("src.model.train_sft", "--config", "configs/sft.yaml")
    for group in GROUPS:
        _run("src.model.train_dpo", "--config", _dpo_config(group),
             "--preference_group", group)

    # 主对照链: base / sft / deconfounded-DPO
    _run("src.model.teacher_forced", "--base_model", base,
         "--sft_dir", "runs/sft", "--dpo_dir", "runs/dpo_deconfounded",
         "--test_file", test_file)
    _run("src.model.evaluate", "--base_model", base,
         "--sft_dir", "runs/sft", "--dpo_dir", "runs/dpo_deconfounded",
         "--test_file", test_file,
         "--max_new_tokens", str(int(eval_cfg["max_new_tokens"])))
    _run("src.model.diagnostics", "--base_model", base,
         "--sft_dir", "runs/sft", "--dpo_dir", "runs/dpo_deconfounded",
         "--test_file", test_file)

    # 三个偏好组的下游对照, 用来验证去混淆是否真的带来数学能力上的差异
    for group in ("shortcut", "length_reversed"):
        _run("src.model.evaluate", "--base_model", base,
             "--sft_dir", "runs/sft", "--dpo_dir", f"runs/dpo_{group}",
             "--test_file", test_file, "--tag", group, "--only", "dpo",
             "--max_new_tokens", str(int(eval_cfg["max_new_tokens"])))
    _summarize_model_stage(eval_cfg)


def _summarize_model_stage(eval_cfg: dict) -> None:
    """把模型阶段产物汇总进 results.jsonl, 并跑真实聚类推断。"""
    from src.core.results import ResultStore, append_records, render_all, verify_consistency
    from src.research.cluster_inference import run_from_predictions

    reports = ROOT / "reports"
    preds = {}
    for name, fn in (("base", "predictions_base.jsonl"),
                     ("sft", "predictions_sft.jsonl"),
                     ("dpo", "predictions_dpo.jsonl")):
        p = reports / fn
        if p.exists():
            preds[name] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]

    # 主链的 dpo 就是 deconfounded 组, 补一条同名记录以便三组横向对照
    main_dpo = reports / "predictions_dpo.jsonl"
    records = []
    if main_dpo.exists():
        rows = [json.loads(l) for l in main_dpo.read_text(encoding="utf-8").splitlines()]
        records.append({"experiment": "model_stage",
                        "metric": "generative_accuracy_dpo_deconfounded",
                        "value": sum(x["verdict"] == "correct" for x in rows) / len(rows),
                        "unit": "rate", "n": len(rows),
                        "ci_low": None, "ci_high": None, "status": "measured",
                        "note": "main chain dpo uses deconfounded preferences",
                        "source": str(main_dpo)})
    if records:
        append_records(ROOT, records)
    if len(preds) >= 2:
        store = ResultStore(ROOT)
        run_from_predictions(store, preds, int(eval_cfg["seed"]),
                             B=int(eval_cfg["cluster_bootstrap_B"]))
        append_records(ROOT, store.records)
    meta_path = reports / "experiment_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    render_all(ROOT, meta)
    print(json.dumps(verify_consistency(ROOT, meta), ensure_ascii=False, indent=2))


def stage_check() -> None:
    _run("tests.run_three_round_checks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("research", "algo", "pilot", "model", "check", "all"),
                    default="research")
    a = ap.parse_args()
    if a.stage == "research":
        stage_research()
    elif a.stage == "algo":
        stage_algo()
    elif a.stage == "pilot":
        stage_pilot()
    elif a.stage == "model":
        stage_model()
    elif a.stage == "check":
        stage_check()
    else:
        stage_research()
        stage_model()
        stage_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
