"""无需模型权重的一键研究入口。

    python -m src.run_research

依次完成: 生成参数化数据集 -> 严格泄露检查 -> 构造去混淆偏好对 ->
旧项目审计 -> 三个子贡献的全部统计计算 -> 写入 results.jsonl ->
从 results.jsonl 生成报告。

所有 measured 数字都是真实执行结果。模型阶段缺失的指标记为 pending_model_run,
报告中只显示占位符, 不写任何编造数值。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import data_gen, preferences, results, splits, tech_report  # noqa: E402
from src.core.config import load as load_config  # noqa: E402
from src.research import (cluster_inference, identifiability,  # noqa: E402
                          legacy_audit, verifier_bias)
from src.model import algo_lab  # noqa: E402

DEFAULT_LEGACY = ROOT.parent / "math-posttraining-project"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--legacy_root", default=str(DEFAULT_LEGACY))
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--bootstrap_B", type=int, default=10000)
    ap.add_argument("--bias_B", type=int, default=2000)
    a = ap.parse_args()
    data_cfg = load_config(ROOT / "configs" / "data.yaml")
    if a.seed is None:
        a.seed = int(data_cfg["seed"])
    t0 = time.perf_counter()
    legacy = Path(a.legacy_root)
    data_dir = ROOT / a.data_dir

    print("[1/8] generate parametric dataset", flush=True)
    rows = data_gen.generate(a.seed,
                             n_per_family=int(data_cfg["n_per_family"]),
                             n_extra=int(data_cfg["n_extra_variants"]),
                             n_train_families=int(data_cfg["n_train_families"]))
    gen = data_gen.write_dataset(rows, data_dir, a.seed)
    print(f"      n={len(rows)}, leakage_ok={gen['check']['ok']}", flush=True)

    print("[2/8] build deconfounded preference groups", flush=True)
    groups = preferences.build(rows)
    diag = preferences.write(groups, data_dir)
    print("      " + json.dumps({k: v["rows"] for k, v in diag.items()}), flush=True)

    print("      verify oracle/verifier agreement on all chosen answers", flush=True)
    from src.core.verifier import CORRECT, VERIFIERS  # noqa: E402
    n_consistent = sum(
        VERIFIERS["strict_poly"].verify(r["chosen"], r["answer"]).verdict == CORRECT
        for r in rows)
    if n_consistent != len(rows):
        raise AssertionError(f"oracle/verifier mismatch: {n_consistent}/{len(rows)}")

    store = results.ResultStore(ROOT)
    store.record("data_audit", "n_rows", len(rows), unit="rows", seed=a.seed)
    store.record("data_audit", "n_families", gen["manifest"]["n_families"], unit="families")
    store.record("data_audit", "n_unique_questions",
                 gen["check"]["n_unique_questions"], unit="questions")
    store.record("data_audit", "n_unique_parameter_signatures",
                 gen["check"]["n_unique_signatures"], unit="signatures")
    store.record("data_audit", "cross_split_overlap",
                 gen["check"]["checks"]["cross_split_parameter_overlap"], unit="rows")
    store.record("data_audit", "oracle_verifier_consistent_rows", n_consistent,
                 unit="rows", n=len(rows),
                 note="measured by re-verifying every chosen answer with strict_poly")
    for g, d in diag.items():
        store.record("data_audit", f"preference_rows_{g}", d["rows"], unit="pairs")
        store.record("data_audit", f"preference_true_error_rate_{g}",
                     d["true_math_error_rate"], unit="rate", n=d["rows"])
        store.record("data_audit", f"preference_format_match_{g}",
                     d["format_match_rate"], unit="rate", n=d["rows"])
        if d["length_diff_min"] is not None:
            store.record("data_audit", f"preference_abs_length_diff_{g}",
                         abs(d["length_diff_mean"]), unit="chars", n=d["rows"])

    print("[3/8] audit legacy project", flush=True)
    aud = legacy_audit.run(store, legacy)
    print("      " + json.dumps({"unique_questions": aud["dataset"].get("n_unique_questions"),
                                "false_positives": aud["generative"].get("old_false_positives")},
                               ensure_ascii=False), flush=True)

    print("[4/8] contribution 1: identifiability", flush=True)
    ident = identifiability.run(store, groups, legacy, a.seed)
    print("      " + json.dumps({k: round(v["cv_auc"], 4)
                                for k, v in ident["groups"].items()}), flush=True)

    print("[5/8] contribution 2: cluster inference", flush=True)
    cl = cluster_inference.run(store, legacy, rows, a.seed, B=a.bootstrap_B)
    if cl["legacy"]:
        print(f"      effective_n={cl['legacy']['effective_n']:.2f}, "
              f"cluster_ci={cl['legacy']['cluster_ci']}", flush=True)

    print("[6/8] contribution 3: verifier bias correction", flush=True)
    vb = verifier_bias.run(store, rows, a.seed, B=a.bias_B)
    biased = vb["verifiers"]["legacy_substring"]
    print(f"      pi_true={vb['pi_true']:.4f}, "
          f"naive={biased['observed_accuracy']:.4f}, "
          f"corrected={biased['corrected_accuracy']:.4f}, "
          f"latent_class={vb['latent_class']['pi_hat']:.4f}", flush=True)

    print("[7/8] algorithm lab: SFT -> DPO on an analytic policy", flush=True)
    algo = algo_lab.run(store, ROOT, rows, a.seed)
    print("      " + json.dumps({
        "hack_rate_shortcut": algo["groups"]["shortcut"]["probe_hack_rate"],
        "hack_rate_deconfounded": algo["groups"]["deconfounded"]["probe_hack_rate"],
        "wrong_ref_kl": algo["anchor"]["wrong_kl"],
        "correct_ref_kl": algo["anchor"]["correct_kl"],
    }), flush=True)

    print("[8/8] record pending model metrics and render reports", flush=True)
    # 权重层 CPU pilot 的记录由 src/model/pilot.py 在真实 Qwen3-0.6B 权重上
    # 真实运行产生, 研究阶段不重算。重跑 research 时原样保留,
    # 避免把已经测得的权重层数字抹掉, 也避免用 pilot 冒充完整协议。
    existing_path = ROOT / "reports" / "results.jsonl"
    pilot_records = []
    if existing_path.exists():
        pilot_records = [r for r in results.ResultStore.load(existing_path)
                         if r["experiment"] == "model_pilot"]
    store.records.extend(pilot_records)
    for metric, note in [
        ("base_test_loss_masked", "assistant-token-only NLL on independent test set"),
        ("sft_test_loss_masked", "assistant-token-only NLL on independent test set"),
        ("dpo_test_loss_masked", "assistant-token-only NLL on independent test set"),
        ("generative_accuracy_base", "strict verifier, all four generalization subsets"),
        ("generative_accuracy_sft", "strict verifier, all four generalization subsets"),
        ("generative_accuracy_dpo", "strict verifier, all four generalization subsets"),
        ("dpo_kl_to_sft_ref", "Monte Carlo sequence-level KL with bootstrap CI"),
        ("dpo_accuracy_deconfounded_minus_shortcut",
         "downstream effect of deconfounded hard negatives"),
        ("beta_sweep_downstream_delta", "margin/beta is an identity, report downstream instead"),
    ]:
        store.pending("model_stage", metric, note)
    store.flush()

    dataset_sha = results.sha256_of_dir_files(
        ROOT, ("data/train.jsonl", "data/valid.jsonl", "data/test.jsonl",
               "data/manifest.json"))
    meta = {
        "seed": a.seed,
        "dataset": a.data_dir,
        "n_rows": len(rows),
        "n_families": gen["manifest"]["n_families"],
        "dataset_sha256": dataset_sha,
        "leakage_ok": gen["check"]["ok"],
        "legacy_root": str(legacy),
        "bootstrap_B": a.bootstrap_B,
        "bias_B": a.bias_B,
        "n_research_records": len(store.records) - len(pilot_records),
        "n_pilot_records": len(pilot_records),
    }
    (ROOT / "reports" / "experiment_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = results.render_all(ROOT, meta)
    tech = tech_report.write(ROOT)
    print(f"      tech report: {tech.name} ({tech.stat().st_size} bytes)", flush=True)
    check = results.verify_consistency(ROOT, meta)
    print(json.dumps({
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "n_records": len(store.records),
        "n_pending": sum(r["status"] == results.PENDING for r in store.records),
        "reports": [str(p) for p in paths.values()],
        "consistency": check,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if check["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
