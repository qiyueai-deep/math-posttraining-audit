"""三轮检查。

第一轮: 全量语法编译 + 单元测试(表达式、验证器、数据、偏好、统计估计量、
prompt mask、算法实验台的梯度与 KL)。
第二轮: 运行结果与报告数字一致性、数据哈希、单一事实源、无旧指标残留。
第三轮: 逐条对照 8 个致命问题与 3 个子贡献, 并确认原项目文件未被改动。

    python -m tests.run_three_round_checks
"""

from __future__ import annotations

import io
import json
import math
import py_compile
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.results import ResultStore, sha256_of_dir_files, verify_consistency  # noqa: E402


def _failures() -> list[str]:
    return []


# --------------------------------------------------------------- 第一轮

def round1() -> dict:
    failures = []
    files = sorted(p for p in ROOT.glob("**/*.py")
                   if "__pycache__" not in p.parts and ".python-deps" not in p.parts)
    for p in files:
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            failures.append(f"compile failed: {p.relative_to(ROOT)}: {e}")
    stream = io.StringIO()
    suite = unittest.TestLoader().loadTestsFromName("tests.test_core")
    res = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    for t, tb in res.failures + res.errors:
        failures.append(f"unittest {t}: {tb.strip().splitlines()[-1]}")
    return {"ok": not failures, "n_files_compiled": len(files),
            "tests_run": res.testsRun, "failures": failures,
            "unittest_log": stream.getvalue()}


# --------------------------------------------------------------- 第二轮

def round2() -> dict:
    failures = []
    reports = ROOT / "reports"
    meta_path = reports / "experiment_meta.json"
    if not meta_path.exists():
        return {"ok": False, "failures": ["missing reports/experiment_meta.json"]}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    check = verify_consistency(ROOT, meta)
    if not check["ok"]:
        failures.append(f"results/report consistency: {check}")

    records = ResultStore.load(reports / "results.jsonl")
    from src.core import tech_report
    html_path = reports / "tech_report.html"
    if not html_path.exists():
        failures.append("missing reports/tech_report.html")
    elif html_path.read_text(encoding="utf-8") != tech_report.render(records, ROOT):
        failures.append("tech_report.html is stale or hand-edited")
    m = re.search(r"研究阶段 (\d+) 条记录",
                  (ROOT / "README.md").read_text(encoding="utf-8"))
    n_pilot = sum(1 for r in records if r["experiment"] == "model_pilot")
    expected = meta.get("n_research_records", len(records) - n_pilot)
    if m and int(m.group(1)) != expected:
        failures.append(f"README research record count {m.group(1)} != {expected}")
    if len(records) - n_pilot != expected:
        failures.append("research record count inconsistent with experiment_meta.json")
    if n_pilot:
        mp = reports / "model_pilot_manifest.json"
        if not mp.exists():
            failures.append("missing reports/model_pilot_manifest.json")
        else:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            if not manifest.get("formal_protocol_still_pending"):
                failures.append("pilot manifest does not declare the formal protocol pending")
            if not manifest.get("adapters"):
                failures.append("pilot manifest has no adapter hashes")
            if manifest.get("seed") != meta.get("seed"):
                failures.append("pilot seed differs from research seed")
    for r in records:
        if r["status"] == "measured":
            if r["value"] is None:
                failures.append(f"measured with null value: {r['experiment']}/{r['metric']}")
            elif isinstance(r["value"], (int, float)) and not math.isfinite(r["value"]):
                failures.append(f"non-finite measured: {r['experiment']}/{r['metric']}")
        elif r["status"] == "pending_model_run" and r["value"] is not None:
            failures.append(f"pending record has a value: {r['experiment']}/{r['metric']}")

    # 报告表格第 2 列的每个数字都必须能在 results.jsonl 里找到出处
    allowed = set()
    for r in records:
        for key in ("value", "ci_low", "ci_high"):
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                allowed.add(f"{v:.4f}")
                allowed.add(str(int(v)) if float(v).is_integer() else f"{v:.4f}")
    report = (reports / "research_report.md").read_text(encoding="utf-8")
    orphan = []
    for line in report.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "指标 |" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[4] != "measured":
            continue
        val = cells[1]
        if val in ("PENDING_MODEL_RUN", "-"):
            continue
        if re.fullmatch(r"-?\d+\.\d+|-?\d+", val) and val not in allowed \
           and f"{float(val):.4f}" not in allowed:
            orphan.append(val)
    if orphan:
        failures.append(f"report numbers without a results.jsonl record: {orphan[:8]}")
    ci_vals = re.findall(r"\[(-?\d+\.\d+), (-?\d+\.\d+)\]", report)
    for lo, hi in ci_vals:
        if lo not in allowed or hi not in allowed:
            failures.append(f"CI endpoint without a record: [{lo}, {hi}]")

    # 面试与简历文档里出现的四位小数也必须有出处, 防止手写数字悄悄混进来
    allowed |= {f"{abs(v):.4f}" for r in records
                for k in ("value", "ci_low", "ci_high")
                if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                for v in [r[k]]}
    doc_orphans = []
    for name in ("interview_guide.md", "resume_rewrite.md", "resume_bullets.md",
                 "algorithm_interview.md"):
        p = reports / name
        if not p.exists():
            failures.append(f"missing document: {name}")
            continue
        for num in set(re.findall(r"-?\d+\.\d{4}", p.read_text(encoding="utf-8"))):
            if num not in allowed:
                doc_orphans.append(f"{name}:{num}")
    if doc_orphans:
        failures.append(f"document numbers without a results.jsonl record: {sorted(doc_orphans)[:8]}")

    # 数据哈希与泄露检查
    digest = sha256_of_dir_files(
        ROOT, ("data/train.jsonl", "data/valid.jsonl", "data/test.jsonl", "data/manifest.json"))
    if digest != meta.get("dataset_sha256"):
        failures.append("dataset SHA256 does not match experiment_meta.json")
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["leakage"]["ok"]:
        failures.append("data/manifest.json reports leakage")
    if manifest["seed"] != meta["seed"]:
        failures.append("seed mismatch between data/manifest.json and experiment_meta.json")

    # 偏好诊断与结果记录一致
    diag = json.loads((ROOT / "data" / "preferences_diagnostics.json").read_text(encoding="utf-8"))
    for g, d in diag.items():
        key = ("data_audit", f"preference_rows_{g}")
        hit = [r for r in records if (r["experiment"], r["metric"]) == key]
        if not hit or hit[0]["value"] != d["rows"]:
            failures.append(f"preference row count mismatch for {g}")

    # 源码与配置里不得残留旧项目的硬编码指标
    legacy_numbers = ("12.1811", "2.4999", "81.9352", "4.4059", "12.3767", "2.5158")
    for p in sorted(list((ROOT / "src").glob("**/*.py"))
                    + list((ROOT / "configs").glob("*.yaml"))
                    + list((ROOT / "data").glob("*.jsonl"))
                    + list(ROOT.glob("*.py"))):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for num in legacy_numbers:
            if num in text:
                failures.append(f"legacy metric hardcoded in {p.relative_to(ROOT)}: {num}")

    return {"ok": not failures, "n_records": len(records),
            "n_pending": sum(r["status"] == "pending_model_run" for r in records),
            "dataset_sha256": digest, "consistency": check, "failures": failures}


# --------------------------------------------------------------- 第三轮

def round3() -> dict:
    failures, checks = [], {}
    reports = ROOT / "reports"
    records = ResultStore.load(reports / "results.jsonl")
    idx = {(r["experiment"], r["metric"]): r["value"] for r in records}
    src = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "src").glob("**/*.py")}
    all_src = "\n".join(src.values())
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))

    # 1 数据: 唯一题干、跨 split 零重叠、未见族只在 test
    checks["P1_data"] = (manifest["leakage"]["n_unique_questions"] == manifest["n_rows"]
                         and manifest["leakage"]["checks"]["cross_split_parameter_overlap"] == 0
                         and not manifest["leakage"]["checks"]["family_in_both_train_and_unseen"])
    # 2 评测器: 无子串匹配、无静默回退、UNVERIFIABLE 存在
    verifier_src = src.get("verifier.py", "")
    checks["P2_verifier"] = ("in norm(pred)" not in verifier_src
                             and "return exact(pred,ans)" not in verifier_src
                             and "UNVERIFIABLE" in verifier_src)
    # 3 loss mask
    tf = src.get("teacher_forced.py", "")
    checks["P3_prompt_mask"] = ("offset_mapping" in tf and "-100" in tf
                                and "token_weighted" in tf)
    # 4 偏好: 去混淆组真错误、长度格式匹配, 旧缺陷组作为对照
    checks["P4_preferences"] = (
        idx.get(("identifiability", "surface_auc_shortcut"), 0) > 0.95
        and abs(idx.get(("identifiability", "surface_auc_deconfounded"), 1) - 0.5) < 0.12
        and idx.get(("data_audit", "preference_true_error_rate_deconfounded")) == 1.0
        and idx.get(("data_audit", "preference_true_error_rate_shortcut")) == 0.0)
    # 5 beta 恒等式与真 KL
    checks["P5_beta"] = (
        idx.get(("identifiability", "margin_over_beta_cv"), 1) < 0.05
        and "kl_estimate" in src.get("diagnostics.py", ""))
    # 6 显式 ref_model
    dpo = src.get("train_dpo.py", "")
    checks["P6_explicit_ref"] = ("ref_model=ref" in dpo and "ref_model=None" not in dpo
                                 and "assert_ref_matches_policy" in dpo)
    # 7 不再把解码截断当数学错误
    eval_cfg = (ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8")
    checks["P7_no_truncation_bug"] = ("max_new_tokens: 256" in eval_cfg
                                      and "missing_format" not in all_src)
    # 8 单一事实源
    checks["P8_single_source"] = (
        "results.jsonl" in src.get("results.py", "")
        and "verify_consistency" in all_src
        and (reports / "results.jsonl").exists())
    # 9 PyCharm 右键入口: 三个脚本存在, 自动 chdir, src 下无相对导入
    entries = ("run_pycharm.py", "check_pycharm.py", "show_results_pycharm.py",
               "algo_lab_pycharm.py", "build_pdf_pycharm.py",
               "run_model_pycharm.py")
    checks["P9_pycharm"] = (
        all((ROOT / name).exists() for name in entries)
        and all("os.chdir(ROOT)" in (ROOT / name).read_text(encoding="utf-8")
                for name in entries)
        and not re.search(r"^\s*from \.", all_src, re.M)
        and "import yaml" not in all_src
        and (ROOT / "PYCHARM_GUIDE.md").exists())
    checks["P10_docs"] = all((reports / name).exists() for name in
                             ("interview_guide.md", "resume_rewrite.md",
                              "resume_bullets.md", "algorithm_interview.md"))
    # 11 算法实验台: 真跑 SFT->DPO, 梯度校验, 三组对照方向, 锚点对照, 曲线产物
    checks["P11_algo_lab"] = (
        idx.get(("algo_lab", "dpo_gradient_max_abs_error"), 1) < 1e-6
        and idx.get(("algo_lab", "gradient_check_passed")) == 1
        and idx.get(("algo_lab", "margin_over_beta_cv_fixed_theta"), 1) < 1e-6
        and idx.get(("algo_lab", "probe_hack_rate_shortcut"), 0) > 0.6
        and idx.get(("algo_lab", "probe_hack_rate_deconfounded"), 1) < 0.25
        and idx.get(("algo_lab", "probe_hack_rate_shortcut"), 0)
        > idx.get(("algo_lab", "probe_hack_rate_deconfounded"), 1) + 0.3
        and idx.get(("algo_lab", "probe_accuracy_deconfounded"), 0)
        > idx.get(("algo_lab", "probe_accuracy_shortcut"), 1)
        and idx.get(("algo_lab", "theta_quality_deconfounded"), 0)
        > idx.get(("algo_lab", "theta_quality_shortcut"), 1e9)
        and idx.get(("algo_lab", "wrong_anchor_kl"), 0)
        > idx.get(("algo_lab", "kl_deconfounded"), 1e9)
        and (reports / "algo_lab_curve.svg").exists()
        and (reports / "algo_lab_curves.json").exists())
    # 13 权重层 CPU pilot: 若已跑过, 记录必须完整且完整协议仍为 pending
    pilot = [r for r in records if r["experiment"] == "model_pilot"]
    pidx = {(r["experiment"], r["metric"]): r["value"] for r in pilot}
    checks["P13_model_pilot"] = True
    if pilot:
        needs = ["mask_self_test_completion_tokens", "ref_assert_checked_parameters",
                 "sft_loss_first", "sft_loss_last", "test_loss_masked_base",
                 "test_loss_masked_sft", "sequence_kl_mc", "elapsed_seconds"]
        needs += [f"reward_margin_gain_{g}" for g in
                  ("shortcut", "deconfounded", "length_reversed")]
        needs += [f"generative_accuracy_dpo_{g}" for g in
                  ("shortcut", "deconfounded", "length_reversed")]
        checks["P13_model_pilot"] = (
            all(("model_pilot", k) in pidx and pidx[("model_pilot", k)] is not None
                for k in needs)
            and idx.get(("model_pilot", "ref_is_frozen_sft_adapter")) == 1
            and all(r["status"] == "pending_model_run"
                    for r in records if r["experiment"] == "model_stage"))
    # 12 技术报告: HTML 存在, 含全部章节与待运行标注
    html = (reports / "tech_report.html")
    checks["P12_tech_report"] = False
    if html.exists():
        text = html.read_text(encoding="utf-8")
        checks["P12_tech_report"] = (
            html.stat().st_size > 30000
            and all(h in text for h in ("问题与起点", "后训练链路", "算法机制实验台",
                                        "三个测量问题", "讨论", "复现"))
            and "results.jsonl" in text and "pending" in text)
    for k, v in checks.items():
        if not v:
            failures.append(f"{k} not satisfied")

    # 三个子贡献都必须有真实 measured 结果
    contributions = {
        "C1_identifiability": [("identifiability", "surface_auc_shortcut"),
                               ("identifiability", "surface_auc_deconfounded"),
                               ("identifiability", "max_abs_change_under_Q_permute")],
        "C2_cluster_inference": [("cluster_inference", "effective_n"),
                                 ("cluster_inference", "design_effect"),
                                 ("cluster_inference", "cluster_ci_low"),
                                 ("cluster_inference", "clusters_needed_for_5pp")],
        "C3_verifier_bias": [("verifier_bias", "fpr_legacy_substring"),
                             ("verifier_bias", "corrected_accuracy"),
                             ("verifier_bias", "latent_class_pi"),
                             ("verifier_bias", "latent_class_selection_bias")],
        "A1_algorithm_lab": [("algo_lab", "dpo_gradient_max_abs_error"),
                             ("algo_lab", "reward_margin_shortcut"),
                             ("algo_lab", "reward_margin_deconfounded"),
                             ("algo_lab", "wrong_anchor_kl"),
                             ("algo_lab", "beta_0p5_kl")],
    }
    for name, keys in contributions.items():
        missing = [k for k in keys if k not in idx or idx[k] is None]
        checks[name] = not missing
        if missing:
            failures.append(f"{name} missing measured records: {missing}")

    # 原项目未被改动
    snap = reports / "original_project_snapshot.sha256"
    legacy = ROOT.parent / "math-posttraining-project"
    if snap.exists() and legacy.exists():
        import hashlib
        mism = []
        for line in snap.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            want, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
            p = legacy / rel
            if not p.exists():
                mism.append(f"missing {rel}")
                continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want:
                mism.append(f"changed {rel}")
        checks["original_project_untouched"] = not mism
        if mism:
            failures.append(f"original project changed: {mism[:5]}")
    else:
        checks["original_project_untouched"] = "snapshot unavailable"

    return {"ok": not failures, "checks": checks, "failures": failures}


def main() -> int:
    out = {"round1": round1(), "round2": round2(), "round3": round3()}
    out["ok"] = all(out[k]["ok"] for k in ("round1", "round2", "round3"))
    (ROOT / "reports" / "check_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in ("round1", "round2", "round3"):
        r = out[k]
        print(f"{k}: {'PASS' if r['ok'] else 'FAIL'} "
              + json.dumps({x: r[x] for x in r if x not in ("failures", "unittest_log")},
                           ensure_ascii=False))
        for f in r["failures"]:
            print("   -", f)
    print("ALL_CHECKS_PASS" if out["ok"] else "CHECKS_FAILED")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
