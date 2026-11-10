"""第一轮: 单元测试。全部零第三方依赖, 不需要 torch。"""

from __future__ import annotations

import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.data_gen import FAMILIES, generate  # noqa: E402
from src.core.expressions import ParseError, extract_answer, values_equal  # noqa: E402
from src.core.preferences import build as build_prefs  # noqa: E402
from src.core.splits import leakage_check  # noqa: E402
from src.core.verifier import CORRECT, INCORRECT, UNVERIFIABLE, VERIFIERS, verify  # noqa: E402
from src.model.teacher_forced import self_test as mask_self_test  # noqa: E402
from src.research.cluster_inference import (design_effect, holm, icc_anova,  # noqa: E402
                                            mcnemar_exact, power_at,
                                            required_clusters, wilson_ci)
from src.research.identifiability import auc_score, cv_auc, dpo_loss_invariance  # noqa: E402
from src.research.verifier_bias import corrected_accuracy, latent_class_em  # noqa: E402


def _rows():
    out = []
    for name in ("train", "valid", "test"):
        p = ROOT / "data" / f"{name}.jsonl"
        if not p.exists():
            return []
        out += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    return out


class TestExpressions(unittest.TestCase):
    def test_exact_rational_equality(self):
        self.assertTrue(values_equal("5/6", "10/12"))
        self.assertTrue(values_equal("x^2-4*x+3", "(x-2)^2-1"))
        self.assertTrue(values_equal("1,2", "2,1"))
        self.assertFalse(values_equal("-1", "-2"))
        self.assertFalse(values_equal("21", "22"))

    def test_latex_normalization(self):
        self.assertTrue(values_equal("\\frac{1}{2}+\\frac{1}{3}", "5/6"))
        self.assertTrue(values_equal("\\boxed{-1}", "-1"))

    def test_parse_failure_is_explicit(self):
        with self.assertRaises(ParseError):
            values_equal("1/2", "not an expression at all")

    def test_answer_extraction(self):
        self.assertEqual(extract_answer("推导... 最终答案：21。"), "21")
        self.assertIsNone(extract_answer("设 X~Bernoulli(p)，结果为 0 或 1。"))
        self.assertEqual(extract_answer("最终答案：x=4"), "4")
        self.assertEqual(extract_answer("最终答案：1,-5"), "1,-5")


class TestVerifier(unittest.TestCase):
    def test_no_substring_containment(self):
        old_trap = "设 X~Bernoulli(p)，概率为 p，结果为 0 或 1。"
        self.assertEqual(verify(old_trap, "p").verdict, UNVERIFIABLE)
        self.assertEqual(verify("推导里出现 1，最终答案：2", "1").verdict, INCORRECT)

    def test_three_states(self):
        self.assertEqual(verify("最终答案：\\boxed{5/6}", "5/6").verdict, CORRECT)
        self.assertEqual(verify("我无法确定。", "21").verdict, UNVERIFIABLE)
        self.assertEqual(verify("最终答案：我无法确定", "21").verdict, UNVERIFIABLE)

    def test_legacy_verifier_is_biased_but_not_used_for_scoring(self):
        leak = "推导中出现了 5/6，但没有按协议给出最终答案。"
        self.assertEqual(VERIFIERS["legacy_substring"].verify(leak, "5/6").verdict, CORRECT)
        self.assertEqual(verify(leak, "5/6").verdict, UNVERIFIABLE)
        self.assertEqual(set(VERIFIERS), {"strict_poly", "numeric_float", "legacy_substring"})


class TestDataset(unittest.TestCase):
    def test_oracle_cross_checks(self):
        for fam in FAMILIES:
            for _ in range(12):
                import random
                p = fam.sample(random.Random(12345 + _), False)
                ans = fam.oracle(p)
                self.assertTrue(fam.cross_check(p, ans), f"{fam.name}: {p} -> {ans}")

    def test_no_leakage_and_unique_questions(self):
        rows = _rows()
        if not rows:
            self.skipTest("data/ not generated")
        check = leakage_check(rows)
        self.assertTrue(check["ok"], check["checks"])
        self.assertEqual(check["n_unique_questions"], len(rows))
        self.assertEqual(check["n_unique_signatures"], len(rows))
        self.assertEqual(check["checks"]["cross_split_parameter_overlap"], 0)
        self.assertFalse(check["checks"]["family_in_both_train_and_unseen"])

    def test_every_chosen_answer_verifies(self):
        rows = _rows()
        if not rows:
            self.skipTest("data/ not generated")
        bad = [r["id"] for r in rows
               if verify(r["chosen"], r["answer"]).verdict != CORRECT]
        self.assertEqual(bad, [], f"oracle/verifier mismatch: {bad[:5]}")

    def test_generalization_subsets_present(self):
        rows = _rows()
        if not rows:
            self.skipTest("data/ not generated")
        c = Counter(r["subset"] for r in rows)
        for s in ("test_ood_param", "test_paraphrase", "test_counterfactual",
                  "test_unseen_family"):
            self.assertGreater(c[s], 0, s)


class TestPreferences(unittest.TestCase):
    def setUp(self):
        self.rows = _rows()
        if not self.rows:
            self.skipTest("data/ not generated")
        self.groups = build_prefs(self.rows)

    def test_shortcut_group_has_no_true_math_error(self):
        g = self.groups["shortcut"]
        self.assertEqual(sum(x["contains_true_math_error"] for x in g), 0)
        self.assertTrue(all(x["length_diff"] > 0 for x in g))

    def test_deconfounded_is_length_and_format_matched(self):
        g = self.groups["deconfounded"]
        self.assertTrue(g)
        self.assertTrue(all(x["length_diff"] == 0 for x in g))
        self.assertTrue(all(x["format_match"] for x in g))
        self.assertTrue(all(x["contains_true_math_error"] for x in g))
        for x in g[:50]:
            self.assertNotEqual(
                VERIFIERS["strict_poly"].verify(x["rejected"], x["answer"]).verdict,
                CORRECT)

    def test_length_reversed_is_longer_but_wrong(self):
        g = self.groups["length_reversed"]
        self.assertTrue(g)
        self.assertTrue(all(x["length_diff"] < 0 for x in g))
        self.assertTrue(all(x["contains_true_math_error"] for x in g))

    def test_surface_auc_separates_shortcut_not_deconfounded(self):
        a = cv_auc(self.groups["shortcut"], seed=7)["cv_auc"]
        b = cv_auc(self.groups["deconfounded"], seed=7)["cv_auc"]
        self.assertGreater(a, 0.95)
        self.assertLess(abs(b - 0.5), 0.12, f"deconfounded AUC={b}")


class TestStatistics(unittest.TestCase):
    def test_auc_known_cases(self):
        self.assertEqual(auc_score([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(auc_score([1, 2, 3, 4], [0, 1, 0, 1]), 0.75)

    def test_icc_and_design_effect(self):
        clusters = {"a": [1] * 20, "b": [0] * 20}
        self.assertAlmostEqual(icc_anova(clusters), 1.0, places=6)
        self.assertAlmostEqual(design_effect(1.0, 20), 20.0)

    def test_mcnemar_exact(self):
        self.assertAlmostEqual(mcnemar_exact(5, 0), 0.0625, places=6)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_holm_monotone(self):
        adj = holm([0.01, 0.04, 0.03])
        self.assertLessEqual(adj[0], adj[2])
        self.assertLessEqual(adj[2], adj[1])

    def test_wilson_and_power(self):
        lo, hi = wilson_ci(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreater(required_clusters(0.05, 0.2), 0)
        self.assertGreater(power_at(100, 0.05, 0.2), power_at(2, 0.05, 0.2))

    def test_bias_correction_identity(self):
        self.assertAlmostEqual(corrected_accuracy(0.7, 0.0, 0.0), 0.7)
        self.assertAlmostEqual(corrected_accuracy(0.8, 0.2, 0.1), 0.6 / 0.7, places=6)

    def test_dpo_invariance(self):
        inv = dpo_loss_invariance(seed=7)
        self.assertLess(inv["max_abs_change_under_Q_permute"], 1e-12)
        self.assertGreater(inv["max_abs_change_when_delta_depends_on_Q"], 1e-6)

    def test_latent_class_identifiability(self):
        em2 = latent_class_em([[1, 0] * 50, [1, 0] * 50], restarts=3, iters=80, seed=7)
        self.assertFalse(em2["identified"])
        em3 = latent_class_em([[1, 0] * 50, [1, 0] * 50, [1, 1, 0, 0] * 25],
                              restarts=3, iters=80, seed=7)
        self.assertTrue(em3["identified"])


class TestPromptMask(unittest.TestCase):
    def test_mask_covers_prompt(self):
        r = mask_self_test()
        self.assertTrue(r["ok"])
        self.assertGreater(r["n_masked"], 0)
        self.assertEqual(r["n_completion"], len("ANSWER\n"))


class TestModelLayerStatic(unittest.TestCase):
    """模型层本机跑不了, 但可以静态验证配置生成与依赖检查不会骗人。"""

    def test_dpo_config_is_a_copy_not_an_overwrite(self):
        from src.core.config import load as load_config
        from src.run_all import _dpo_config, ROOT as R
        original = (R / "configs" / "dpo.yaml").read_text(encoding="utf-8")
        p = R / _dpo_config("shortcut")
        c = load_config(p)
        self.assertEqual(c["output_dir"], "runs/dpo_shortcut")
        self.assertEqual(c["train_file"], "data/preferences_shortcut.jsonl")
        self.assertEqual((R / "configs" / "dpo.yaml").read_text(encoding="utf-8"), original)

    def test_dependency_check_does_not_raise_when_probed(self):
        from src.model import check_deps
        state = check_deps(raise_on_missing=False)
        self.assertIsInstance(state, dict)
        self.assertEqual(set(state), {"torch", "transformers", "peft", "trl", "datasets"})

    def test_model_scripts_contain_required_fixes(self):
        d = ROOT / "src" / "model"
        sft = (d / "train_sft.py").read_text(encoding="utf-8")
        dpo = (d / "train_dpo.py").read_text(encoding="utf-8")
        tf = (d / "teacher_forced.py").read_text(encoding="utf-8")
        diag = (d / "diagnostics.py").read_text(encoding="utf-8")
        ev = (d / "evaluate.py").read_text(encoding="utf-8")
        self.assertIn("build_labels", sft)
        self.assertIn("masked_valid_loss", sft)
        self.assertIn("ref.requires_grad_(False)", dpo)
        self.assertIn("ref.eval()", dpo)
        self.assertIn("enable_thinking=False", tf)
        self.assertIn("kl_estimate", diag)
        self.assertIn("bootstrap", diag)
        self.assertIn("max_new_tokens", ev)


class TestZeroDependencyConfig(unittest.TestCase):
    def test_roundtrip_all_supported_value_types(self):
        from src.core.config import dump, load
        src = {"seed": 20251109, "lr": 0.0002, "enable_thinking": False,
               "model": "D:/models/Qwen3-0.6B", "modules": ["q_proj", "v_proj"]}
        # 不删临时文件, 沙箱的回收站拦截会让 TemporaryDirectory 清理失败
        p = ROOT / "runs" / "_tmp_configs" / "config_roundtrip.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        dump(src, p)
        self.assertEqual(load(p), src)

    def test_core_layer_does_not_import_yaml(self):
        for p in sorted((ROOT / "src").glob("**/*.py")):
            text = p.read_text(encoding="utf-8")
            self.assertNotIn("import yaml", text, p.relative_to(ROOT))


class TestPyCharmEntries(unittest.TestCase):
    """PyCharm 是右键直接跑文件, 不是 -m, 所以要按脚本方式验证。"""

    @staticmethod
    def _run(script: str, *args: str):
        import subprocess
        return subprocess.run([sys.executable, str(ROOT / script), *args],
                              cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")

    def test_main_entry_help(self):
        r = self._run("run_pycharm.py", "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--stage", r.stdout)

    def test_show_results_panel(self):
        r = self._run("show_results_pycharm.py")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("面试数字速记面板", r.stdout)
        self.assertIn("有效样本量", r.stdout)

    def test_module_can_run_as_plain_script(self):
        r = self._run("src/core/verifier.py")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("relative import", r.stderr)
        self.assertNotIn("ModuleNotFoundError", r.stderr)

    def test_three_generated_reports_exist(self):
        from src.core.results import GENERATED_REPORTS
        for name in GENERATED_REPORTS:
            self.assertTrue((ROOT / "reports" / name).exists(), name)


class TestAlgorithmLab(unittest.TestCase):
    """后训练算法实验台: 目标函数、梯度与锚点的可运行验证。"""

    @classmethod
    def setUpClass(cls):
        from src.model import algo_lab
        cls.al = algo_lab
        if not (ROOT / "data" / "preferences_deconfounded.jsonl").exists():
            raise unittest.SkipTest("preference data not generated yet")

    def test_analytic_gradient_matches_finite_difference(self):
        r = self.al.self_test()
        self.assertTrue(r["ok"])
        self.assertLess(r["gradient_max_abs_error"], 1e-6)
        self.assertEqual(r["kl_self"], 0.0)
        self.assertTrue(r["kl_nonnegative"])

    def test_softmax_and_kl_basic_properties(self):
        p = self.al.softmax([1.0, 2.0, -3.0])
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        self.assertTrue(all(x > 0 for x in p))
        c = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        a, b = [0.4, -0.2], [-0.3, 0.7]
        self.assertGreaterEqual(self.al.kl(a, b, c), 0.0)
        self.assertAlmostEqual(self.al.kl(a, a, c), 0.0, places=12)

    def test_probe_is_balanced_two_by_two(self):
        rows = _rows()
        probes = self.al.build_probe(rows, n=20, seed=11)
        self.assertGreater(len(probes), 10)
        for p in probes:
            f = p["feats"]
            self.assertEqual(len(p["hack_feats"]), 2)
            self.assertEqual(f[0][self.al.QUALITY], 1.0)
            self.assertEqual(f[1][self.al.QUALITY], 1.0)
            self.assertEqual(f[2][self.al.QUALITY], 0.0)
            self.assertEqual(f[3][self.al.QUALITY], 0.0)
            self.assertAlmostEqual(f[0][self.al.LENGTH], f[2][self.al.LENGTH], places=9)
            self.assertAlmostEqual(f[1][self.al.LENGTH], f[3][self.al.LENGTH], places=9)
            self.assertGreater(p["hack_feats"][1][self.al.LENGTH],
                               p["hack_feats"][0][self.al.LENGTH])

    def test_deconfounded_pairs_only_move_quality_direction(self):
        d = self.al.build_pairs("deconfounded", ROOT / "data")
        self.assertTrue(d)
        for row in d:
            self.assertAlmostEqual(row[self.al.LENGTH], 0.0, places=9)
            self.assertAlmostEqual(row[self.al.QUALITY], 1.0, places=9)
            for name in ("marker", "latex", "digits", "punct", "newline"):
                self.assertAlmostEqual(row[self.al.FEATURES.index(name)], 0.0, places=9)
        s = self.al.build_pairs("shortcut", ROOT / "data")
        self.assertGreater(sum(r[self.al.LENGTH] for r in s) / len(s), 0.1)

    def test_sft_then_dpo_reproduces_shortcut_hacking(self):
        rows = _rows()
        probes = self.al.build_probe(rows, n=30, seed=7)
        theta_sft, _ = self.al.train_sft(rows, 7, steps=60)
        before = self.al.probe_scores([0.0] * self.al.DIM, probes)
        after = self.al.probe_scores(theta_sft, probes)
        self.assertGreaterEqual(after["accuracy"], before["accuracy"])
        dec = self.al.build_pairs("deconfounded", ROOT / "data")
        sho = self.al.build_pairs("shortcut", ROOT / "data")
        r_dec = self.al.train_dpo(list(theta_sft), list(theta_sft), dec, 0.1,
                                  probes, steps=160)
        r_sho = self.al.train_dpo(list(theta_sft), list(theta_sft), sho, 0.1,
                                  probes, steps=160)
        self.assertAlmostEqual(r_dec["reward_accuracy"], 1.0, places=9)
        self.assertAlmostEqual(r_sho["reward_accuracy"], 1.0, places=9)
        self.assertGreater(r_sho["probe_hack_rate"], r_dec["probe_hack_rate"])
        self.assertGreater(r_dec["theta_quality"], r_sho["theta_quality"])
        self.assertGreater(r_sho["theta_length"], r_dec["theta_length"])

    def test_margin_over_beta_is_identity_at_fixed_theta(self):
        diffs = self.al.build_pairs("deconfounded", ROOT / "data")
        theta = [0.1] * self.al.DIM
        ref = [0.0] * self.al.DIM
        gap = [a - b for a, b in zip(theta, ref)]
        mean_delta = sum(self.al.dot(gap, d) for d in diffs) / len(diffs)
        ratios = [(beta * mean_delta) / beta for beta in (0.05, 0.1, 0.3, 0.5)]
        self.assertLess(max(ratios) - min(ratios), 1e-12)

    def test_wrong_reference_anchor_inflates_kl(self):
        rows = _rows()
        probes = self.al.build_probe(rows, n=30, seed=7)
        theta_sft, _ = self.al.train_sft(rows, 7, steps=60)
        diffs = self.al.build_pairs("deconfounded", ROOT / "data")
        good = self.al.train_dpo(list(theta_sft), list(theta_sft), diffs, 0.1,
                                 probes, steps=160)
        bad = self.al.train_dpo(list(theta_sft), [0.0] * self.al.DIM, diffs, 0.1,
                                probes, steps=160)
        self.assertGreater(bad["kl"], good["kl"])

    def test_curve_artifacts_exist(self):
        p = ROOT / "reports" / "algo_lab_curves.json"
        s = ROOT / "reports" / "algo_lab_curve.svg"
        if not p.exists():
            self.skipTest("run_research not executed yet")
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(len(data["curves"]), 3)
        self.assertTrue(s.exists())
        self.assertIn("<path", s.read_text(encoding="utf-8"))


class TestModelPilotStatic(unittest.TestCase):
    """权重层 pilot 的静态断言, 不需要 torch 也能验证口径没有被悄悄改掉。"""

    def test_pilot_script_and_config_exist(self):
        from pathlib import Path
        self.assertTrue((ROOT / "src" / "model" / "pilot.py").exists())
        self.assertTrue((ROOT / "configs" / "model_pilot.yaml").exists())
        self.assertTrue((ROOT / "run_model_pycharm.py").exists())

    def test_pilot_keeps_explicit_ref_and_pending_protocol(self):
        from src.core.config import load
        from pathlib import Path
        s = (ROOT / "src" / "model" / "pilot.py").read_text(encoding="utf-8")
        self.assertIn("assert_ref_matches_policy", s)
        self.assertIn("is_trainable=False", s)
        self.assertIn("model_pilot", s)
        self.assertIn("pending_model_run", s)
        self.assertNotIn("store.pending", s)
        c = load(ROOT / "configs" / "model_pilot.yaml")
        self.assertLess(int(c["max_new_tokens"]), 256)
        self.assertIn("D:/models/Qwen3-0.6B", c["base_model"])

    def test_run_model_entry_uses_chdir(self):
        s = (ROOT / "run_model_pycharm.py").read_text(encoding="utf-8")
        self.assertIn("os.chdir(ROOT)", s)
        self.assertIn("PYTHONPATH", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
