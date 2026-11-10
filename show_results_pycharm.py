"""PyCharm 数字速记面板，右键 Run 即可，不需要任何参数。

面试前跑一下，把所有能对答的数字看一遍。数字全部从 reports/results.jsonl 读，
不在脚本里硬编码，因此用手改过的结果不会出现在这里。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.core.results import ResultStore  # noqa: E402

SECTIONS = [
    ("后训练算法实验台（CPU 实跑，解析策略，非 Qwen 权重）", [
        ("algo_lab", "n_probe_items", "对抗探针题数"),
        ("algo_lab", "sft_train_items", "SFT 训练样本数"),
        ("algo_lab", "dpo_gradient_max_abs_error", "DPO 梯度有限差分误差"),
        ("algo_lab", "probe_accuracy_sft", "SFT 后主探针准确率"),
        ("algo_lab", "reward_accuracy_shortcut", "shortcut 组 reward accuracy"),
        ("algo_lab", "reward_accuracy_deconfounded", "去混淆组 reward accuracy"),
        ("algo_lab", "probe_hack_rate_shortcut", "shortcut 组长度偏置劫持率"),
        ("algo_lab", "probe_hack_rate_deconfounded", "去混淆组长度偏置劫持率"),
        ("algo_lab", "theta_quality_shortcut", "shortcut 组质量系数"),
        ("algo_lab", "theta_quality_deconfounded", "去混淆组质量系数"),
        ("algo_lab", "margin_over_beta_cv_fixed_theta", "固定策略 margin/beta 变异系数"),
        ("algo_lab", "kl_deconfounded", "正确参考模型锚点的 KL"),
        ("algo_lab", "wrong_anchor_kl", "锚错到 base 的 KL"),
        ("algo_lab", "anchor_kl_gap", "锚点造成的 KL 差"),
    ]),
    ("权重层 CPU pilot（真实 Qwen3-0.6B 权重，缩减协议，非能力结论）", [
        ("model_pilot", "config_n_sft_steps", "SFT 步数"),
        ("model_pilot", "config_n_dpo_steps", "每组 DPO 步数"),
        ("model_pilot", "config_max_new_tokens", "生成 token 上限"),
        ("model_pilot", "sft_loss_first", "SFT 首步 loss"),
        ("model_pilot", "sft_loss_last", "SFT 末步 loss"),
        ("model_pilot", "test_loss_masked_base", "Base masked test NLL"),
        ("model_pilot", "test_loss_masked_sft", "SFT masked test NLL"),
        ("model_pilot", "generative_accuracy_base", "Base 生成准确率"),
        ("model_pilot", "generative_accuracy_sft", "SFT 生成准确率"),
        ("model_pilot", "unverifiable_rate_base", "Base 不可核验率"),
        ("model_pilot", "reward_margin_gain_deconfounded", "去混淆组 margin 增益"),
        ("model_pilot", "sequence_kl_mc", "序列级 KL 的 MC 估计"),
        ("model_pilot", "elapsed_seconds", "pilot 耗时（秒）"),
    ]),
    ("实验台与旧项目审计", [
        ("data_audit", "n_rows", "新数据集题目数"),
        ("data_audit", "n_families", "题型族数"),
        ("data_audit", "n_unique_questions", "唯一题干数"),
        ("data_audit", "cross_split_overlap", "跨 split 参数重叠"),
        ("data_audit", "legacy_unique_questions", "旧测试集唯一题干数"),
        ("data_audit", "legacy_pseudo_replication", "旧伪重复倍数"),
        ("data_audit", "legacy_actually_wrong_pairs", "旧 80 条错误答案里真错的条数"),
        ("data_audit", "legacy_exact_equals_symbolic_rate", "旧 exact 与 symbolic 相同比例"),
        ("legacy_audit", "old_false_positive_rate", "旧评测器假阳性率"),
        ("legacy_audit", "ppl_sources_conflicting", "同一模型 PPL 的不同取值个数"),
    ]),
    ("子贡献 1  偏好信号可识别性", [
        ("identifiability", "surface_auc_shortcut", "shortcut 组表面特征 AUC"),
        ("identifiability", "surface_auc_deconfounded", "去混淆组表面特征 AUC"),
        ("identifiability", "surface_auc_legacy", "旧项目 240 条 AUC"),
        ("identifiability", "max_abs_change_under_Q_permute", "重排 Q 时 DPO 损失变化"),
        ("identifiability", "max_abs_change_coupled_Q", "耦合 Q 时损失变化"),
        ("identifiability", "margin_over_beta_cv", "margin/beta 变异系数"),
        ("identifiability", "beta_sweep_test_loss_range", "beta 扫描 test loss 变动"),
    ]),
    ("子贡献 2  伪重复与配对推断", [
        ("cluster_inference", "n_clusters", "旧评测的真实簇数"),
        ("cluster_inference", "icc", "簇内相关 ICC"),
        ("cluster_inference", "design_effect", "设计效应 deff"),
        ("cluster_inference", "effective_n", "有效样本量"),
        ("cluster_inference", "iid_ci_low", "iid 区间下界"),
        ("cluster_inference", "iid_ci_high", "iid 区间上界"),
        ("cluster_inference", "cluster_ci_low", "聚类区间下界"),
        ("cluster_inference", "cluster_ci_high", "聚类区间上界"),
        ("cluster_inference", "mcnemar_p_base_sft", "McNemar 精确 p"),
        ("cluster_inference", "holm_adjusted_min_p", "Holm 校正后最小 p"),
        ("cluster_inference", "clusters_needed_for_5pp", "检出 5pp 所需独立簇数"),
    ]),
    ("子贡献 3  验证器偏差校正", [
        ("verifier_bias", "fpr_legacy_substring", "旧式验证器 FPR"),
        ("verifier_bias", "survey_pi_true", "构造真值 pi"),
        ("verifier_bias", "observed_accuracy", "观测准确率"),
        ("verifier_bias", "corrected_accuracy", "校正后准确率"),
        ("verifier_bias", "naive_bias_legacy_substring", "未校正偏差"),
        ("verifier_bias", "corrected_bias_legacy_substring", "校正后偏差"),
        ("verifier_bias", "latent_class_pi", "三 verifier 潜类别估计"),
        ("verifier_bias", "latent_class_selection_bias", "丢弃 UNVERIFIABLE 的选择偏差"),
        ("verifier_bias", "latent_class_dependence_spread", "条件独立敏感性幅度"),
    ]),
]


def main() -> int:
    path = ROOT / "reports" / "results.jsonl"
    if not path.exists():
        print("还没有结果，请先右键运行 run_pycharm.py（默认 research 阶段）。")
        return 1
    records = ResultStore.load(path)
    idx = {(r["experiment"], r["metric"]): r for r in records}
    pending = [r for r in records if r["status"] == "pending_model_run"]

    print("=" * 78)
    print("面试数字速记面板    数据源:", path)
    print("=" * 78)
    for title, keys in SECTIONS:
        print()
        print(title)
        print("-" * 78)
        for exp, met, label in keys:
            r = idx.get((exp, met))
            if r is None:
                print(f"  {label:<34} 缺失记录")
                continue
            v = r["value"]
            if v is None:
                s = "PENDING_MODEL_RUN"
            elif isinstance(v, float):
                s = f"{v:.2e}" if 0 < abs(v) < 1e-4 else f"{v:.4f}"
            else:
                s = str(v)
            ci = ""
            if r["ci_low"] is not None:
                ci = f"   95% [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            print(f"  {label:<34} {s:>12}{ci}")
    print()
    print("=" * 78)
    print(f"模型阶段尚未运行的指标: {len(pending)} 个，一律记为 pending_model_run。")
    print("面试中不要报出任何 Base/SFT/DPO 的训练效果数字，也不要复述旧项目那组")
    print("无法由保存的 adapter 复现的 loss 与 PPL 数字。")
    print("算法岗专项问答见 reports/algorithm_interview.md，完整口径见")
    print("reports/interview_guide.md 与 reports/resume_rewrite.md")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
