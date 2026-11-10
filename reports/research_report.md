# 技术报告：中文数学大模型后训练算法与可信评估

本报告由 `src/core/results.py` 从 `reports/results.jsonl` 生成，
表中每个数字都对应一条真实执行产生的记录。

## 项目定位

面向大模型算法岗的求职作品集。主线是一条完整的后训练链路：
参数化数据构造 → 严格三态验证器 → LoRA-SFT（assistant-only loss）→
偏好对构造 → DPO（显式冻结 SFT 参考模型）→ 序列级 KL 诊断 →
四个泛化子集上的生成评测。差异化在于把评测链路本身做成可估计的研究对象：
偏好信号可识别性、伪重复下的配对推断、验证器误判校正。

三层验证刻意分开。算法层用可解析梯度的 log-linear 策略在 CPU 上真实跑通
SFT 与 DPO，得到目标函数、梯度、KL、reward margin 与长度偏置的数字。
权重层用真实 Qwen3-0.6B 权重跑明确缩减规模的 CPU pilot，验证链路可执行。
完整协议层是 256 token、303 题的正式评测，本机 CPU 跑不完，
9 个指标记为 pending_model_run。三层绝不互相冒充。

## 研究问题

在小样本、合成偏好数据、自动验证器三者同时存在时，
后训练报告的提升有多少来自真实数学能力，又有多少来自表面线索、
伪重复和验证器误判。三个子贡献分别回答可识别性、统计推断与测量偏差。

## 实验台

- 数据集：`data`，
  共 600 题，12 个题型族，
  seed=20251109。
- 数据 SHA256：`c5bb982652077f9717952d1b234061b7683ef55b4d46e2c88095b5dd32bbc1cc`。
- 泄露检查：`True`。
- 算法层：`src/model/algo_lab.py`，7 维解析策略，梯度经有限差分校验；
  `reports/algo_lab_curve.svg` 是其真实训练曲线。
- 权重层 CPU pilot：`src/model/pilot.py`，真实 Qwen3-0.6B 权重，缩减规模写在
  `configs/model_pilot.yaml`，逐条结果与 adapter 哈希见 `model_pilot_manifest.json`。
- 完整协议模型阶段：本机 CPU 跑不完 303 题 × 256 token，
  9 个指标记为 pending_model_run，需在有 GPU 环境执行
  `python -m src.run_all --stage model` 后填入。

## 一、数据与评测协议

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| n_rows rows | 600 | - | - | measured |
| n_families families | 12 | - | - | measured |
| n_unique_questions questions | 600 | - | - | measured |
| n_unique_parameter_signatures signatures | 600 | - | - | measured |
| cross_split_overlap rows | 0 | - | - | measured |
| oracle_verifier_consistent_rows rows | 600 | - | 600 | measured |
| preference_rows_shortcut pairs | 756 | - | - | measured |
| preference_true_error_rate_shortcut rate | 0.0000 | - | 756 | measured |
| preference_format_match_shortcut rate | 0.3333 | - | 756 | measured |
| preference_abs_length_diff_shortcut chars | 27.7460 | - | 756 | measured |
| preference_rows_deconfounded pairs | 249 | - | - | measured |
| preference_true_error_rate_deconfounded rate | 1.0000 | - | 249 | measured |
| preference_format_match_deconfounded rate | 1.0000 | - | 249 | measured |
| preference_abs_length_diff_deconfounded chars | 0.0000 | - | 249 | measured |
| preference_rows_length_reversed pairs | 249 | - | - | measured |
| preference_true_error_rate_length_reversed rate | 1.0000 | - | 249 | measured |
| preference_format_match_length_reversed rate | 1.0000 | - | 249 | measured |
| preference_abs_length_diff_length_reversed chars | 22.0000 | - | 249 | measured |
| legacy_test_rows rows | 40 | - | - | measured |
| legacy_unique_questions questions | 2 | - | - | measured |
| legacy_pseudo_replication x | 20.0000 | - | - | measured |
| legacy_claimed_wrong_pairs pairs | 80 | - | - | measured |
| legacy_actually_wrong_pairs pairs | 0 | - | - | measured |
| legacy_length_diff_all_positive bool | 1 | - | - | measured |
| legacy_exact_equals_symbolic_rate rate | 1.0000 | - | 120 | measured |

## 二、后训练算法实验台（解析策略，非 Qwen 权重）

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| dpo_gradient_max_abs_error abs | 0.0000 | - | - | measured |
| gradient_check_passed bool | 1 | - | - | measured |
| n_probe_items items | 120 | - | - | measured |
| n_feature_dims dims | 7 | - | - | measured |
| sft_train_items items | 252 | - | - | measured |
| probe_accuracy_base rate | 0.5000 | - | 120 | measured |
| probe_accuracy_sft rate | 0.8399 | - | 120 | measured |
| sft_gain_over_base abs | 0.3399 | - | 120 | measured |
| sft_theta_quality coef | 1.6577 | - | - | measured |
| n_pairs_shortcut pairs | 756 | - | - | measured |
| loss_shortcut value | 0.6125 | - | 120 | measured |
| reward_margin_shortcut value | 0.1696 | - | 120 | measured |
| reward_accuracy_shortcut rate | 1.0000 | - | 120 | measured |
| kl_shortcut value | 0.0847 | - | 120 | measured |
| theta_length_shortcut value | 0.5651 | - | 120 | measured |
| theta_quality_shortcut value | 2.0335 | - | 120 | measured |
| probe_accuracy_shortcut rate | 0.8843 | - | 120 | measured |
| probe_long_rate_shortcut rate | 0.8171 | - | 120 | measured |
| probe_correct_within_length_shortcut value | 0.8843 | - | 120 | measured |
| probe_hack_rate_shortcut rate | 0.7241 | - | 120 | measured |
| dpo_gain_over_sft_shortcut abs | 0.0443 | - | 120 | measured |
| theta_quality_delta_shortcut coef | 0.3758 | - | - | measured |
| theta_length_delta_shortcut coef | 0.3181 | - | - | measured |
| quality_per_length_shortcut ratio | 1.1454 | - | - | measured |
| n_pairs_deconfounded pairs | 249 | - | - | measured |
| loss_deconfounded value | 0.6366 | - | 120 | measured |
| reward_margin_deconfounded value | 0.1165 | - | 120 | measured |
| reward_accuracy_deconfounded rate | 1.0000 | - | 120 | measured |
| kl_deconfounded value | 0.0513 | - | 120 | measured |
| theta_length_deconfounded value | 0.2470 | - | 120 | measured |
| theta_quality_deconfounded value | 2.8226 | - | 120 | measured |
| probe_accuracy_deconfounded rate | 0.9439 | - | 120 | measured |
| probe_long_rate_deconfounded rate | 0.6512 | - | 120 | measured |
| probe_correct_within_length_deconfounded value | 0.9439 | - | 120 | measured |
| probe_hack_rate_deconfounded rate | 0.1864 | - | 120 | measured |
| dpo_gain_over_sft_deconfounded abs | 0.1040 | - | 120 | measured |
| theta_quality_delta_deconfounded coef | 1.1648 | - | - | measured |
| theta_length_delta_deconfounded coef | 0.0000 | - | - | measured |
| quality_per_length_deconfounded ratio | 116.4842 | - | - | measured |
| n_pairs_length_reversed pairs | 249 | - | - | measured |
| loss_length_reversed value | 0.6293 | - | 120 | measured |
| reward_margin_length_reversed value | 0.1321 | - | 120 | measured |
| reward_accuracy_length_reversed rate | 1.0000 | - | 120 | measured |
| kl_length_reversed value | 0.0534 | - | 120 | measured |
| theta_length_length_reversed value | -0.0082 | - | 120 | measured |
| theta_quality_length_reversed value | 2.8178 | - | 120 | measured |
| probe_accuracy_length_reversed rate | 0.9436 | - | 120 | measured |
| probe_long_rate_length_reversed rate | 0.6190 | - | 120 | measured |
| probe_correct_within_length_length_reversed value | 0.9436 | - | 120 | measured |
| probe_hack_rate_length_reversed rate | 0.0720 | - | 120 | measured |
| dpo_gain_over_sft_length_reversed abs | 0.1037 | - | 120 | measured |
| theta_quality_delta_length_reversed coef | 1.1601 | - | - | measured |
| theta_length_delta_length_reversed coef | -0.2552 | - | - | measured |
| quality_per_length_length_reversed ratio | 4.3741 | - | - | measured |
| margin_over_beta_cv_fixed_theta cv | 0.0000 | - | - | measured |
| fixed_theta_mean_delta logratio | 0.0500 | - | - | measured |
| beta_0p05_kl nats | 0.0179 | - | 120 | measured |
| beta_0p05_probe_accuracy rate | 0.9049 | - | 120 | measured |
| beta_0p05_reward_margin value | 0.0298 | - | - | measured |
| beta_0p1_kl nats | 0.0513 | - | 120 | measured |
| beta_0p1_probe_accuracy rate | 0.9439 | - | 120 | measured |
| beta_0p1_reward_margin value | 0.1165 | - | - | measured |
| beta_0p3_kl nats | 0.1310 | - | 120 | measured |
| beta_0p3_probe_accuracy rate | 0.9886 | - | 120 | measured |
| beta_0p3_reward_margin value | 0.8416 | - | - | measured |
| beta_0p5_kl nats | 0.1455 | - | 120 | measured |
| beta_0p5_probe_accuracy rate | 0.9934 | - | 120 | measured |
| beta_0p5_reward_margin value | 1.6746 | - | - | measured |
| margin_over_beta_cv_after_training cv | 0.5727 | - | - | measured |
| beta_kl_range nats | 0.1276 | - | - | measured |
| beta_probe_accuracy_range abs | 0.0884 | - | - | measured |
| wrong_anchor_kl nats | 0.5179 | - | 120 | measured |
| anchor_kl_gap nats | 0.4666 | - | - | measured |
| anchor_probe_accuracy_gap abs | 0.0053 | - | 120 | measured |
| anchor_margin_gap value | 0.1562 | - | - | measured |

## 三、权重层 CPU pilot（真实 Qwen3-0.6B 权重，缩减协议）

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| mask_self_test_completion_tokens tokens | 33 | - | - | measured |
| test_loss_masked_base nats | 2.5602 | - | 5 | measured |
| generative_accuracy_base rate | 0.4000 | - | 5 | measured |
| unverifiable_rate_base rate | 0.6000 | - | 5 | measured |
| sft_steps steps | 20 | - | - | measured |
| sft_loss_first nats | 1.8023 | - | - | measured |
| sft_loss_last nats | 0.0530 | - | - | measured |
| test_loss_masked_sft nats | 1.2551 | - | 5 | measured |
| generative_accuracy_sft rate | 0.6000 | - | 5 | measured |
| unverifiable_rate_sft rate | 0.2000 | - | 5 | measured |
| ref_assert_checked_parameters params | 422 | - | - | measured |
| ref_is_frozen_sft_adapter bool | 1 | - | - | measured |
| dpo_loss_before_shortcut nats | 0.6931 | - | - | measured |
| dpo_loss_after_shortcut nats | 0.5305 | - | - | measured |
| reward_margin_before_shortcut value | 0.0000 | - | - | measured |
| reward_margin_after_shortcut value | 0.3617 | - | - | measured |
| reward_margin_gain_shortcut value | 0.3617 | - | - | measured |
| test_loss_masked_dpo_shortcut nats | 1.2602 | - | 5 | measured |
| generative_accuracy_dpo_shortcut rate | 0.6000 | - | 5 | measured |
| unverifiable_rate_dpo_shortcut rate | 0.2000 | - | 5 | measured |
| mean_pred_chars_dpo_shortcut chars | 71.8000 | - | - | measured |
| dpo_loss_before_deconfounded nats | 0.6931 | - | - | measured |
| dpo_loss_after_deconfounded nats | 0.6324 | - | - | measured |
| reward_margin_before_deconfounded value | 0.0000 | - | - | measured |
| reward_margin_after_deconfounded value | 0.1260 | - | - | measured |
| reward_margin_gain_deconfounded value | 0.1260 | - | - | measured |
| test_loss_masked_dpo_deconfounded nats | 1.2635 | - | 5 | measured |
| generative_accuracy_dpo_deconfounded rate | 0.6000 | - | 5 | measured |
| unverifiable_rate_dpo_deconfounded rate | 0.2000 | - | 5 | measured |
| mean_pred_chars_dpo_deconfounded chars | 71.8000 | - | - | measured |
| dpo_loss_before_length_reversed nats | 0.6931 | - | - | measured |
| dpo_loss_after_length_reversed nats | 0.4580 | - | - | measured |
| reward_margin_before_length_reversed value | 0.0000 | - | - | measured |
| reward_margin_after_length_reversed value | 0.5432 | - | - | measured |
| reward_margin_gain_length_reversed value | 0.5432 | - | - | measured |
| test_loss_masked_dpo_length_reversed nats | 1.2583 | - | 5 | measured |
| generative_accuracy_dpo_length_reversed rate | 0.6000 | - | 5 | measured |
| unverifiable_rate_dpo_length_reversed rate | 0.2000 | - | 5 | measured |
| mean_pred_chars_dpo_length_reversed chars | 71.8000 | - | - | measured |
| sequence_kl_mc nats | -0.0129 | - | 4 | measured |
| sequence_kl_n samples | 4 | - | - | measured |
| sequence_kl_ci_low nats | -0.0497 | - | 4 | measured |
| sequence_kl_ci_high nats | 0.0238 | - | 4 | measured |
| elapsed_seconds seconds | 477.6300 | - | - | measured |
| config_n_sft_steps count | 20 | - | - | measured |
| config_n_dpo_steps count | 10 | - | - | measured |
| config_n_pairs_per_group count | 6 | - | - | measured |
| config_n_eval_per_subset count | 1 | - | - | measured |
| config_max_new_tokens value | 96 | - | - | measured |
| config_beta value | 0.1000 | - | - | measured |
| config_lora_r value | 4 | - | - | measured |

## 四、子贡献 1：偏好信号可识别性

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| surface_auc_shortcut AUC | 1.0000 | - | 756 | measured |
| shortcut_index_shortcut SI | 1.0000 | - | 756 | measured |
| true_error_rate_shortcut rate | 0.0000 | - | 756 | measured |
| surface_auc_deconfounded AUC | 0.5000 | - | 249 | measured |
| shortcut_index_deconfounded SI | 0.0000 | - | 249 | measured |
| true_error_rate_deconfounded rate | 1.0000 | - | 249 | measured |
| surface_auc_length_reversed AUC | 1.0000 | - | 249 | measured |
| shortcut_index_length_reversed SI | 1.0000 | - | 249 | measured |
| true_error_rate_length_reversed rate | 1.0000 | - | 249 | measured |
| max_abs_change_under_Q_permute abs | 0.0000 | - | 200 | measured |
| max_abs_change_coupled_Q abs | 0.0016 | - | 200 | measured |
| margin_over_beta_cv CV | 0.0016 | - | 4 | measured |
| beta_sweep_test_loss_range nats | 0.0050 | - | 4 | measured |
| surface_auc_legacy AUC | 1.0000 | - | 240 | measured |

## 五、子贡献 2：伪重复与配对推断

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| legacy_n rows | 40 | - | - | measured |
| n_clusters clusters | 2 | - | - | measured |
| icc rho | 0.7368 | - | 40 | measured |
| design_effect deff | 15.0000 | - | 40 | measured |
| effective_n rows | 2.6667 | - | 40 | measured |
| iid_ci_low diff | 0.0212 | - | 40 | measured |
| iid_ci_high diff | 0.2288 | - | 40 | measured |
| cluster_ci_low diff | 0.0000 | [0.0000, 0.2500] | 2 | measured |
| cluster_ci_high diff | 0.2500 | - | 2 | measured |
| point_difference_sft_minus_base diff | 0.1250 | - | 40 | measured |
| mcnemar_p_base_sft p | 0.0625 | - | 40 | measured |
| holm_adjusted_min_p p | 0.1875 | - | - | measured |
| clusters_needed_for_5pp clusters | 99 | - | - | measured |
| power_at_K2 power | 0.0685 | - | - | measured |
| power_at_K5 power | 0.0969 | - | - | measured |
| power_at_K10 power | 0.1455 | - | - | measured |
| power_at_K20 power | 0.2441 | - | - | measured |
| power_at_K50 power | 0.5160 | - | - | measured |
| power_at_K100 power | 0.8074 | - | - | measured |
| new_effective_n_independence rows | 303.0000 | - | 303 | measured |
| new_effective_n_moderate rows | 36.6163 | - | 303 | measured |
| new_effective_n_legacy_measured rows | 16.0586 | - | 303 | measured |
| new_n_test rows | 303 | - | - | measured |
| new_n_family_clusters clusters | 12 | - | - | measured |

## 六、子贡献 3：验证器偏差校正

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| fpr_strict_poly rate | 0.0000 | [0.0000, 0.0064] | 750 | measured |
| fnr_strict_poly rate | 0.0000 | [0.0000, 0.0250] | 750 | measured |
| coverage_strict_poly rate | 0.4000 | - | 750 | measured |
| observed_accuracy_strict_poly rate | 0.6625 | - | 400 | measured |
| corrected_accuracy_strict_poly rate | 0.6625 | [0.6150, 0.7100] | 400 | measured |
| naive_bias_strict_poly bias | 0.0000 | - | 400 | measured |
| corrected_bias_strict_poly bias | 0.0000 | - | 400 | measured |
| ci_covers_truth_strict_poly bool | 1 | - | - | measured |
| fpr_numeric_float rate | 0.0050 | [0.0017, 0.0146] | 750 | measured |
| fnr_numeric_float rate | 0.0267 | [0.0104, 0.0666] | 750 | measured |
| coverage_numeric_float rate | 0.4000 | - | 750 | measured |
| observed_accuracy_numeric_float rate | 0.6475 | - | 400 | measured |
| corrected_accuracy_numeric_float rate | 0.6635 | [0.6127, 0.7165] | 400 | measured |
| naive_bias_numeric_float bias | -0.0150 | - | 400 | measured |
| corrected_bias_numeric_float bias | 0.0010 | - | 400 | measured |
| ci_covers_truth_numeric_float bool | 1 | - | - | measured |
| fpr_legacy_substring rate | 0.5000 | [0.4601, 0.5399] | 750 | measured |
| fnr_legacy_substring rate | 0.1267 | [0.0826, 0.1894] | 750 | measured |
| coverage_legacy_substring rate | 1.0000 | - | 750 | measured |
| observed_accuracy_legacy_substring rate | 0.7425 | - | 400 | measured |
| corrected_accuracy_legacy_substring rate | 0.6496 | [0.5099, 0.8050] | 400 | measured |
| naive_bias_legacy_substring bias | 0.0800 | - | 400 | measured |
| corrected_bias_legacy_substring bias | -0.0129 | - | 400 | measured |
| ci_covers_truth_legacy_substring bool | 1 | - | - | measured |
| survey_pi_true rate | 0.6625 | - | 400 | measured |
| observed_accuracy rate | 0.7425 | - | 400 | measured |
| corrected_accuracy rate | 0.6496 | - | 400 | measured |
| correction_denominator abs | 0.3733 | - | 400 | measured |
| latent_class_pi rate | 0.8682 | - | 297 | measured |
| latent_class_abs_error abs | 0.0241 | - | 297 | measured |
| latent_class_coverage rate | 0.7425 | - | 400 | measured |
| latent_class_selection_bias bias | 0.2298 | - | 297 | measured |
| latent_class_identified bool | 1 | - | - | measured |
| latent_class_pi_dependence_0.0 rate | 0.8682 | - | 297 | measured |
| latent_class_pi_dependence_0.2 rate | 0.8718 | - | 297 | measured |
| latent_class_pi_dependence_0.5 rate | 0.8753 | - | 297 | measured |
| latent_class_dependence_spread abs | 0.0072 | - | 297 | measured |

## 七、旧项目假象复算

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| old_reported_correct count | 65 | - | 120 | measured |
| strict_verifier_correct count | 33 | - | 120 | measured |
| strict_verifier_unverifiable count | 87 | - | 120 | measured |
| old_false_positives count | 32 | - | 65 | measured |
| old_false_positive_rate rate | 0.4923 | - | - | measured |
| ppl_sources_conflicting distinct values | 3 | - | - | measured |
| ppl_spread PPL | 42.0887 | - | - | measured |
| old_prompt_mask_present bool | 0 | - | - | measured |
| old_ref_model_none bool | 1 | - | - | measured |
| old_substring_matcher bool | 1 | - | - | measured |

## 八、Qwen3-0.6B 完整协议模型阶段（未运行）

| 指标 | 值 | 95% 区间 | n | 状态 |
|---|---:|---:|---:|---|
| base_test_loss_masked  | PENDING_MODEL_RUN | - | - | pending_model_run |
| sft_test_loss_masked  | PENDING_MODEL_RUN | - | - | pending_model_run |
| dpo_test_loss_masked  | PENDING_MODEL_RUN | - | - | pending_model_run |
| generative_accuracy_base  | PENDING_MODEL_RUN | - | - | pending_model_run |
| generative_accuracy_sft  | PENDING_MODEL_RUN | - | - | pending_model_run |
| generative_accuracy_dpo  | PENDING_MODEL_RUN | - | - | pending_model_run |
| dpo_kl_to_sft_ref  | PENDING_MODEL_RUN | - | - | pending_model_run |
| dpo_accuracy_deconfounded_minus_shortcut  | PENDING_MODEL_RUN | - | - | pending_model_run |
| beta_sweep_downstream_delta  | PENDING_MODEL_RUN | - | - | pending_model_run |

## 结论边界

`algo_lab` 的 measured 记录来自 CPU 上真实训练的 7 维解析策略，
用于验证 DPO 目标函数层面的机制，不是 Qwen3 权重训练结果。
`model_pilot` 的 measured 记录来自真实 Qwen3-0.6B 权重上的 CPU 缩减协议，
它证明整条链路可执行，规模写在 `configs/model_pilot.yaml` 与
`reports/model_pilot_manifest.json`，不用于能力结论，也不替换下面
完整协议的 9 个 `pending_model_run` 指标。其余 measured 记录来自
数据构造、验证器判定与统计估计量。每条记录的 note 都写明其口径。
