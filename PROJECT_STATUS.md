# 修复对照与交付状态

项目定位是大模型算法岗求职作品集，主线是完整的中文数学后训练链路，
差异化是评测链路的可信度研究。原项目 134 个文件的哈希保存在
`reports/original_project_snapshot.sha256`，三轮检查第三轮会逐字节复核，
交付时原项目未改动。

## 八个致命问题

| # | 旧项目的证据 | 新实现 | 自动验证 |
|---|---|---|---|
| 1 | `build_dataset.py` 只有 8 个手写模板，train 80 条实际 4 道题，test 40 条实际 2 道题，`validate_dataset.py` 的字符串去重被编号后缀骗过 | `src/core/data_gen.py` 12 个参数化题型族，每题唯一参数组，oracle 用第二条计算路径交叉验证。`src/core/splits.py` 以参数签名为单位查重，并额外做去标点题干去重 | 三轮检查 P1，单元测试 `test_no_leakage_and_unique_questions`，`data/manifest.json` 里 600 题 600 个唯一题干，跨 split 零重叠 |
| 2 | `evaluate.py` 的 `norm(ans) in norm(pred)` 子串包含判对，`symbolic()` 解析失败静默回退字符串匹配，实测 old 判对 65 条里 32 条是假阳性，exact 与 symbolic 逐条 120/120 相同 | `src/core/expressions.py` 稀疏多项式精确相等，解析失败显式抛 `ParseError`。`src/core/verifier.py` 三态 correct / incorrect / unverifiable，UNVERIFIABLE 单独统计 | 三轮检查 P2，单元测试 `test_no_substring_containment`、`test_three_states`，报告里 `old_false_positive_rate=0.4923` |
| 3 | `evaluate_teacher_forced.py` 用 `labels=input_ids.clone()`，对题干也算 loss，模型记住重复题干即可拉低 loss | `src/model/teacher_forced.py` 用 `offset_mapping` 字符级定位 prompt 边界，只对 assistant completion 算 token 加权 NLL 与 token accuracy。`train_sft.py` 用同一 mask 做训练与验证 | 三轮检查 P3，单元测试 `test_mask_covers_prompt` 与 `test_model_scripts_contain_required_fixes`，`--self-test` 可无 torch 运行 |
| 4 | 240 条偏好对里 0 条含真实数学错误，`wrong_final_answer` 用 `ans.replace("4","5",1)` 对 21/10/1/1,2 全部无效，chosen 恒长 18 到 28 字符 | `src/core/preferences.py` 三组对照，shortcut 组复刻旧缺陷，deconfounded 组长度差严格为 0、格式相同、真错误率 1.0，length_reversed 组反转长度线索 | 三轮检查 P4，单元测试 `test_deconfounded_is_length_and_format_matched`，报告里 surface AUC 从 1.0000 降到 0.5000 |
| 5 | β 扫描的 margin 从 0.1685 到 1.6781 恰好与 β 同比例，是 `margin=beta×log-ratio` 恒等式，test loss 只动了 0.005 | `src/research/identifiability.py` 报告 `margin_over_beta_cv=0.0016` 并指出恒等式，`src/model/diagnostics.py` 改为估计真正的序列级 KL 及其按 prompt 聚类的 bootstrap 区间 | 三轮检查 P5，报告里 `margin_over_beta_cv` |
| 6 | `train_dpo.py` 与 `sweep_beta.py` 都把参考模型参数置空，TRL 会 disable adapter 把 KL 锚到 base，`runs/qwen3-dpo/ref/` 的 SFT 快照从未被引用 | `src/model/train_dpo.py` 用独立基座实例加载冻结的 SFT adapter 显式传给 `ref_model`，训练前逐参数断言 ref 与 policy 初始一致，并保存真正被使用的 ref 快照 | 三轮检查 P6，`assert_ref_matches_policy` |
| 7 | `mine_errors.py` 用 `max_new_tokens=64`，32 条错误全是解码截断造成的 `missing_format`，构造的 rejected 平均比 chosen 长 75 字符，偏好方向反转 | 取消「截断即错误」逻辑，`configs/eval.yaml` 固定 `max_new_tokens: 256`，`src/model/evaluate.py` 用三态 verifier 单独统计 UNVERIFIABLE | 三轮检查 P7，源码中无 `missing_format` 分类 |
| 8 | 同一模型的 PPL 在三份报告写成 12.18 / 54.27 / 53.99，`qwen_training_blocker.md` 说没跑过 Qwen，`README` 说 demo 不能进简历但 `resume_bullets.md` 已写入，仓库无 git，`smoke_test.py` 会用 4 变式覆盖正式数据 | `src/core/results.py` 单一事实源 `results.jsonl`，所有 md 由脚本渲染，`verify_consistency` 重新渲染逐字节比对，`data/manifest.json` 记录 SHA256 与 seed，smoke 不再写正式数据目录 | 三轮检查 P8 与第二轮全部检查，报告里 `ppl_sources_conflicting=3` |

## 三个子贡献

| 子贡献 | 实现 | 真实结果 |
|---|---|---|
| 1 偏好信号可识别性 | 只用表面特征的 5 折 pair-grouped 交叉验证 AUC，SI 指标，DPO 损失在随机重排真实质量下不变的数值佐证，命题与证明见 `THEORY.md` | shortcut AUC 1.0000，deconfounded 0.5000，旧项目 240 条 AUC 1.0000，Q 重排下损失变化 0，耦合 Q 时变化非零 |
| 2 伪重复与配对推断 | ICC 的 ANOVA 估计，设计效应与有效样本量，聚类 bootstrap，配对 McNemar 精确检验，Holm 校正，功效曲线 | 旧 40 条评测只有 2 个簇，ICC 0.7368，deff 15，有效样本量 2.6667，iid 区间 [0.0212, 0.2288] 对比聚类区间 [0.0000, 0.2500]，检出 5 个百分点需要约 99 个独立簇 |
| 3 验证器偏差校正 | 真值已知的 calibration 与 survey，FPR/FNR 的 Wilson 区间，Rogan-Gladen 型矩估计与 bootstrap 区间，三 verifier 潜类别 EM 及条件独立敏感性 | 旧式宽松验证器 FPR 0.5000，观测准确率 0.7425 相对真值 0.6625 高估 0.0800，校正后 0.6496 偏差 -0.0129，区间覆盖真值。三 verifier 时 7 个自由频率恰好识别 7 个参数，丢弃 UNVERIFIABLE 造成 0.2298 的选择偏差 |

## 后训练算法实验台（CPU 真实运行）

`src/model/algo_lab.py` 把策略写成候选集上的 7 维 log-linear 模型，配分函数可算，
因此 SFT 的 MLE 梯度、DPO 的梯度与序列级 KL 都有闭式解。这一层验证的是
目标函数与优化行为，不是 Qwen3 权重训练结果，每条记录的 note 都写明这一点。

| 实验 | 真实结果 |
|---|---|
| 梯度校验 | 解析梯度与中心有限差分最大绝对误差 3.44e-11，`kl(θ\|\|θ)=0` |
| SFT → DPO | 120 道 2x2 平衡对抗探针，252 条 SFT 指令，三组偏好对照全部真跑 |
| 三组对照 | reward accuracy 均为 1.0000、margin 均上升，但长度偏置劫持率 shortcut 0.7241、deconfounded 0.1864、length_reversed 0.0720 |
| 系数分解 | 质量系数 2.0335 对 2.8226，长度系数 0.5651 对 0.2470，主探针准确率 0.8843 对 0.9439 |
| 参考模型锚点 | ref=SFT 时 KL 0.0513，锚错到 base 时 0.5179，差 0.4666 |
| beta | 固定策略 margin/beta 变异系数 0.0000（恒等式），真实训练后 0.5727，四个 beta 的 KL 跨度 0.1276 |
| 产物 | `reports/algo_lab_curves.json` 与 `reports/algo_lab_curve.svg`，纯标准库生成 |

三轮检查 P11 断言梯度校验、恒等式、三组对照方向与锚点差异，
单元测试 `TestAlgorithmLab` 覆盖有限差分、KL 性质、2x2 探针平衡与端到端训练。

## 模型阶段

算法层已如上实跑。以下指标属于 Qwen3-0.6B 权重层，在本机保持
`pending_model_run`，因为本机没有 torch/transformers/peft/trl。
脚本已经写好，在有 GPU 的环境执行 `python -m src.run_all --stage model` 即可填充，
填充逻辑会把同名 pending 记录替换成 measured 并重新渲染报告。

- Base / SFT / DPO 的 assistant-only test loss、PPL、token accuracy
- 四个泛化子集上的生成准确率与 UNVERIFIABLE 率
- 三组偏好训练的 DPO 下游对照
- 序列级 KL 及其聚类 bootstrap 区间
- 以题型族为聚类单位的真实配对推断

在它完成之前，这 9 个数字不得写入简历或对外宣称。

## 权重层 CPU pilot（真实 Qwen3-0.6B 权重）

本机其实有完整权重 `D:/models/Qwen3-0.6B`，旧项目 `.python-deps` 里也有
torch 2.15.0+cpu、transformers 5.22、peft 0.21、trl 1.17。用 Python 3.12
只读引用这些依赖即可运行，不需要重新安装包，也不改动原项目。
`src/model/pilot.py` 在 `configs/model_pilot.yaml` 写明的缩减规模下真实执行了
LoRA-SFT、显式冻结 SFT ref 的 DPO、assistant-only masked NLL、
严格三态生成评测与序列级 KL 的采样估计。

| 项 | 真实结果 |
|---|---|
| SFT 20 步的训练 loss | 1.8023 → 0.0530 |
| Base / SFT 的 masked test NLL | 2.5602 → 1.2551 |
| Base / SFT 的生成准确率（5 题） | 0.4000 → 0.6000，unverifiable 率 0.6000 → 0.2000 |
| 参考模型逐参数断言 | 422 个共享参数与 policy 初值一致，ref 冻结 |
| 三组 DPO 的 margin 增益 | shortcut 0.3617，deconfounded 0.1260，length_reversed 0.5432 |
| 序列级 KL 的 MC 估计 | -0.0129，区间 [-0.0497, 0.0238]，只有 4 条采样序列 |
| 环境与产物 | 477.63 秒，8 线程 CPU；4 个 adapter 哈希与逐条预测落在 `model_pilot_manifest.json` |

三组在 5 道题上的生成准确率完全相同，这是缩减规模的必然结果，不是结论。
KL 的 MC 估计为负同样是 4 条采样序列的噪声，KL 本身非负，报告里已写明。
9 个 `model_stage` 指标仍是 `pending_model_run`，pilot 数字记在
`model_pilot` 实验里，绝不替换它们。三轮检查 P13 断言 pilot 记录完整、
ref 冻结、且完整协议保持 pending；重跑研究阶段会原样保留 pilot 记录。

## 技术报告

`src/core/tech_report.py` 从 `results.jsonl` 与 `algo_lab_curves.json` 渲染一份
自包含 HTML（无 CDN、无外部字体、内嵌 SVG），再由 `build_pdf_pycharm.py` 调用
本机 Chrome 无头打印成 A4 PDF。报告 9 页，六节：问题与起点、后训练链路、
算法机制实验台、三个测量问题、讨论、复现，含 8 张表与 4 张图
（三组 margin 曲线、两组柱状图、iid 与聚类区间对比）。

| 产物 | 说明 |
|---|---|
| `reports/tech_report.html` | 源文件，浏览器可直接打开，数据变了重跑 research 即更新 |
| `reports/tech_report.pdf` | A4 打印版，右键 `build_pdf_pycharm.py` 重新生成 |
| `src/core/tech_report.py` | 渲染器，所有数字从结果文件读取，不手写任何指标 |

第二轮检查会重新渲染 HTML 并与磁盘逐字节比对，第三轮 P12 断言六节齐全、
含待运行标注，所以手改报告里的数字或结论会被抓出来。

## PyCharm 可运行性与求职材料

`src/` 下全部相对导入已改成绝对导入，每个模块带 sys.path 引导，直接右键单个文件不会报
`attempted relative import with no known parent package`。项目根目录五个右键入口：

| 文件 | 作用 |
|---|---|
| `run_pycharm.py` | 主入口，默认 research，可加 `--stage algo` / `--stage check` / `--stage model` / `--stage all` |
| `algo_lab_pycharm.py` | 现场跑 SFT → DPO 算法实验台并生成训练曲线 |
| `build_pdf_pycharm.py` | 把技术报告 HTML 打印成 PDF，自动找本机 Chrome |
| `check_pycharm.py` | 一键三轮检查 |
| `show_results_pycharm.py` | 打印面试数字速记面板，数字从 results.jsonl 读 |

五个入口都主动 `os.chdir(ROOT)`，PyCharm 的 Working directory 无论填什么都不会找不到
`data/`。配置、参数、报错处理见 `PYCHARM_GUIDE.md`。三轮检查第三轮 P9 与 P10
断言入口存在、自动 chdir、`src/` 无相对导入，以及四份求职文档齐全。

求职材料按算法岗重新定位：

| 文件 | 内容 |
|---|---|
| `reports/algorithm_interview.md` | 算法岗专项，后训练链路逐环节、白板推导、三组对照、11 个算法追问 |
| `reports/interview_guide.md` | 30 秒陈述、2 分钟讲述、白板三块、数字速记、10 个算法岗必答加 24 个项目与统计追问、不能说的话 |
| `reports/resume_rewrite.md` | 项目名改为算法链路在前，前两条 bullet 讲算法与工程，后三条讲差异化研究 |
| `reports/resume_bullets.md` | 精简版成句 |

项目名里的「自学工程项目」标签已去掉，改为在面试口述中直接讲链路环节与验证边界。
四份文档的数字都由
`results.jsonl` 渲染或逐条反查，纳入单一事实源的一致性校验。
