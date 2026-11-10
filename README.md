# 中文数学大模型后训练算法与可信评估

大模型算法岗求职作品集。主线是一条能讲清楚每个环节的后训练链路：

参数化题目生成 → 严格三态验证器 → LoRA-SFT（assistant-only loss）→
三组对照偏好数据 → DPO（显式冻结 SFT 参考模型）→ 序列级 KL 诊断 →
四个泛化子集上的生成评测。

差异化在于把评测链路本身做成可估计的研究对象：偏好信号可识别性、
伪重复下的配对推断、验证器误判校正。起点是一次真实的失败复现，
reward margin 单调上升而生成准确率退化，最后被解释成三重偏差的必然结果，
而不是调参问题。

三层验证刻意分开。算法层用可解析梯度的 log-linear 策略在 CPU 上真实跑通
SFT 最大似然与 DPO 梯度下降，给出目标函数、梯度、KL、reward margin、
beta 与参考模型锚点的数字。权重层用本机已有的 CPU 版 torch 与完整的
Qwen3-0.6B 权重跑一个明确缩减规模的 pilot，验证 LoRA-SFT、显式冻结 SFT ref
的 DPO、masked loss、三态验证器与生成评测确实可执行。完整协议层是
256 token、303 题、四个泛化子集上的正式评测，本机 CPU 跑不完，
9 个指标记为 `pending_model_run`。三层绝不互相冒充。

## 为什么要新建目录

原目录 `math-posttraining-project` 的所有文件保持内容与修改时间不变，
`reports/original_project_snapshot.sha256` 保存了 134 个文件的哈希，
三轮检查会逐字节复核。按你一贯的做法，改动一律在副本上进行。

## 在 PyCharm 里运行

用 PyCharm 打开 `math-posttraining-project-v2`，解释器选 Python 3.10 以上即可，
核心层与算法实验台零第三方依赖。在项目面板右键下面四个文件之一点 Run，
不需要配置 Script path、Working directory 或 Parameters。

| 右键文件 | 作用 |
|---|---|
| `run_pycharm.py` | 主入口，默认跑研究阶段。Parameters 加 `--stage algo` / `--stage check` / `--stage model` / `--stage all` 切换 |
| `algo_lab_pycharm.py` | 现场跑 SFT → DPO 算法实验台，打印三组对照并生成训练曲线图 |
| `build_pdf_pycharm.py` | 把技术报告 HTML 打印成 PDF，自动找本机 Chrome |
| `run_model_pycharm.py` | 在真实 Qwen3-0.6B 权重上跑 CPU pilot，自动找本机 Python 3.12 与已装好的 CPU 版依赖 |
| `check_pycharm.py` | 一键三轮检查 |
| `show_results_pycharm.py` | 打印面试数字速记面板 |

六个入口都会把工作目录自动切到项目根，PyCharm 的 Working directory 填什么都不会出现找不到 `data/` 的问题。`src/` 下每个模块也带了 sys.path 引导，直接右键单个文件运行不会报相对导入错误。完整配置、参数与报错处理见 `PYCHARM_GUIDE.md`。

## 快速开始

算法层与研究层只用 Python 标准库，不需要 GPU，不需要 torch。

```bash
python -m src.run_all --stage research     # 生成数据 + 算法实验台 + 三个子贡献 + 出报告
python -m src.run_all --stage algo         # 单独演示算法实验台
python -m src.run_all --stage check        # 三轮检查
python -m tests.test_core -v               # 单独跑单元测试
python run_model_pycharm.py               # 真实 Qwen3-0.6B 权重上的 CPU pilot，约 8 到 12 分钟
```

研究阶段约 8 秒，产出：

- `data/` 600 道参数化中文数学题，12 个题型族，跨 split 零重叠
- `reports/results.jsonl` 单一事实源，研究阶段 194 条记录，权重层 pilot 记录另计
- `reports/research_report.md`、`reports/resume_bullets.md`、`reports/resume_rewrite.md`、`reports/algorithm_interview.md`，全部由 results.jsonl 生成
- `reports/algo_lab_curves.json` 与 `reports/algo_lab_curve.svg`，真实训练曲线
- `reports/model_pilot_manifest.json`、`reports/pilot_predictions_*.jsonl` 与 `runs/pilot/` 下的 adapter，权重层 pilot 产物
- `reports/tech_report.html` 技术报告（自包含，含 11 张表与 4 张内嵌 SVG 图），
  `reports/tech_report.pdf` 是其 A4 打印版，右键 `build_pdf_pycharm.py` 重新生成
- `reports/interview_guide.md` 面试对答手册，数字同样逐条可追溯
- `reports/check_report.json` 三轮检查结果

模型阶段需要 GPU 环境。

```bash
pip install -r requirements.txt
python -m src.run_all --stage model
```

本机其实有完整的 Qwen3-0.6B 权重和 CPU 版 torch（在旧项目 `.python-deps` 里，
只读引用，不改动原项目），所以权重层的 CPU pilot 可以直接跑。默认的 Python 3.13
解释器没有这些依赖，`run_model_pycharm.py` 会自动切到可用的 Python 3.12。
完整 256-token 协议在 CPU 上跑不完，仍需 GPU；在它跑完之前，`results.jsonl` 里
9 个 `model_stage` 指标保持 `pending_model_run`，报告中显示 `PENDING_MODEL_RUN`，
pilot 的实测数字记在 `model_pilot` 实验里，绝不替换它们。

## 目录

| 路径 | 内容 |
|---|---|
| `src/core/expressions.py` | 受限语法的精确多项式解析，解析失败显式抛错 |
| `src/core/verifier.py` | 三态验证器，含复刻旧缺陷的研究对照验证器 |
| `src/core/data_gen.py` | 12 个题型族，参数唯一，oracle 双路径交叉验证 |
| `src/core/splits.py` | 按族切分与参数签名级泄露检查 |
| `src/core/preferences.py` | 三组偏好对：shortcut / deconfounded / length_reversed |
| `src/core/results.py` | 单一事实源与报告渲染，含一致性校验 |
| `src/core/tech_report.py` | 技术报告 HTML，自包含样式、内嵌 SVG 图，数字全部来自结果文件 |
| `src/model/algo_lab.py` | CPU 可跑的 SFT → DPO 算法实验台，解析梯度、精确 KL、三组对照、beta 扫描、锚点对照 |
| `src/model/pilot.py`, `configs/model_pilot.yaml` | 真实 Qwen3-0.6B 权重上的 CPU pilot，缩减规模、环境、adapter 哈希全部落盘 |
| `src/model/train_sft.py`, `train_dpo.py` | Qwen3-0.6B 的 LoRA-SFT 与显式 ref 的 DPO |
| `src/model/teacher_forced.py`, `diagnostics.py`, `evaluate.py` | assistant-only loss、序列级 KL 估计、严格生成评测 |
| `src/research/identifiability.py` | 子贡献 1：表面特征 AUC、SI、损失不变性 |
| `src/research/cluster_inference.py` | 子贡献 2：ICC、deff、聚类 bootstrap、McNemar、功效 |
| `src/research/verifier_bias.py` | 子贡献 3：FPR/FNR 校正、三 verifier 潜类别 EM |
| `src/research/legacy_audit.py` | 旧项目真实文件复算审计 |
| `tests/run_three_round_checks.py` | 三轮检查 |

## 面试与简历

| 文件 | 内容 |
|---|---|
| `reports/tech_report.pdf` | 技术报告：问题与起点、后训练链路、算法实验台、权重层 CPU pilot、三个测量问题、讨论、复现 |
| `reports/algorithm_interview.md` | 算法岗专项：后训练链路逐环节、白板推导、三组对照数字、11 个算法追问 |
| `reports/interview_guide.md` | 30 秒陈述、2 分钟讲述、白板三块、数字速记、35 个高频追问与回答、不能说的话 |
| `reports/resume_rewrite.md` | 项目名改写、旧 bullet 逐句处置、可直接粘贴的成句（含空格全字符数已核对）、自评与技能栏、禁用清单 |
| `reports/resume_bullets.md` | 精简版成句，数字同样来自 results.jsonl |

四份文档里的四位小数都从 `results.jsonl` 渲染或逐条反查，改了数据源要重跑 `run_pycharm.py` 让它们同步，第二轮检查会自动抓出手写数字。

## 诚实边界

`algo_lab` 的 measured 记录来自 CPU 上真实训练的 7 维解析策略，
用于验证 DPO 目标函数与优化行为，不是权重训练结果。
`model_pilot` 的 measured 记录来自真实 Qwen3-0.6B 权重上的 CPU 缩减协议，
它证明整条链路可执行，规模写在 `configs/model_pilot.yaml` 与
`reports/model_pilot_manifest.json`，不用于能力结论，也不替换完整协议的
9 个 `pending_model_run` 指标。其余 measured 记录来自数据构造、
验证器判定与统计估计量。每条记录的 note 都写明其口径。

涉及完整协议下 Base / SFT / DPO 三方对比的 9 个指标全部是 `pending_model_run`。
旧项目那组无法由保存的 adapter 复现的 loss 与 PPL 数字已从新项目彻底移除。
简历里「难度分层」「答案规范化」这类没有真实实现的描述也已删除，
新数据集里的对应能力可以在 `data/manifest.json` 查到。

这是转行期间自修的工程作品集，与博士后期间的分布依赖随机过程研究分栏表述，
不合并叙述。

## 三轮检查

1. 全量语法编译加 41 个单元测试，覆盖表达式、验证器三态、数据泄露、
   偏好去混淆、统计估计量、prompt mask、模型层静态断言、零依赖配置、
   PyCharm 右键入口、算法实验台的梯度有限差分校验与权重层 pilot 记录。
2. 报告与四份面试/简历文档里的数字逐条反查 `results.jsonl`、技术报告 HTML
   重新渲染逐字节比对、数据 SHA256、单一事实源一致性、pilot manifest 完整性、
   源码与配置里不得残留旧指标硬编码。
3. 逐条断言 8 个致命问题的修复、3 个子贡献、算法实验台与权重层 pilot 的
   measured 记录、技术报告章节完整性、PyCharm 右键入口与零依赖配置，
   以及原项目 134 个文件未被改动。
