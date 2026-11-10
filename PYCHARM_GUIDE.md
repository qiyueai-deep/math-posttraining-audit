# PyCharm 运行指南

## 最短路径

1. 用 PyCharm 打开 `math-posttraining-project-v2` 这个目录。
2. Settings → Project → Python Interpreter 选任意 Python 3.10 及以上。核心层与算法实验台零第三方依赖，运行研究阶段、算法阶段和检查阶段不需要安装 `requirements.txt`。
3. 在 Project 面板右键 `run_pycharm.py`，点 Run。约 8 秒后 Run 窗口会打印八个阶段的进度和报告路径。
4. 想现场演示训练过程，右键 `algo_lab_pycharm.py`，会真实跑一遍 SFT 最大似然与 DPO 梯度下降，并打印三组偏好对照。

不需要设置 Script path、Working directory 或 Parameters。六个入口脚本都会把工作目录自动切到项目根，因此 PyCharm 的 Working directory 填什么都不影响 `data/` 与 `reports/` 的读写。这一点已经从 `C:\Users\24399\Desktop` 启动脚本验证过。

核心层与算法实验台真的零第三方依赖，连 PyYAML 都没用，配置读写由 `src/core/config.py` 自己处理。用 `python -S -m src.run_all --stage research` 禁用所有 site-packages 也能完整跑通，所以解释器只要 Python 3.10 以上就行，不用 `pip install -r requirements.txt`。

## 六个右键入口

| 文件 | 作用 | 需要安装依赖 |
|---|---|---|
| `run_pycharm.py` | 主入口，默认跑研究阶段。加 `--stage algo` 演示算法实验台，加 `--stage check` 跑三轮检查，加 `--stage model` 跑模型阶段，加 `--stage all` 跑全部 | 默认不需要 |
| `algo_lab_pycharm.py` | 现场跑 SFT → DPO，打印梯度校验、三组对照、beta 恒等式与参考模型锚点，并生成训练曲线 | 不需要 |
| `build_pdf_pycharm.py` | 把 `reports/tech_report.html` 打印成 `reports/tech_report.pdf`，自动找 Chrome 或 Edge | 不需要 |
| `run_model_pycharm.py` | 在真实 Qwen3-0.6B 权重上跑 CPU pilot，自动找 Python 3.12 与已装好的 CPU 版依赖，约 8 到 12 分钟 | 不需要 |
| `check_pycharm.py` | 一键三轮检查，等价于 `run_pycharm.py --stage check` | 不需要 |
| `show_results_pycharm.py` | 打印面试数字速记面板，从 `reports/results.jsonl` 读，不硬编码 | 不需要 |

`run_pycharm.py` 加参数的方式：Run → Edit Configurations → Parameters 里填 `--stage check`。

## 也可以按模块方式跑

PyCharm Terminal 里，工作目录设为项目根，执行：

```bash
python -m src.run_all --stage research
python -m src.run_all --stage algo
python -m src.run_all --stage pilot
python -m src.run_all --stage check
python -m tests.test_core -v
python -m src.model.algo_lab --self-test
python -m src.core.data_gen --output_dir data --seed 20251109
```

`src/` 下每个模块都带 sys.path 引导，直接右键某个文件运行也不会报 `attempted relative import with no known parent package`。例如右键 `src/model/teacher_forced.py` 并加参数 `--self-test`，可以在完全没有 torch 的机器上验证 prompt mask 的正确性；右键 `src/model/algo_lab.py --self-test` 可以验证 DPO 解析梯度与有限差分一致。

## 输出在哪里

| 文件 | 内容 |
|---|---|
| `reports/results.jsonl` | 单一事实源，所有数字 |
| `reports/tech_report.html` | 技术报告源文件，自包含，浏览器可直接打开 |
| `reports/tech_report.pdf` | 技术报告 A4 打印版 |
| `reports/model_pilot_manifest.json` | 权重层 pilot 的规模、环境、耗时与 adapter 哈希 |
| `reports/pilot_predictions_*.jsonl` | pilot 逐条生成输出与三态判定 |
| `reports/research_report.md` | 指标表，由 results.jsonl 生成 |
| `reports/algorithm_interview.md` | 大模型算法岗专项对答 |
| `reports/interview_guide.md` | 面试对答手册 |
| `reports/resume_bullets.md` | 简历成句 |
| `reports/resume_rewrite.md` | 简历逐句改写对照 |
| `reports/algo_lab_curves.json`, `reports/algo_lab_curve.svg` | 算法实验台的真实训练曲线与图 |
| `reports/check_report.json` | 三轮检查结果 |
| `data/manifest.json` | 数据版本、seed、SHA256、泄露检查 |

## 技术报告怎么更新

1. 右键 `run_pycharm.py`（默认 research 阶段），重新生成 `reports/tech_report.html`。
2. 右键 `build_pdf_pycharm.py`，调用本机 Chrome 更新 `reports/tech_report.pdf`。

找不到 Chrome 或 Edge 时，脚本会打印 HTML 路径。直接用浏览器打开，打印时选
「另存为 PDF」，纸张 A4、默认边距、关闭页眉页脚，效果一致。报告里的数字全部来自
`results.jsonl`，不要手工改 HTML，第二轮检查会重新渲染并逐字节比对。

## 三层验证与模型阶段

第一层是算法实验台，训练 7 维可解析梯度的 log-linear 策略，用来验证目标函数与优化行为，
不需要 GPU，也不是 Qwen3 权重结果。第二层是权重层 CPU pilot，在真实 Qwen3-0.6B
权重上跑缩减规模，本机有 CPU 版 torch 与完整权重，`run_model_pycharm.py` 会自动找到
Python 3.12 与旧项目 `.python-deps` 里的依赖（只读引用），直接右键即可跑通。
第三层是完整 256-token 协议，303 题在 CPU 上跑不完，需要 GPU 环境：

1. 在有 GPU 的机器上执行 `pip install -r requirements.txt`。
2. 准备完整的 Qwen3-0.6B 权重目录，包含 `config.json`、tokenizer 文件和 `.safetensors`。
3. 把 `configs/sft.yaml`、`configs/dpo.yaml`、`configs/eval.yaml` 里的 `model_name` / `base_model` / `test_file` 路径改成实际位置。
4. 在 PyCharm 里给 `run_pycharm.py` 加参数 `--stage model`，或在 Terminal 执行 `python -m src.run_all --stage model`。

模型阶段会依次完成 SFT、三组偏好数据各自的 DPO、assistant-only teacher-forced 评测、严格生成式评测、序列级 KL 诊断，以及以题型族为聚类单位的真实配对推断。跑完后 `results.jsonl` 里 9 个 `pending_model_run` 记录会被替换成 `measured`，报告自动重新渲染。

## 常见报错

| 现象 | 原因与处理 |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | 用了自己新建的 Run Configuration 且 Working directory 不对。改用右键 `run_pycharm.py`，或在配置里把 Working directory 设为项目根 |
| 找不到 `data/train.jsonl` 或 `reports/` | 同上。六个入口脚本已自动 chdir，若仍出现说明是从别处直接调用内部模块，先用右键入口跑一次 research |
| Run 窗口中文乱码 | Settings → Editor → File Encodings 全部设为 UTF-8，Run Configuration 的 VM/环境里加 `PYTHONIOENCODING=utf-8` |
| `模型阶段依赖缺失: torch, ...` | 正常提示，说明核心层正按预期工作。装依赖或继续只用 research、algo 与 check |
| 报告里出现 `PENDING_MODEL_RUN` | Qwen3 模型阶段未运行。这是设计，不是报错 |
| `SyntaxError` 或 `ImportError` 于 `src/core/*` | 先右键 `check_pycharm.py` 跑第一轮编译检查，确认文件没有被手工改动 |

## 面试前的演示顺序

右键 `algo_lab_pycharm.py` 现场跑一遍训练并看三组对照，右键 `show_results_pycharm.py` 看数字，右键 `check_pycharm.py` 确认三轮全 PASS，最后打开 `reports/algorithm_interview.md` 与 `reports/interview_guide.md` 过一遍追问。要给面试官演示可复现性时，右键 `run_pycharm.py` 重跑一次研究阶段，输出与之前逐字节一致。
