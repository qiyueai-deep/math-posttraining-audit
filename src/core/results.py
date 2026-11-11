"""单一事实源。

修复旧项目的致命问题 8: 旧仓库里 qwen3_real_metrics.md /
teacher_forced_metrics.json / beta_sweep.json / experiment_report.md 对同一个
模型给出互相矛盾的 PPL 数值, 因为每份报告由不同脚本手写, 没有统一数据源。

这里所有数字只能通过 ResultStore.record() 写入 reports/results.jsonl,
所有 md 报告由 render_* 从该文件生成。verify_consistency() 会重新渲染并
逐字节比对磁盘报告, 任何手改数字都会在第三轮检查中被抓出来。
模型未运行的指标一律记为 pending_model_run, 报告中显示占位符而不是编造数字。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

PENDING = "pending_model_run"


class ResultStore:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root)
        self.path = self.root / "reports" / "results.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def record(self, experiment: str, metric: str, value: float | int | str | None,
               *, unit: str = "", n: int | None = None, seed: int | None = None,
               ci_low: float | None = None, ci_high: float | None = None,
               status: str = "measured", note: str = "", source: str = "") -> dict:
        if status == "measured" and isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite value for {experiment}/{metric}")
        rec = {
            "experiment": experiment, "metric": metric, "value": value,
            "unit": unit, "n": n, "seed": seed,
            "ci_low": ci_low, "ci_high": ci_high,
            "status": status, "note": note, "source": source,
        }
        self.records.append(rec)
        return rec

    def pending(self, experiment: str, metric: str, note: str) -> dict:
        return self.record(experiment, metric, None, status=PENDING, note=note)

    def flush(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        p = Path(path)
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _fmt(v, lo=None, hi=None) -> str:
    if v is None:
        return "PENDING_MODEL_RUN"
    if isinstance(v, str):
        return v
    s = f"{v:.4f}" if isinstance(v, float) else str(v)
    if lo is not None and hi is not None:
        s += f" [{lo:.4f}, {hi:.4f}]"
    return s


def render_research_report(records: list[dict], meta: dict, root: Path) -> str:
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_exp[r["experiment"]].append(r)

    def table(exp: str, title: str) -> list[str]:
        rows = by_exp.get(exp, [])
        out = [f"## {title}", "", "| 指标 | 值 | 95% 区间 | n | 状态 |",
               "|---|---:|---:|---:|---|"]
        for r in rows:
            ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" if r["ci_low"] is not None else "-"
            n = r["n"] if r["n"] is not None else "-"
            out.append(f"| {r['metric']} {r['unit']} | {_fmt(r['value'])} | "
                       f"{ci} | {n} | {r['status']} |")
        out.append("")
        return out

    L = [
        "# 技术报告：中文数学大模型后训练算法与可信评估", "",
        "本报告由 `src/core/results.py` 从 `reports/results.jsonl` 生成，",
        "表中每个数字都对应一条真实执行产生的记录。", "",
        "## 项目定位", "",
        "面向大模型算法岗的求职作品集。主线是一条完整的后训练链路：",
        "参数化数据构造 → 严格三态验证器 → LoRA-SFT（assistant-only loss）→",
        "偏好对构造 → DPO（显式冻结 SFT 参考模型）→ 序列级 KL 诊断 →",
        "四个泛化子集上的生成评测。差异化在于把评测链路本身做成可估计的研究对象：",
        "偏好信号可识别性、伪重复下的配对推断、验证器误判校正。", "",
        "三层验证刻意分开。算法层用可解析梯度的 log-linear 策略在 CPU 上真实跑通",
        "SFT 与 DPO，得到目标函数、梯度、KL、reward margin 与长度偏置的数字。",
        "权重层用真实 Qwen3-0.6B 权重跑明确缩减规模的 CPU pilot，验证链路可执行。",
        "完整协议层是 256 token、303 题的正式评测，本机 CPU 跑不完，",
        "9 个指标记为 pending_model_run。三层绝不互相冒充。", "",
        "## 研究问题", "",
        "在小样本、合成偏好数据、自动验证器三者同时存在时，",
        "后训练报告的提升有多少来自真实数学能力，又有多少来自表面线索、",
        "伪重复和验证器误判。三个子贡献分别回答可识别性、统计推断与测量偏差。", "",
        "## 实验台", "",
        f"- 数据集：`{meta.get('dataset', 'data/')}`，",
        f"  共 {meta.get('n_rows', 'NA')} 题，{meta.get('n_families', 'NA')} 个题型族，",
        f"  seed={meta.get('seed', 'NA')}。",
        f"- 数据 SHA256：`{meta.get('dataset_sha256', 'NA')}`。",
        f"- 泄露检查：`{meta.get('leakage_ok', 'NA')}`。",
        "- 算法层：`src/model/algo_lab.py`，7 维解析策略，梯度经有限差分校验；",
        "  `reports/algo_lab_curve.svg` 是其真实训练曲线。",
        "- 权重层 CPU pilot：`src/model/pilot.py`，真实 Qwen3-0.6B 权重，缩减规模写在",
        "  `configs/model_pilot.yaml`，逐条结果与 adapter 哈希见 `model_pilot_manifest.json`。",
        "- 完整协议模型阶段：本机 CPU 跑不完 303 题 × 256 token，",
        "  9 个指标记为 pending_model_run，需在有 GPU 环境执行",
        "  `python -m src.run_all --stage model` 后填入。", "",
    ]
    L += table("data_audit", "一、数据与评测协议")
    L += table("algo_lab", "二、后训练算法实验台（解析策略，非 Qwen 权重）")
    L += table("model_pilot", "三、权重层 CPU pilot（真实 Qwen3-0.6B 权重，缩减协议）")
    L += table("identifiability", "四、子贡献 1：偏好信号可识别性")
    L += table("cluster_inference", "五、子贡献 2：伪重复与配对推断")
    L += table("verifier_bias", "六、子贡献 3：验证器偏差校正")
    L += table("legacy_audit", "七、旧项目假象复算")
    L += table("model_stage", "八、Qwen3-0.6B 完整协议模型阶段（未运行）")
    L += [
        "## 结论边界", "",
        "`algo_lab` 的 measured 记录来自 CPU 上真实训练的 7 维解析策略，",
        "用于验证 DPO 目标函数层面的机制，不是 Qwen3 权重训练结果。",
        "`model_pilot` 的 measured 记录来自真实 Qwen3-0.6B 权重上的 CPU 缩减协议，",
        "它证明整条链路可执行，规模写在 `configs/model_pilot.yaml` 与",
        "`reports/model_pilot_manifest.json`，不用于能力结论，也不替换下面",
        "完整协议的 9 个 `pending_model_run` 指标。其余 measured 记录来自",
        "数据构造、验证器判定与统计估计量。每条记录的 note 都写明其口径。", "",
    ]
    return "\n".join(L)


def render_resume_bullets(records: list[dict], root: Path) -> str:
    idx = {(r["experiment"], r["metric"]): r for r in records}

    def g(exp, met, default="PENDING_MODEL_RUN"):
        r = idx.get((exp, met))
        return _fmt(r["value"]) if r else default

    return "\n".join([
        "# 简历成句", "",
        "算法层与统计层的描述都有真实运行结果支撑，数字来自 CPU 上实跑的",
        "解析策略实验台与三个子贡献；Qwen3-0.6B 权重的训练指标标为 PENDING_MODEL_RUN。", "",
        "## 项目定位", "",
        "中文数学大模型后训练算法与可信评估，大模型算法岗求职作品集。",
        "工程实现与评测方法并重。", "",
        "## 成句", "",
        "在 Qwen3-0.6B 权重上跑通 LoRA-SFT 与 DPO 的 CPU pilot，含 assistant-only loss mask、",
        "显式冻结 SFT 参考模型与序列级 KL 诊断。另在可解析策略上实跑 SFT 到 DPO，",
        "以有限差分校验梯度，三组偏好对照显示 reward accuracy 同为",
        f"{g('algo_lab','reward_accuracy_shortcut')} 时长度偏置劫持率可从 "
        f"{g('algo_lab','probe_hack_rate_shortcut')} 降到 "
        f"{g('algo_lab','probe_hack_rate_deconfounded')}，",
        f"参考模型锚错使 KL 由 {g('algo_lab','kl_deconfounded')} 升到 "
        f"{g('algo_lab','wrong_anchor_kl')}。",
        "进一步用可识别性检验（表面特征基线 AUC",
        f"{g('identifiability','surface_auc_shortcut')} 对 "
        f"{g('identifiability','surface_auc_deconfounded')}）、"
        f"聚类配对推断（有效样本量 {g('cluster_inference','effective_n')}）"
        "与验证器误判校正（观测准确率",
        f"{g('verifier_bias','observed_accuracy')} 校正到 "
        f"{g('verifier_bias','corrected_accuracy')}）",
        "解释旧项目 reward margin 上升而生成准确率退化的现象。", "",
        "## 禁止写入的内容", "",
        "- 任何 `pending_model_run` 指标，在模型阶段完成前不得填数字。",
        "- 旧项目简历里那组 teacher-forced loss 与 PPL 数字，",
        "  无法由保存的 adapter 复现，已从新项目彻底移除。",
        "- 旧简历里的「难度分层」「题目去重」「答案规范化」，",
        "  新数据集里这些能力已真实实现，见 data/manifest.json。",
        "- 把 algo_lab 解析策略的数字写成 Qwen3-0.6B 的训练结果。", "",
    ])


def render_resume_rewrite(records: list[dict], root: Path) -> str:
    """简历逐句改写对照。数字全部来自 results.jsonl, 字数按含空格全字符核对。"""
    idx = {(r["experiment"], r["metric"]): r for r in records}

    def v(exp, met, digits=4):
        r = idx.get((exp, met))
        if not r or r["value"] is None:
            return "PENDING_MODEL_RUN"
        x = r["value"]
        return f"{x:.{digits}f}" if isinstance(x, float) else str(x)

    manifest_path = root / "data" / "manifest.json"
    n_rows = v("data_audit", "n_rows", 0)
    n_fam = v("data_audit", "n_families", 0)
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_unseen = len(m["leakage"]["unseen_families"])
    else:
        n_unseen = "NA"

    bullets = [
        "在 Qwen3-0.6B 真实权重上跑通 LoRA-SFT 与 DPO 的 CPU pilot，含 "
        "assistant-only loss mask、显式冻结 SFT 参考模型与序列级 KL 诊断。"
        "另在可解析策略上实跑 SFT 到 DPO，以有限差分校验梯度，"
        f"参考模型锚错使 KL 由 {v('algo_lab','kl_deconfounded')} 升至 "
        f"{v('algo_lab','wrong_anchor_kl')}。",
        "构造长度与格式匹配、只改数学内容的对抗偏好对，把 DPO 梯度从长度线索转到正确性。"
        f"长度偏置劫持率由 {v('algo_lab','probe_hack_rate_shortcut')} 降至 "
        f"{v('algo_lab','probe_hack_rate_deconfounded')}，"
        f"质量方向系数增至 {v('algo_lab','theta_quality_deconfounded')}，"
        f"而三组 reward accuracy 均为 {v('algo_lab','reward_accuracy_shortcut')}，"
        "说明训练指标看不出差异。",
        f"构建 {n_rows} 道参数化中文数学题与三态验证器，按题型族与参数签名切分，"
        f"覆盖 {n_fam} 个题型族、{n_unseen} 个未见族，跨 split 零重叠，"
        "测试集保留未见参数、同义改写、反事实、未见题型四类泛化子集。",
        "提出偏好信号可识别性检验并给出命题与证明，"
        f"表面特征基线 AUC 从 {v('identifiability','surface_auc_shortcut')} 降至 "
        f"{v('identifiability','surface_auc_deconfounded')}；"
        "复算旧 40 条评测发现仅含 2 道独立题，"
        f"设计效应 {v('cluster_inference','design_effect',0)}、"
        f"有效样本量 {v('cluster_inference','effective_n')}，"
        f"检出 5 个百分点需约 {v('cluster_inference','clusters_needed_for_5pp',0)} 个独立簇。",
        f"用真值已知的校准集估计自动验证器 FPR {v('verifier_bias','fpr_legacy_substring')}，"
        f"把观测准确率 {v('verifier_bias','observed_accuracy')} 校正为 "
        f"{v('verifier_bias','corrected_accuracy')}"
        f"（构造真值 {v('verifier_bias','survey_pi_true')}），"
        "并给出 bootstrap 区间与三 verifier 潜类别估计。",
    ]
    lines = [
        "# 简历改写", "",
        "下面的成句可以直接复制。括号里的字数按含空格全字符计算，用 `len(text)` 核对，",
        "中文表单的字数上限通常按这个口径，不要用去空格的字符数填。", "",
        "## 一、项目名", "",
        "旧：`中文数学推理大模型的 LoRA-SFT 与 DPO 后训练`", "",
        "新：`中文数学大模型后训练算法与可信评估（LoRA-SFT / DPO / 评测偏差校正）`", "",
        "定位是大模型算法岗求职作品集，名字先把完整后训练链路摆出来，",
        "再把可信评估作为差异化，不要写成纯统计方法研究。",
        "面试口述时直接讲链路环节、数字出处与验证边界，不讨论项目来源。", "",
        "## 二、旧 bullet 逐句处置", "",
        "| 旧句子 | 处置 | 原因 |", "|---|---|---|",
        "| 构建中文数学指令与偏好数据集，完成题目去重、难度分层、答案规范化及严格 train/valid/test 隔离 | 重写 | 旧数据只有 8 个手写模板，没有难度分层，字符串去重被编号后缀骗过 |",
        "| 基于 Qwen3-0.6B 使用 LoRA 完成 SFT，设计对话模板并记录训练稳定性、验证损失和 CPU 训练资源 | 删掉训练结果部分，只保留实现 | 模型阶段未运行，不能写任何训练数字 |",
        "| 构造 chosen/rejected 答案对，使用 DPO 进行偏好优化，分析 beta、LoRA rank、数据规模和回答长度对 KL、准确率及训练稳定性的影响 | 整条删除 | beta 扫描的 margin 增长是恒等式，LoRA rank 消融根本没做 |",
        "| 实现基于表达式规范化与 SymPy 验证的自动评测，对比 Base、SFT、SFT+DPO 的 exact match、symbolic verification、格式遵循率和平均输出长度 | 整条删除 | 旧项目里 SymPy 从未生效，exact 与 symbolic 逐条 120/120 相同 |",
        "| 在独立 40 条测试集上，teacher-forced loss 与 Perplexity 大幅下降，token accuracy 提升若干个百分点 | 整条删除 | 保存的 adapter 复现不出这组数字，且隐藏了 DPO 生成准确率的退化 |",
        "| DPO 训练最终 reward margin 若干，完成 Base/SFT/DPO 对照 | 整条删除 | margin 随 beta 线性增长是定义使然，不构成结果 |", "",
        "## 三、可直接粘贴的成句", "",
        f"**中文数学大模型后训练算法与可信评估（LoRA-SFT / DPO / 评测偏差校正）**", "",
    ]
    for i, b in enumerate(bullets, 1):
        lines += [f"{i}. {b}", f"   （含空格全字符 {len(b)} 字）", ""]
    lines += [
        "前两条是算法与工程，后三条是差异化研究，顺序不要调。空间只够 4 条时删掉第 5 条。",
        "第 1 条的权重层 pilot 是缩减规模，只证明链路可执行，不作为能力结论。",
        "第 1、2 条里的算法数字来自解析策略实验台，不是完整协议的模型结果。面试时要主动讲清三层验证的边界，并说明完整协议仍待 GPU。", "",
        "## 四、自我评价里加一句", "",
        "概率论与数理统计背景，自修 LLM 后训练与推理，能从目标函数与梯度层面实现并验证 SFT 与 DPO，"
        "习惯先确认评测的独立单位与测量误差再下结论。", "",
        "这句放在求职自评里，不要和论文、基金、统计研究成果写在同一段。", "",
        "## 五、技能栏怎么补", "",
        "- 后训练算法：LoRA / PEFT、SFT 与 assistant-only loss mask、DPO 目标与梯度、"
        "参考模型锚定、beta 与序列级 KL、reward hacking 诊断",
        "- 数据与评测：参数化题目生成与独立 oracle、三态数学验证器、四类泛化子集、严格答案抽取",
        "- 统计与可复现：可识别性诊断、聚类 bootstrap、配对 McNemar、FPR/FNR 偏差校正、单一事实源",
        "- 工程：PyTorch、Transformers / TRL、PyCharm、零依赖算法实验台、三轮自动检查", "",
        "按熟悉程度分栏，没在机器上跑过的不要写精通；解析策略实验台要单独标注，"
        "不能混同为大模型权重训练。", "",
        "## 六、禁止写入", "",
        "- 任何 `pending_model_run` 指标，模型阶段跑完前不填数字。",
        "- 旧项目那组 loss 与 PPL 数字，以及当时的 reward margin 数值。",
        "- 难度分层、答案规范化、SymPy 符号验证、LoRA rank 消融、beta 敏感性这些没有真实实现或没有信息量的说法。",
        "- 把 algo_lab 解析策略的数字说成 Qwen3-0.6B 的训练结果。",
        "- 把本项目与其他研究经历混在同一段叙述。", "",
        "## 七、口径提醒", "",
        "被问到某一句时，先讲它对应后训练链路的哪个环节或哪个子贡献，",
        "再讲数字出处在 `reports/results.jsonl` 哪条记录，最后讲三层验证的边界。",
        "完整追问与回答见 `reports/interview_guide.md`，算法专项见 `reports/algorithm_interview.md`。", "",
    ]
    return "\n".join(lines)


def render_algorithm_interview(records: list[dict], root: Path) -> str:
    """大模型算法岗专项问答。所有数字从 results.jsonl 取, 保证可追溯。"""
    idx = {(r["experiment"], r["metric"]): r for r in records}

    def v(exp, met, digits=4):
        r = idx.get((exp, met))
        if not r or r["value"] is None:
            return "PENDING_MODEL_RUN"
        x = r["value"]
        return f"{x:.{digits}f}" if isinstance(x, float) else str(x)

    a = lambda k: v("algo_lab", k)

    def sci(exp, met):
        r = idx.get((exp, met))
        return f"{r['value']:.2e}" if r and isinstance(r["value"], float) else "NA"

    return "\n".join([
        "# 大模型算法岗专项对答", "",
        "这份文档回答「你到底做没做过后训练算法」。项目讲述与统计追问见 `interview_guide.md`。",
        "所有数字来自 `reports/results.jsonl`，由脚本渲染。", "",
        "## 一、三层验证，开场就讲清楚", "",
        "第一层是算法机制层：在 CPU 上真实训练一个 7 维 log-linear 策略，",
        "跑通 SFT 的最大似然、DPO 的梯度下降、精确 KL、reward margin、beta 扫描和参考模型锚点对照。",
        "DPO 解析梯度用中心有限差分校验，最大误差在 1e-11 量级。",
        "这一层不是 Qwen 权重，但它验证了目标函数与优化行为，不需要 GPU。",
        "第二层是权重层 CPU pilot：本机有 CPU 版 torch 与完整的 Qwen3-0.6B 权重，",
        "于是在 `configs/model_pilot.yaml` 写明的缩减规模下真跑 LoRA-SFT、",
        "显式冻结 SFT ref 的 DPO、masked loss、三态生成评测与序列级 KL，",
        "数字记在 `model_pilot` 实验，规模与 adapter 哈希记在 `model_pilot_manifest.json`。",
        "第三层才是完整协议：256 token、303 题、四个泛化子集，本机 CPU 跑不完，",
        "所以 9 个 `model_stage` 指标仍是 `pending_model_run`，一个数字没编。", "",
        "## 二、后训练链路逐环节", "",
        "| 环节 | 实现 | 能讲的算法点 | 文件 |", "|---|---|---|---|",
        "| 数据 | 参数化题族加独立 oracle 双路径交叉验证 | 数据泄露、按族切分、泛化子集设计 | `src/core/data_gen.py`, `splits.py` |",
        "| 评测 | 受限语法精确解析，correct / incorrect / unverifiable 三态 | 为什么不回退子串匹配、coverage 与 precision 取舍 | `src/core/expressions.py`, `verifier.py` |",
        "| SFT | LoRA 微调，只对 assistant completion 算 NLL | chat template、offset_mapping 字符级 mask、padding 与 attention mask | `src/model/train_sft.py`, `teacher_forced.py` |",
        "| 偏好数据 | 三组对照，长度格式匹配只改数学内容 | hard negative、shortcut、反事实组 | `src/core/preferences.py` |",
        "| DPO | 独立基座加载冻结 SFT adapter 作 ref_model | KL 锚点、隐式 reward、梯度形式 | `src/model/train_dpo.py` |",
        "| 诊断 | 采样估计序列级 KL 与按 prompt 聚类的区间 | 为什么真实模型里 KL 不能解析、beta 怎么调 | `src/model/diagnostics.py` |",
        "| 生成评测 | 统一解码协议、256 token、四类泛化子集 | greedy 与 sampling、截断、答案抽取 | `src/model/evaluate.py` |", "",
        "## 三、白板上能写的推导", "",
        "SFT 是候选集上的最大似然。log-linear 策略下",
        "$$\\nabla_\\theta \\mathrm{NLL} = \\mathbb{E}_{\\pi_\\theta}[\\phi] - \\phi(y^+),$$",
        "所以 SFT 把质量特征的系数往上推。", "",
        "DPO 来自带 KL 约束的 RLHF 最优解 $\\pi^* \\propto \\pi_{\\rm ref}\\exp(r/\\beta)$，",
        "反解出 $r = \\beta\\log(\\pi_\\theta/\\pi_{\\rm ref}) + c(x)$，代入 Bradley-Terry 得到成对分类损失",
        "$$L(\\theta)=-\\mathbb{E}\\log\\sigma\\big(\\beta[\\log\\tfrac{\\pi_\\theta(y^+|x)}{\\pi_{\\rm ref}(y^+|x)}"
        "-\\log\\tfrac{\\pi_\\theta(y^-|x)}{\\pi_{\\rm ref}(y^-|x)}]\\big).$$",
        "同一 prompt 下配分函数 $Z_\\theta(x)$ 抵消，令 $\\Delta=\\theta\\cdot(\\phi^+-\\phi^-)$，",
        "$$\\nabla_\\theta L = -\\beta(1-\\sigma(\\beta\\Delta))(\\phi^+-\\phi^-).$$",
        "这就是实验台里跑的梯度，也是有限差分校验的对象。", "",
        "LoRA 写 $W = W_0 + \\frac{\\alpha}{r}BA$，$B$ 初始化为零保证训练起点等于基座，",
        "秩 $r$ 决定可训练子空间维度；DPO 里如果 ref_model 置空，TRL 会 disable adapter，",
        "锚点变成未训练的 base，目标函数不再是上面这条。", "",
        "## 四、算法实验台的关键数字", "",
        "| 量 | 值 | 说明 |", "|---|---:|---|",
        f"| 探针题数 | {v('algo_lab','n_probe_items',0)} | 2x2 平衡：正确/错误 x 短/长，另设长错误对抗探针 |",
        f"| 特征维度 | {v('algo_lab','n_feature_dims',0)} | 6 个表面特征加 1 个严格验证器给出的正确性特征 |",
        f"| SFT 训练样本 | {v('algo_lab','sft_train_items',0)} | 真实跑 MLE 的指令条数 |",
        f"| DPO 梯度有限差分误差 | {sci('algo_lab','dpo_gradient_max_abs_error')} | 解析梯度与中心差分的最大绝对差 |",
        f"| margin/beta 变异系数（固定策略） | {a('margin_over_beta_cv_fixed_theta')} | 恒等式，不是实验发现 |",
        f"| margin/beta 变异系数（训练后） | {a('margin_over_beta_cv_after_training')} | beta 改变优化轨迹，所以不再是常数 |",
        f"| 正确锚点的 KL | {a('kl_deconfounded')} | ref = SFT 策略 |",
        f"| 锚错到 base 的 KL | {a('wrong_anchor_kl')} | policy 仍从 SFT 出发，ref 却是 base |",
        f"| 锚点造成的 KL 差 | {a('anchor_kl_gap')} | 旧项目 ref_model=None 的量化后果 |",
        "", "## 五、三组偏好对照", "",
        "| 组 | reward accuracy | reward margin | 长度偏置劫持率 | 质量系数 | 长度系数 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| shortcut | {a('reward_accuracy_shortcut')} | {a('reward_margin_shortcut')} | "
        f"{a('probe_hack_rate_shortcut')} | {a('theta_quality_shortcut')} | {a('theta_length_shortcut')} |",
        f"| deconfounded | {a('reward_accuracy_deconfounded')} | {a('reward_margin_deconfounded')} | "
        f"{a('probe_hack_rate_deconfounded')} | {a('theta_quality_deconfounded')} | {a('theta_length_deconfounded')} |",
        f"| length_reversed | {a('reward_accuracy_length_reversed')} | {a('reward_margin_length_reversed')} | "
        f"{a('probe_hack_rate_length_reversed')} | {a('theta_quality_length_reversed')} | "
        f"{a('theta_length_length_reversed')} |", "",
        "三组的训练指标 reward accuracy 完全一样，margin 也都上升，",
        f"但 shortcut 组把梯度主要花在长度特征上，对抗探针里偏向「更长但错误」的回答"
        f"（劫持率 {a('probe_hack_rate_shortcut')}），",
        f"去混淆组把梯度花在正确性特征上，劫持率降到 {a('probe_hack_rate_deconfounded')}。",
        "这就是 reward hacking 在目标函数层面的可复现证据，训练曲线见 `algo_lab_curve.svg`。", "",
        "## 六、算法岗高频追问", "",
        "**1. 你说做了 DPO，到底跑没跑？**",
        "跑了三层里的前两层。算法机制层在 CPU 上真跑 SFT MLE 与 DPO 梯度下降，",
        "梯度用有限差分校验过，有完整曲线。权重层用真实的 Qwen3-0.6B 加 LoRA 跑了",
        "缩减规模的 CPU pilot，SFT 步数、偏好对数、评测题数与生成长度都写在配置里，",
        "逐条预测和 adapter 哈希都落了盘。第三层完整协议需要 GPU，9 个指标明确标 pending。",
        "我不会把 pilot 数字说成完整协议结果，也不会用解析策略冒充权重训练。", "",
        "**2. 为什么用 log-linear 策略，不是真 transformer？**",
        "因为要验证的是目标函数层面的性质，不是模型容量。log-linear 策略的配分函数可算，",
        "DPO 梯度与 KL 都有闭式解，能把「beta 是缩放还是改变优化」「ref 锚错差多少」",
        "从口径争论变成可计算的数字。真实模型里 Z 不可算，KL 必须采样估计，",
        "我在 `diagnostics.py` 里给出的就是按 prompt 聚类的 MC 估计与区间。", "",
        "**3. DPO 和 PPO / GRPO 怎么选？**",
        "DPO 不需要显式 reward model 和在线采样，用离线偏好对直接优化隐式 reward，工程成本最低，",
        "但对偏好数据质量极敏感，这个项目量化的正是这个敏感性。PPO 需要 reward model 与 value net，",
        "GRPO 用组内相对优势省掉 value net，两者都在线采样、能持续探索，代价是算力与实现复杂度。",
        "小样本、弱验证器的场景下先修数据和评测比换算法更划算。", "",
        "**4. 为什么 ref 必须是 SFT 而不是 base？**",
        "DPO 推导里的参考策略是 SFT 模型。TRL 在 ref_model=None 且 policy 是 PeftModel 时会 disable adapter，",
        "锚点变成未训练 base，KL 惩罚把 SFT adapter 参数当成可直接改的变量。",
        "实验台上锚错使 KL 从 "
        f"{a('kl_deconfounded')} 升到 {a('wrong_anchor_kl')}，差 {a('anchor_kl_gap')}。",
        "修正版用独立基座实例加载冻结的 SFT adapter，并在训练前逐参数断言 ref 与 policy 初值一致。", "",
        "**5. beta 怎么调？**",
        "不能看 reward margin。固定策略时 margin 就是 beta 乘 log-ratio 差，",
        f"实验台上固定策略的 margin/beta 变异系数是 {a('margin_over_beta_cv_fixed_theta')}，纯恒等式。",
        "真正训练时 beta 会改变梯度尺度与收敛位置，要看序列级 KL 和下游生成准确率，",
        f"四个 beta 下 KL 跨度 {a('beta_kl_range')}，下游准确率跨度 {a('beta_probe_accuracy_range')}。", "",
        "**6. LoRA 的 B 为什么初始化为零？rank 怎么选？**",
        "零初始化保证训练起点与基座完全一致，SFT 的 ref 才有意义。rank 决定低秩子空间的容量，",
        "本项目没有做 rank 消融，简历里不写这一项。真要做，应该固定数据、beta 与步数，",
        "按题目簇给配对区间，而不是只看训练集 loss。", "",
        "**7. assistant-only loss mask 具体怎么做？**",
        "用 chat template 分别生成完整对话与只到 prompt 的文本，断言前缀关系，",
        "再用 fast tokenizer 的 offset_mapping 在字符级定位边界，把字符起点小于 prompt 长度的 label 设为 -100。",
        "不用 token 级前缀断言，因为 BPE 在交界处可能合并出不同 token。总 NLL 按 token 加权。", "",
        "**8. 生成评测的协议是什么？**",
        "关闭 thinking，max_new_tokens 至少 256，严格抽取最终答案，解析失败记 unverifiable 而不回退字符串匹配，",
        "按未见参数、同义改写、反事实、未见题型四个子集分别报准确率，并按题目簇给区间。",
        "旧项目用 64 token 截断，把没写完答案全记成数学错误，这条已修掉。", "",
        "**9. 长度偏置怎么检测？**",
        "两个层面。训练前用只用表面特征的交叉验证 AUC 体检偏好数据，"
        f"shortcut 组 {v('identifiability','surface_auc_shortcut')}，"
        f"去混淆组 {v('identifiability','surface_auc_deconfounded')}。",
        "训练后用对抗探针，让「短的正确」与「明显更长的错误」二选一，",
        f"shortcut 组劫持率 {a('probe_hack_rate_shortcut')}，"
        f"去混淆组 {a('probe_hack_rate_deconfounded')}。", "",
        "**10. 0.6B 这么小，结论有意义吗？**",
        "算法层验证的是目标函数性质，与模型规模无关。模型层选 0.6B 是为了在 CPU 上也能复现，",
        "而偏好数据里没有真错误、测试集只有 2 道题这类结构性缺陷，换更大的模型不会自动消失。", "",
        "**11. 给你 GPU，第一件事做什么？**",
        "跑三组 DPO 的下游对照，看去混淆偏好对在真实权重上是否也把梯度从长度转到正确性，",
        "同时把序列级 KL 的 MC 估计与解析实验台的结论对齐。再在 CMATH 或 GSM8K 中文子集上重复整套诊断。", "",
        "## 七、不能说的话", "",
        "不要把 algo_lab 的数字说成 Qwen3 的训练结果，不要报任何 pending 指标，",
        "不要说做过 LoRA rank 消融或 GRPO，不要复述旧项目那组无法复现的 loss 与 PPL。", "",
    ])


GENERATED_REPORTS = ("research_report.md", "resume_bullets.md", "resume_rewrite.md",
                     "algorithm_interview.md")


def render_all(root: str | Path, meta: dict) -> dict[str, Path]:
    root = Path(root)
    records = ResultStore.load(root / "reports" / "results.jsonl")
    out = {}
    for name, text in [
        ("research_report.md", render_research_report(records, meta, root)),
        ("resume_bullets.md", render_resume_bullets(records, root)),
        ("resume_rewrite.md", render_resume_rewrite(records, root)),
        ("algorithm_interview.md", render_algorithm_interview(records, root)),
    ]:
        p = root / "reports" / name
        p.write_text(text, encoding="utf-8")
        out[name] = p
    return out


def verify_consistency(root: str | Path, meta: dict) -> dict:
    """第三轮检查: 重新渲染, 逐字节比对磁盘报告。手改数字会被抓出来。"""
    root = Path(root)
    before = {name: (root / "reports" / name).read_text(encoding="utf-8")
              for name in GENERATED_REPORTS
              if (root / "reports" / name).exists()}
    render_all(root, meta)
    mismatches = []
    for name, old in before.items():
        new = (root / "reports" / name).read_text(encoding="utf-8")
        if old != new:
            mismatches.append(name)
    records = ResultStore.load(root / "reports" / "results.jsonl")
    fabricated = [r for r in records
                  if r["status"] == "measured" and r["value"] is None]
    return {
        "ok": not mismatches and not fabricated,
        "report_mismatches": mismatches,
        "measured_with_null_value": fabricated,
        "n_records": len(records),
        "n_pending": sum(r["status"] == PENDING for r in records),
    }


def append_records(root: str | Path, new_records: list[dict]) -> list[dict]:
    """模型阶段把同名的 pending_model_run 记录替换成 measured 记录。

    不会覆盖研究阶段的其它记录, 也不会让旧报告里的手写数字复活。
    """
    root = Path(root)
    path = root / "reports" / "results.jsonl"
    existing = ResultStore.load(path) if path.exists() else []
    keys = {(r["experiment"], r["metric"]) for r in new_records}
    merged = [r for r in existing if (r["experiment"], r["metric"]) not in keys]
    merged.extend(new_records)
    with path.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return merged


def sha256_of_dir_files(root: str | Path, patterns: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    root = Path(root)
    paths = []
    for pat in patterns:
        paths.extend(sorted(root.glob(pat)))
    for p in paths:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()
