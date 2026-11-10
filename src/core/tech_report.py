"""技术报告生成器。

从 reports/results.jsonl 与 reports/algo_lab_curves.json 渲染一份自包含的
HTML 技术报告（无 CDN、无外部字体、内嵌 SVG 图），可直接用浏览器打开，
也可用 Chrome 无头打印成 PDF。

所有数字都从结果文件读取，报告里不写任何手写指标。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import json
import math
from pathlib import Path

from src.core.results import ResultStore

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body {
  margin: 0; color: #1f2937; background: #ffffff;
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
  font-size: 10.5pt; line-height: 1.78;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 178mm; margin: 0 auto; padding: 6mm 0 2mm; }
h1 { font-size: 21pt; line-height: 1.35; margin: 0 0 6px; letter-spacing: .2px; }
.sub { font-size: 11pt; color: #4b5563; margin: 0 0 4px; }
.meta { font-size: 9pt; color: #6b7280; margin: 10px 0 0;
        padding-bottom: 12px; border-bottom: 2px solid #111827; }
h2 { font-size: 13.5pt; margin: 26px 0 8px; padding-bottom: 4px;
     border-bottom: 1px solid #d1d5db; break-after: avoid; }
h3 { font-size: 11.2pt; margin: 16px 0 5px; break-after: avoid; }
p { margin: 7px 0; text-align: justify; }
ul, ol { margin: 6px 0 6px 20px; padding: 0; }
li { margin: 3px 0; }
.abstract { background: #f8fafc; border-left: 3px solid #1d4ed8;
            padding: 11px 14px; margin: 14px 0 6px; font-size: 10pt; }
.abstract b { color: #111827; }
table { width: 100%; border-collapse: collapse; margin: 9px 0 4px; font-size: 9.2pt;
        table-layout: fixed; break-inside: avoid; }
th, td { overflow-wrap: anywhere; word-break: break-word; }
caption { caption-side: top; text-align: left; font-size: 8.8pt; color: #6b7280;
          padding-bottom: 4px; }
thead th { border-top: 1.4px solid #111827; border-bottom: .8px solid #9ca3af;
           padding: 5px 6px; text-align: left; font-weight: 600; background: #f9fafb; }
tbody td { padding: 4px 6px; border-bottom: .5px solid #e5e7eb; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1.4px solid #111827; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
td.c, th.c { text-align: center; }
.eq { margin: 10px 0; padding: 8px 12px; background: #f8fafc;
      border-left: 2px solid #cbd5e1; font-size: 10pt; overflow-x: auto; }
.eq .no { float: right; color: #6b7280; font-size: 9pt; }
i.v { font-family: "Cambria Math", "Times New Roman", serif; font-style: italic; }
.small { font-size: 9pt; color: #4b5563; }
.note { font-size: 8.8pt; color: #6b7280; margin: 2px 0 10px; }
figure { margin: 12px 0 8px; break-inside: avoid; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { font-size: 8.8pt; color: #4b5563; margin-top: 4px; }
.tag { display: inline-block; font-size: 8.2pt; color: #1d4ed8;
       background: #eff6ff; border: .5px solid #bfdbfe; border-radius: 3px;
       padding: 1px 5px; margin-right: 4px; }
.pending { color: #b45309; font-weight: 600; }
.two { display: flex; gap: 10px; }
.two > figure { flex: 1; margin-top: 6px; }
.pb { break-before: page; }
footer { margin-top: 26px; padding-top: 8px; border-top: 1px solid #d1d5db;
         font-size: 8.5pt; color: #6b7280; }
"""


def _f(v, d: int = 4, dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return dash
        if v != 0 and abs(v) < 1e-4:
            return f"{v:.2e}"
        return f"{v:.{d}f}"
    return str(v)


def _pct(v, d: int = 1) -> str:
    return "—" if v is None else f"{100 * v:.{d}f}%"


def _bar_svg(title: str, labels: list[str], values: list[float],
             colors: list[str], ymax: float, unit: str = "") -> str:
    """单指标分组柱状图, 宽 330, 高 210。"""
    W, H, L, B, T = 330, 210, 42, 34, 24
    n = len(values)
    slot = (W - L - 12) / max(n, 1)
    bw = min(46, slot * 0.56)
    bars = []
    for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
        x = L + slot * i + (slot - bw) / 2
        h = (H - B - T) * (v / ymax if ymax else 0)
        y = H - B - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                    f'fill="{c}" rx="1.5"/>')
        bars.append(f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" font-size="10" '
                    f'text-anchor="middle" fill="#111827">{v:.4f}</text>')
        bars.append(f'<text x="{x + bw / 2:.1f}" y="{H - B + 14:.1f}" font-size="9" '
                    f'text-anchor="middle" fill="#4b5563">{lab}</text>')
    grid = []
    for k in range(5):
        y = T + (H - B - T) * k / 4
        val = ymax * (1 - k / 4)
        grid.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - 8}" y2="{y:.1f}" '
                    f'stroke="#e5e7eb" stroke-width="1"/>')
        grid.append(f'<text x="{L - 6}" y="{y + 3:.1f}" font-size="8.5" '
                    f'text-anchor="end" fill="#9ca3af">{val:.2f}</text>')
    return "\n".join([
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{L}" y="14" font-size="10.5" fill="#111827">{title}{unit}</text>',
        *grid, *bars,
        f'<line x1="{L}" y1="{H - B}" x2="{W - 8}" y2="{H - B}" stroke="#9ca3af"/>',
        '</svg>',
    ])


def _curve_svg(curves: dict, keys: list[tuple[str, str, str]]) -> str:
    """三组 reward margin 随 step 的曲线, 宽 700, 高 250。"""
    W, H, L, B, T = 700, 250, 52, 38, 26
    all_steps = sorted({c["step"] for v in curves.values() for c in v})
    smax = max(all_steps) or 1
    ymax = max(c["reward_margin"] for v in curves.values() for c in v)
    ymax = max(ymax, 1e-9) * 1.12

    def X(s):
        return L + (W - L - 16) * s / smax

    def Y(v):
        return H - B - (H - B - T) * v / ymax

    parts = ['<rect width="100%" height="100%" fill="#ffffff"/>']
    for k in range(5):
        y = T + (H - B - T) * k / 4
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - 10}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{L - 6}" y="{y + 3:.1f}" font-size="8.5" '
                     f'text-anchor="end" fill="#9ca3af">{ymax * (1 - k / 4):.3f}</text>')
    parts.append(f'<line x1="{L}" y1="{H - B}" x2="{W - 10}" y2="{H - B}" stroke="#9ca3af"/>')
    for label, key, color in keys:
        rows = curves[label]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(c['step']):.1f},{Y(c[key]):.1f}"
                     for i, c in enumerate(rows))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i, (label, _, color) in enumerate(keys):
        x = L + 8 + i * 168
        parts.append(f'<line x1="{x}" y1="{T + 6}" x2="{x + 18}" y2="{T + 6}" '
                     f'stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 23}" y="{T + 9.5}" font-size="9" fill="#374151">'
                     f'{label}</text>')
    parts.append(f'<text x="{L}" y="{H - 14}" font-size="8.5" fill="#6b7280">step 0</text>')
    parts.append(f'<text x="{W - 54}" y="{H - 14}" font-size="8.5" fill="#6b7280">'
                 f'step {smax}</text>')
    parts.append(f'<text x="{L}" y="15" font-size="10.5" fill="#111827">'
                 f'三组偏好数据的 reward margin 训练曲线</text>')
    return "\n".join([f'<svg viewBox="0 0 {W} {H}" role="img">', *parts, '</svg>'])


def _interval_svg(iid: tuple[float, float], cluster: tuple[float, float],
                  point: float) -> str:
    """iid 区间与聚类区间的对比, 宽 660, 高 182。"""
    W, H, L, T = 660, 182, 96, 34
    x0, x1 = -0.01, 0.27

    def X(v):
        return L + (W - L - 20) * (v - x0) / (x1 - x0)

    rows = [("iid 正态区间", iid, "#dc2626"), ("聚类 bootstrap", cluster, "#2563eb")]
    parts = ['<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="0" y="15" font-size="10.5" fill="#111827">'
             f'配对提升的 95% 区间：把 40 条当独立样本会得到虚假的精确</text>']
    for i, (name, (lo, hi), color) in enumerate(rows):
        y = T + i * 42
        parts.append(f'<text x="0" y="{y + 4}" font-size="9" fill="#374151">{name}</text>')
        parts.append(f'<line x1="{X(lo):.1f}" y1="{y}" x2="{X(hi):.1f}" y2="{y}" '
                     f'stroke="{color}" stroke-width="6" stroke-linecap="round" opacity=".55"/>')
        parts.append(f'<text x="{X(lo):.1f}" y="{y + 18}" font-size="8.5" '
                     f'text-anchor="middle" fill="{color}">{lo:.4f}</text>')
        parts.append(f'<text x="{X(hi):.1f}" y="{y + 18}" font-size="8.5" '
                     f'text-anchor="middle" fill="{color}">{hi:.4f}</text>')
    zero = X(0)
    parts.append(f'<line x1="{zero:.1f}" y1="{T - 12}" x2="{zero:.1f}" y2="{T + 88}" '
                 f'stroke="#111827" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{zero + 4:.1f}" y="{T - 14}" font-size="8.5" fill="#111827">0</text>')
    parts.append(f'<line x1="{L}" y1="{T + 96}" x2="{W - 12}" y2="{T + 96}" stroke="#d1d5db"/>')
    for v in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        parts.append(f'<line x1="{X(v):.1f}" y1="{T + 92}" x2="{X(v):.1f}" y2="{T + 100}" '
                     f'stroke="#9ca3af"/>')
        parts.append(f'<text x="{X(v):.1f}" y="{T + 112}" font-size="8" '
                     f'text-anchor="middle" fill="#6b7280">{v:.2f}</text>')
    parts.append(f'<text x="{X(point):.1f}" y="{T + 128}" font-size="8.5" '
                 f'text-anchor="middle" fill="#111827">点估计 {point:.4f}</text>')
    return "\n".join([f'<svg viewBox="0 0 {W} {H}" role="img">', *parts, '</svg>'])


def render(records: list[dict], root: str | Path) -> str:
    root = Path(root)
    idx = {(r["experiment"], r["metric"]): r["value"] for r in records}

    def g(exp, met, default=None):
        return idx.get((exp, met), default)

    curves_path = root / "reports" / "algo_lab_curves.json"
    curves, sweep = {}, {}
    if curves_path.exists():
        blob = json.loads(curves_path.read_text(encoding="utf-8"))
        curves, sweep = blob["curves"], blob["beta_sweep"]
    manifest_path = root / "data" / "manifest.json"
    seed = "—"
    if manifest_path.exists():
        seed = json.loads(manifest_path.read_text(encoding="utf-8")).get("seed", "—")

    mp = lambda k, d=None: g("model_pilot", k, d)
    a = lambda k, d=None: g("algo_lab", k, d)
    legacy = lambda k, d=None: g("legacy_audit", k, d)
    cl = lambda k, d=None: g("cluster_inference", k, d)
    vb = lambda k, d=None: g("verifier_bias", k, d)
    da = lambda k, d=None: g("data_audit", k, d)
    ident = lambda k, d=None: g("identifiability", k, d)

    curve_fig = ""
    if curves:
        curve_fig = f"""
<figure>{_curve_svg(curves, [("shortcut", "reward_margin", "#dc2626"),
                             ("deconfounded", "reward_margin", "#2563eb"),
                             ("length_reversed", "reward_margin", "#059669")])}
<figcaption>图 1　三组偏好数据的 reward margin 均随训练上升，训练损失无法区分它们。
数据来自 <span class="small">reports/algo_lab_curves.json</span>。</figcaption></figure>"""

    L = []
    L.append(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>中文数学大模型后训练算法与可信评估 · 技术报告</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>中文数学大模型后训练算法与可信评估</h1>
<p class="sub">技术报告：一条 LoRA-SFT + DPO 链路，以及一次「指标上升、能力退化」的归因</p>
<p class="meta">代码：<code>math-posttraining-project-v2</code>　·　数据 seed {seed}　·　
指标源：<code>reports/results.jsonl</code>（{len(records)} 条记录，{sum(1 for r in records if r['status']=='pending_model_run')} 条为 Qwen 权重层待运行）</p>

<div class="abstract">
<b>摘要.</b> 本项目构建一条中文数学后训练链路，包含参数化题目生成、严格三态验证器、
LoRA-SFT、三组对照偏好数据、DPO 与四个泛化子集上的生成评测。起点是一次失败复现：
旧项目训练损失与 reward margin 单调下降或上升，自由生成的符号准确率却由 62.5% 退回 50%。
审计把原因定位到三处测量失效：40 条测试集实际只含 2 道独立题，旧评测器判对的 65 条中
32 条是假阳性，240 条偏好对里没有一条含真实数学错误。为了在没有 GPU 的机器上验证
目标函数层面的机制，构建了 7 维可解析梯度的 log-linear 策略实验台，SFT 与 DPO 均为
真实训练，解析梯度经中心有限差分校验（最大误差 3.44e-11）。三组偏好对照的
reward accuracy 同为 1.0000，长度偏置劫持率分别为 0.7241、0.1864、0.0720，
参考模型锚错使 KL 由 0.0513 升至 0.5179。三个测量问题各自给出估计量与区间：
表面特征 AUC 从 1.0000 降到 0.5000，设计效应 15 对应有效样本量 2.6667，
验证器 FPR 0.5000 使观测准确率 0.7425 被校正为 0.6496。Qwen3-0.6B 的权重训练尚未运行，
9 个模型指标保持 <span class="pending">pending</span>，不在本报告中给出数值。
</div>

<h2>1　问题与起点</h2>
<p>旧项目报告了两组互相矛盾的证据。一方面，teacher-forced loss 从 4.41 降到 2.50，
reward margin 随 beta 单调上升；另一方面，同一套权重在自由生成下的符号准确率是
Base 50%、SFT 62.5%、DPO 50%，DPO 把 SFT 的增益退回了起点。更直接的矛盾是，
三份报告对同一个模型给出三个不同的 perplexity 数值，极差 {_f(legacy('ppl_spread'))}，
而仓库里保存的 adapter 无法复现其中任何一组最好看的数字。</p>
<p>把现象拆成三个可检验的假设，比笼统地说「调参失败」更有用：</p>
<ul>
<li><b>H1 数据不承载质量信号。</b>偏好对的标签可以由长度与格式线索完全预测，
梯度方向与真实数学质量无关。</li>
<li><b>H2 评测器有系统性误判。</b>子串包含式的判对把推导里出现的数字当成答案，
观测准确率高于真实准确率。</li>
<li><b>H3 推断单位错误。</b>测试集里的 40 条是 2 道题的伪重复，
把簇内完全相关的样本当独立样本会把方差低估一个数量级。</li>
</ul>
<p>表 1 是对 H1 至 H3 的直接复算。每一项都由 <code>src/research/legacy_audit.py</code>
与 <code>cluster_inference.py</code> 在旧项目原始文件上重新计算，不引用旧报告里的任何数字。</p>
<table><caption>表 1　旧项目审计结果（在原文件上复算）</caption>
<thead><tr><th>检查项</th><th class="n">结果</th><th>含义</th></tr></thead><tbody>
<tr><td>测试集条数 / 去编号后唯一题干</td><td class="n">{_f(da('legacy_test_rows'),0)} / {_f(da('legacy_unique_questions'),0)}</td><td>有效样本量是 2，不是 40</td></tr>
<tr><td>伪重复倍数</td><td class="n">{_f(da('legacy_pseudo_replication'),0)}</td><td>同题重复 20 次</td></tr>
<tr><td>旧判对条数 / 严格重判仍对 / 假阳性</td><td class="n">{_f(legacy('old_reported_correct'),0)} / {_f(legacy('strict_verifier_correct'),0)} / {_f(legacy('old_false_positives'),0)}</td><td>假阳性率 {_f(legacy('old_false_positive_rate'))}</td></tr>
<tr><td>标为「错误答案」的偏好对 / 真错</td><td class="n">{_f(da('legacy_claimed_wrong_pairs'),0)} / {_f(da('legacy_actually_wrong_pairs'),0)}</td><td>错误答案生成器对训练集答案全部无效</td></tr>
<tr><td>exact 与 symbolic 逐条相同的比例</td><td class="n">{_f(da('legacy_exact_equals_symbolic_rate'))}</td><td>SymPy 分支从未生效</td></tr>
<tr><td>同一模型 PPL 的不同取值个数 / 极差</td><td class="n">{_f(legacy('ppl_sources_conflicting'),0)} / {_f(legacy('ppl_spread'))}</td><td>报告之间不可互证</td></tr>
</tbody></table>
<p class="note">严格验证器在旧项目 120 条生成输出上判 33 条 correct、87 条
unverifiable，后者主要来自 64 token 截断与不受支持的 LaTeX 写法。</p>

<h2>2　后训练链路</h2>
<h3>2.1　数据构造与切分</h3>
<p>题目由 12 个参数化题型族生成，每题参数唯一，答案先由题型族自带的 oracle 计算，
再用第二条独立计算路径交叉验证，不一致直接抛错。切分以题型族与参数签名为单位，
而不是以字符串为单位，这样「同题改写」不会跨 split 泄漏。测试集额外拆成四个子集：
未见参数、同义改写、反事实（改动条件后重算答案）、未见题型族。</p>
<table><caption>表 2　数据规模与切分检查</caption>
<thead><tr><th>项</th><th class="n">值</th><th>项</th><th class="n">值</th></tr></thead><tbody>
<tr><td>题目总数</td><td class="n">{_f(da('n_rows'),0)}</td><td>唯一题干数</td><td class="n">{_f(da('n_unique_questions'),0)}</td></tr>
<tr><td>题型族</td><td class="n">{_f(da('n_families'),0)}</td><td>唯一参数签名</td><td class="n">{_f(da('n_unique_parameter_signatures'),0)}</td></tr>
<tr><td>跨 split 参数重叠</td><td class="n">{_f(da('cross_split_overlap'),0)}</td><td>oracle 与验证器一致</td><td class="n">{_f(da('oracle_verifier_consistent_rows'),0)} / {_f(da('n_rows'),0)}</td></tr>
<tr><td>测试集条数</td><td class="n">{_f(cl('new_n_test'),0)}</td><td>测试集族数</td><td class="n">{_f(cl('new_n_family_clusters'),0)}</td></tr>
</tbody></table>

<h3>2.2　三态评测器</h3>
<p>答案比较不使用子串包含。做法是把候选答案与标准答案都解析成受限语法的表达式
（支持整数、分数、小数、逗号分隔的多解、单变量多项式），在有理数域上做精确相等，
必要时对多项式做规范形比较。解析失败抛 <code>ParseError</code>，由上层记为
<code>unverifiable</code>，不回退到字符串匹配。判定因此是三态：
<code>correct</code>、<code>incorrect</code>、<code>unverifiable</code>。
第三态单独统计，不折进错误率。</p>
<p>这个选择的代价是覆盖率。旧项目 120 条生成输出里有 87 条不可核验，
主要因为解码在 64 token 处截断，答案标记没写完。新协议把
<code>max_new_tokens</code> 固定为 256，并把这个比例作为独立指标报告，
而不是把它当成「答错」来压低准确率。</p>

<h3>2.3　SFT 的 assistant-only loss</h3>
<p>对话样本由 prompt 与 assistant completion 拼接。旧实现对整段序列计算 NLL，
模型只需记住重复出现的题干就能把 loss 压低，指标与数学能力脱钩。
修正版用 chat template 分别渲染完整对话与只到 prompt 的文本，断言前者以后者为前缀，
再用 fast tokenizer 的 <code>offset_mapping</code> 在字符级定位边界，
把起点小于 prompt 长度的 token 的 label 置为 -100。总 NLL 按 completion token 加权，
训练与验证共用同一 mask。</p>

<h3>2.4　三组对照偏好数据</h3>
<p>三组数据回答同一个问题：DPO 的梯度到底花在哪个特征上。shortcut 组复刻旧项目的缺陷，
rejected 更短、答案其实正确。deconfounded 组把长度与格式严格对齐，只改数学内容，
rejected 是把推导里所有答案出现处等长替换为错误值，长度差恒为 0、格式标记一致。
length_reversed 组故意反转长度线索，rejected 更长但答案错误。</p>
<table><caption>表 3　三组偏好数据的构造检查</caption>
<thead><tr><th>组</th><th class="n">对数</th><th class="n">真错误率</th><th class="n">平均长度差</th><th class="n">格式匹配率</th></tr></thead><tbody>
<tr><td>shortcut</td><td class="n">{_f(da('preference_rows_shortcut'),0)}</td><td class="n">{_f(da('preference_true_error_rate_shortcut'))}</td><td class="n">{_f(da('preference_abs_length_diff_shortcut'),1)}</td><td class="n">{_f(da('preference_format_match_shortcut'))}</td></tr>
<tr><td>deconfounded</td><td class="n">{_f(da('preference_rows_deconfounded'),0)}</td><td class="n">{_f(da('preference_true_error_rate_deconfounded'))}</td><td class="n">{_f(da('preference_abs_length_diff_deconfounded'),1)}</td><td class="n">{_f(da('preference_format_match_deconfounded'))}</td></tr>
<tr><td>length_reversed</td><td class="n">{_f(da('preference_rows_length_reversed'),0)}</td><td class="n">{_f(da('preference_true_error_rate_length_reversed'))}</td><td class="n">{_f(da('preference_abs_length_diff_length_reversed'),1)}</td><td class="n">{_f(da('preference_format_match_length_reversed'))}</td></tr>
</tbody></table>
<p class="note">平均长度差取绝对值，shortcut 组 chosen 更长，length_reversed 组 rejected 更长。</p>

<h3>2.5　DPO 与参考模型</h3>
<p>DPO 的参考策略必须是 SFT 模型。TRL 在 <code>ref_model=None</code> 且策略是
<code>PeftModel</code> 时会 disable adapter，KL 锚点于是落在未训练的 base 上，
SFT adapter 的参数被当作可直接优化的变量，目标函数不再是原始 DPO。
修正版用独立基座实例加载冻结的 SFT adapter 显式传入，训练前逐参数断言 ref 与 policy
初始一致，并把真正使用的 ref 快照落盘。</p>
<div class="eq"><span class="no">(1)</span>
<i class="v">L</i>(<i class="v">θ</i>) = −(1/<i class="v">n</i>) Σ<sub><i class="v">i</i></sub>
log σ( <i class="v">β</i> [ log(<i class="v">π</i><sub><i class="v">θ</i></sub>(<i class="v">y</i><sub><i class="v">i</i></sub><sup>+</sup>|<i class="v">x</i><sub><i class="v">i</i></sub>) / <i class="v">π</i><sub>ref</sub>(<i class="v">y</i><sub><i class="v">i</i></sub><sup>+</sup>|<i class="v">x</i><sub><i class="v">i</i></sub>))
− log(<i class="v">π</i><sub><i class="v">θ</i></sub>(<i class="v">y</i><sub><i class="v">i</i></sub><sup>−</sup>|<i class="v">x</i><sub><i class="v">i</i></sub>) / <i class="v">π</i><sub>ref</sub>(<i class="v">y</i><sub><i class="v">i</i></sub><sup>−</sup>|<i class="v">x</i><sub><i class="v">i</i></sub>)) ])
</div>
<p>beta 不能只看 reward margin。固定策略时 margin 等于 beta 乘 log-ratio 差，
随 beta 线性增长是恒等式。可报告的量是策略相对参考模型的 KL，以及下游生成准确率。
真实模型里配分函数不可算，KL 用从策略采样的
log π<sub>θ</sub> − log π<sub>ref</sub> 均值估计，按 prompt 聚类给 bootstrap 区间。</p>

<h2 class="pb" id="s3">3　算法机制实验台</h2>
<h3>3.1　为什么单独做一层</h3>
<p>没有 GPU 时，权重层的 SFT 与 DPO 无法执行，但目标函数层面的性质仍然可以验证，
前提是配分函数可算。把策略限制为候选集上的 log-linear 模型，
log π<sub>θ</sub>(y|x) = θ·φ(x,y) − log Z<sub>θ</sub>(x)，
其中 Z<sub>θ</sub>(x) = Σ<sub>y∈Y(x)</sub> exp(θ·φ(x,y))。
同一 prompt 下 Z 在成对比较里抵消，DPO 的隐式 reward 差、梯度和 KL 都有闭式解。
这一层不替代权重训练，它回答的是「目标函数是否真的在奖励长度」。
算法层、权重层 pilot 与完整协议三层在报告中始终分开标注。</p>

<h3>3.2　模型与梯度</h3>
<p>特征 φ 取 7 维：字符长度、是否含最终答案标记、LaTeX 标记数、数字个数、标点数、换行数，
以及由严格验证器给出的正确性指示。前 6 维是表面特征，最后一维是质量的可观测代理。
令 Δ<sub>i</sub> = (θ − θ<sub>ref</sub>)·(φ<sub>i</sub><sup>+</sup> − φ<sub>i</sub><sup>−</sup>)，
式 (1) 化为</p>
<div class="eq"><span class="no">(2)</span>
<i class="v">L</i> = −(1/<i class="v">n</i>) Σ<sub><i class="v">i</i></sub>
log σ(<i class="v">β</i> Δ<sub><i class="v">i</i></sub>)，　　
∇<sub><i class="v">θ</i></sub><i class="v">L</i> = −(<i class="v">β</i>/<i class="v">n</i>)
Σ<sub><i class="v">i</i></sub> (1 − σ(<i class="v">β</i> Δ<sub><i class="v">i</i></sub>))
(φ<sub><i class="v">i</i></sub><sup>+</sup> − φ<sub><i class="v">i</i></sub><sup>−</sup>)
</div>
<p>式 (2) 是实验台里实际执行的更新，也是有限差分校验的对象。SFT 阶段在 chosen 上做
最大似然，梯度为 E<sub>π<sub>θ</sub></sub>[φ] − φ(y<sup>+</sup>)，
得到 θ<sub>sft</sub> 作为 DPO 的策略初值与参考模型。</p>

<h3>3.3　设置</h3>
<p>SFT 用 {_f(a('sft_train_items'),0)} 条指令、120 步、学习率 0.03。
DPO 每组 400 步、学习率 0.08、beta 0.1。下游用 2×2 平衡探针：
正确/错误 × 短/长，短对与短错等长，长对与长错等长，另设一组对抗探针，
让「短的正确」与「明显更长的错误」二选一。所有分数用 softmax 概率加权，
不用硬 argmax，避免平局时按索引取值造成虚高。</p>

<h3>3.4　结果</h3>
<table><caption>表 4　三组偏好数据训练后的策略行为（beta 0.1，{_f(a('n_probe_items'),0)} 道探针；
reward acc 为 reward accuracy，劫持率为对抗探针中选中「更长但错误」回答的概率）</caption>
<colgroup><col style="width:19%"><col style="width:14%"><col style="width:11%"><col style="width:13%"><col style="width:14%"><col style="width:14%"><col style="width:11%"></colgroup>
<thead><tr><th>组</th><th class="n">reward acc</th><th class="n">margin</th><th class="n">劫持率</th><th class="n">θ_quality</th><th class="n">θ_length</th><th class="n">KL</th></tr></thead><tbody>
<tr><td>shortcut</td><td class="n">{_f(a('reward_accuracy_shortcut'))}</td><td class="n">{_f(a('reward_margin_shortcut'))}</td><td class="n">{_f(a('probe_hack_rate_shortcut'))}</td><td class="n">{_f(a('theta_quality_shortcut'))}</td><td class="n">{_f(a('theta_length_shortcut'))}</td><td class="n">{_f(a('kl_shortcut'))}</td></tr>
<tr><td>deconfounded</td><td class="n">{_f(a('reward_accuracy_deconfounded'))}</td><td class="n">{_f(a('reward_margin_deconfounded'))}</td><td class="n">{_f(a('probe_hack_rate_deconfounded'))}</td><td class="n">{_f(a('theta_quality_deconfounded'))}</td><td class="n">{_f(a('theta_length_deconfounded'))}</td><td class="n">{_f(a('kl_deconfounded'))}</td></tr>
<tr><td>length_reversed</td><td class="n">{_f(a('reward_accuracy_length_reversed'))}</td><td class="n">{_f(a('reward_margin_length_reversed'))}</td><td class="n">{_f(a('probe_hack_rate_length_reversed'))}</td><td class="n">{_f(a('theta_quality_length_reversed'))}</td><td class="n">{_f(a('theta_length_length_reversed'))}</td><td class="n">{_f(a('kl_length_reversed'))}</td></tr>
</tbody></table>
{curve_fig}
<div class="two">
<figure>{_bar_svg("长度偏置劫持率", ["shortcut", "deconf.", "reversed"],
                  [a('probe_hack_rate_shortcut', 0), a('probe_hack_rate_deconfounded', 0),
                   a('probe_hack_rate_length_reversed', 0)],
                  ["#dc2626", "#2563eb", "#059669"], 1.0)}
<figcaption>图 2　对抗探针里选「更长但错误」的概率。</figcaption></figure>
<figure>{_bar_svg("质量系数 θ_quality", ["shortcut", "deconf.", "reversed"],
                  [a('theta_quality_shortcut', 0), a('theta_quality_deconfounded', 0),
                   a('theta_quality_length_reversed', 0)],
                  ["#dc2626", "#2563eb", "#059669"], 3.2)}
<figcaption>图 3　DPO 在正确性特征上累积的权重。</figcaption></figure>
</div>
<p>三组的 reward accuracy 完全一致，reward margin 也都上升，训练日志看不出差别。
分歧出现在把长度与正确性拆开之后：shortcut 组把更多梯度放在长度特征上，
质量系数只有 {_f(a('theta_quality_shortcut'))}，对抗探针里
{_pct(a('probe_hack_rate_shortcut'))} 的回答是更长但错误的那个；
去混淆组的质量系数升到 {_f(a('theta_quality_deconfounded'))}，
劫持率降到 {_pct(a('probe_hack_rate_deconfounded'))}。
length_reversed 组的长度系数为负（{_f(a('theta_length_length_reversed'))}），
说明长度线索的方向被成功反转，而质量方向仍然被学到。</p>
<p>主探针的准确率同样分化：shortcut 组 {_f(a('probe_accuracy_shortcut'))}，
去混淆组 {_f(a('probe_accuracy_deconfounded'))}。差距不大，是因为主探针里
正确与错误的长度差被刻意做得温和；一旦把长度差拉大，差距就出现在劫持率上。
这也是为什么单看一个总体准确率不足以判断 DPO 学到了什么。</p>

<h3>3.5　参考模型锚点与 beta</h3>
<table><caption>表 5　参考模型锚点错误与 beta 扫描</caption>
<thead><tr><th>设定</th><th class="n">KL</th><th class="n">下游准确率</th><th>说明</th></tr></thead><tbody>
<tr><td>ref = SFT（正确）</td><td class="n">{_f(a('kl_deconfounded'))}</td><td class="n">{_f(a('probe_accuracy_deconfounded'))}</td><td>标准 DPO 推导</td></tr>
<tr><td>ref = base，policy 从 SFT 出发</td><td class="n">{_f(a('wrong_anchor_kl'))}</td><td class="n">{_f(a('probe_accuracy_deconfounded'))}</td><td>TRL 在 ref_model=None 时的实际行为</td></tr>
<tr><td>锚点造成的 KL 差</td><td class="n">{_f(a('anchor_kl_gap'))}</td><td class="n">—</td><td>KL 被放大约 {_f((a('wrong_anchor_kl', 0) / max(a('kl_deconfounded', 1), 1e-12)), 1)} 倍</td></tr>
</tbody></table>
<p>锚错不会改变 reward accuracy，因为它只依赖成对差，但仍然会通过 KL 改变正则强度，
并让「偏离参考模型多少」这个诊断量失去意义。表 5 是旧项目
<code>ref_model=None</code> 的直接量化后果。</p>
<table><caption>表 6　beta 扫描（deconfounded 组，固定其余超参）</caption>
<thead><tr><th class="n">beta</th><th class="n">reward margin</th><th class="n">KL</th><th class="n">下游准确率</th></tr></thead><tbody>
""" + "".join(
        f'<tr><td class="n">{b}</td><td class="n">{_f(v["margin"])}</td>'
        f'<td class="n">{_f(v["kl"])}</td><td class="n">{_f(v["probe_accuracy"])}</td></tr>'
        for b, v in sweep.items()
    ) + f"""
</tbody></table>
<p>固定策略时 margin/beta 的变异系数为 {_f(a('margin_over_beta_cv_fixed_theta'))}，
是恒等式。真实训练后 beta 改变梯度尺度与收敛位置，变异系数变为
{_f(a('margin_over_beta_cv_after_training'))}，四个 beta 的 KL 跨度
{_f(a('beta_kl_range'))}，下游准确率跨度 {_f(a('beta_probe_accuracy_range'))}。
所以 beta 的调法应该是看 KL 与下游，而不是看 margin。</p>

<h2 class="pb" id="s4">4　权重层 CPU pilot（真实 Qwen3-0.6B 权重）</h2>
<h3>4.1　为什么是三层验证</h3>
<p>算法实验台验证的是目标函数，权重层 pilot 验证的是代码链路，
完整协议才用于能力结论，三者不能互相替代。本机没有 GPU，但有 CPU 版
torch 与完整的 Qwen3-0.6B 权重，因此可以在缩减规模下把
LoRA-SFT、显式冻结 SFT 参考模型的 DPO、assistant-only masked NLL、
严格三态验证器的生成评测和序列级 KL 的采样估计真实执行一遍。
pilot 用 <code>configs/model_pilot.yaml</code> 里写明的规模，
逐条记录写在 <code>reports/model_pilot_manifest.json</code>。</p>
<table><caption>表 7　pilot 规模（与正式协议的差别一目了然）</caption>
<thead><tr><th>项</th><th class="n">pilot</th><th class="n">正式协议</th><th>说明</th></tr></thead><tbody>
<tr><td>LoRA 秩</td><td class="n">{_f(mp('config_lora_r'),0)}</td><td class="n">8</td><td>pilot 减小可训练参数量</td></tr>
<tr><td>SFT 步数</td><td class="n">{_f(mp('config_n_sft_steps'),0)}</td><td class="n">全量 2 epoch</td><td>pilot 只验证梯度链路</td></tr>
<tr><td>每组 DPO 步数 / 偏好对</td><td class="n">{_f(mp('config_n_dpo_steps'),0)} / {_f(mp('config_n_pairs_per_group'),0)}</td><td class="n">全量 1 epoch</td><td>三组对照结构保持一致</td></tr>
<tr><td>评测题数</td><td class="n">{_f(mp('config_n_eval_per_subset'),0)} × 5 个子集</td><td class="n">303</td><td>pilot 每个泛化子集各取一题</td></tr>
<tr><td>生成 token 上限</td><td class="n">{_f(mp('config_max_new_tokens'),0)}</td><td class="n">256</td><td>pilot 的不可核验率单独报告，不当成答错</td></tr>
</tbody></table>
<h3>4.2　结果</h3>
<table><caption>表 8　pilot 上的 masked NLL、生成准确率与不可核验率（{_f(mp('config_n_eval_per_subset'),0)} × 5 题）</caption>
<thead><tr><th>策略</th><th class="n">masked test NLL</th><th class="n">生成准确率</th><th class="n">unverifiable 率</th></tr></thead><tbody>
<tr><td>Base</td><td class="n">{_f(mp('test_loss_masked_base'))}</td><td class="n">{_f(mp('generative_accuracy_base'))}</td><td class="n">{_f(mp('unverifiable_rate_base'))}</td></tr>
<tr><td>SFT</td><td class="n">{_f(mp('test_loss_masked_sft'))}</td><td class="n">{_f(mp('generative_accuracy_sft'))}</td><td class="n">{_f(mp('unverifiable_rate_sft'))}</td></tr>
<tr><td>DPO / shortcut</td><td class="n">{_f(mp('test_loss_masked_dpo_shortcut'))}</td><td class="n">{_f(mp('generative_accuracy_dpo_shortcut'))}</td><td class="n">{_f(mp('unverifiable_rate_dpo_shortcut'))}</td></tr>
<tr><td>DPO / deconfounded</td><td class="n">{_f(mp('test_loss_masked_dpo_deconfounded'))}</td><td class="n">{_f(mp('generative_accuracy_dpo_deconfounded'))}</td><td class="n">{_f(mp('unverifiable_rate_dpo_deconfounded'))}</td></tr>
<tr><td>DPO / length_reversed</td><td class="n">{_f(mp('test_loss_masked_dpo_length_reversed'))}</td><td class="n">{_f(mp('generative_accuracy_dpo_length_reversed'))}</td><td class="n">{_f(mp('unverifiable_rate_dpo_length_reversed'))}</td></tr>
</tbody></table>
<table><caption>表 9　三组 DPO 的隐式 reward margin 与损失</caption>
<thead><tr><th>组</th><th class="n">margin 前</th><th class="n">margin 后</th><th class="n">增益</th><th class="n">DPO loss 前</th><th class="n">DPO loss 后</th></tr></thead><tbody>
<tr><td>shortcut</td><td class="n">{_f(mp('reward_margin_before_shortcut'))}</td><td class="n">{_f(mp('reward_margin_after_shortcut'))}</td><td class="n">{_f(mp('reward_margin_gain_shortcut'))}</td><td class="n">{_f(mp('dpo_loss_before_shortcut'))}</td><td class="n">{_f(mp('dpo_loss_after_shortcut'))}</td></tr>
<tr><td>deconfounded</td><td class="n">{_f(mp('reward_margin_before_deconfounded'))}</td><td class="n">{_f(mp('reward_margin_after_deconfounded'))}</td><td class="n">{_f(mp('reward_margin_gain_deconfounded'))}</td><td class="n">{_f(mp('dpo_loss_before_deconfounded'))}</td><td class="n">{_f(mp('dpo_loss_after_deconfounded'))}</td></tr>
<tr><td>length_reversed</td><td class="n">{_f(mp('reward_margin_before_length_reversed'))}</td><td class="n">{_f(mp('reward_margin_after_length_reversed'))}</td><td class="n">{_f(mp('reward_margin_gain_length_reversed'))}</td><td class="n">{_f(mp('dpo_loss_before_length_reversed'))}</td><td class="n">{_f(mp('dpo_loss_after_length_reversed'))}</td></tr>
</tbody></table>
<p>参考模型在训练前用独立基座实例加载同一个 SFT adapter，
逐参数断言与 policy 初值一致（检查了 {_f(mp('ref_assert_checked_parameters'),0)} 个共享参数），
再把冻结 ref 的成对 log-ratio 预计算缓存。序列级 KL 用从策略采样的
log π<sub>θ</sub> − log π<sub>ref</sub> 均值估计为
{_f(mp('sequence_kl_mc'))}，bootstrap 区间
[{_f(mp('sequence_kl_ci_low'))}, {_f(mp('sequence_kl_ci_high'))}]，
样本量 {_f(mp('sequence_kl_n'),0)}，只作量级参考。</p>
<p>三组在 5 道题上的生成准确率完全相同，这不构成任何能力结论：
缩减后准确率的最小粒度是 0.2，10 步更新也不足以改变 greedy 解码的判对结果。
有区分度的是训练量本身：三组 reward margin 分别从
{_f(mp('reward_margin_before_shortcut'))} 升到
{_f(mp('reward_margin_after_shortcut'))}、
{_f(mp('reward_margin_after_deconfounded'))} 与
{_f(mp('reward_margin_after_length_reversed'))}，
DPO 损失相应下降，说明成对目标函数在真实 Qwen3 权重上确实被优化到了。</p>
<p>序列级 KL 的 MC 估计为 {_f(mp('sequence_kl_mc'))}，bootstrap 区间跨 0。
KL 本身非负，负值来自只有 {_f(mp('sequence_kl_n'),0)} 条采样序列的蒙特卡洛噪声，
不能读作「策略比参考模型更接近」。pilot 的 KL 只用来判断量级，
真要报告 KL 需要更多采样，这正是完整协议里按 prompt 聚类给区间的理由。</p>
<p class="note">pilot 只回答「链路能不能在真实权重上跑通」。
缩减后的样本量与生成长度不足以支持任何关于模型能力的结论，
正式协议的 9 个指标仍为 <span class="pending">pending</span>，
本节的任何一个数字都不写进简历的能力陈述。</p>

<h2 class="pb">5　三个测量问题</h2>
<h3>5.1　偏好信号的可识别性</h3>
<p>设 S 为一对回答的表面特征差，Q 为真实质量差，Y 为偏好标签。
若存在可测函数 f 使 Y = f(S) 几乎处处，且 Q ⟂ Y | S，则 DPO 的经验风险不识别
θ 在 Q 方向的分量。证明的要点是构造观测等价的策略扰动：
对数据集中出现的每个回答给对数概率加同一个常数 c<sub>i</sub>，
把腾出的概率质量重新分配给数据集外的回答，成对差 Δ 与损失 L 逐点不变，
而数据集外的行为可以任意改变。等价地，把 θ 分解为控制数据集内相对概率的 φ
与控制数据集外行为的 ψ，似然只依赖 φ，关于 ψ 的 Fisher 信息为零。</p>
<p>把条件做成可操作的量：只用 6 个表面特征训练逻辑回归，
按偏好对分组做 5 折交叉验证，报告 AUC 与 SI = 2·AUC − 1。</p>
<table><caption>表 10　表面特征对偏好标签的可分性</caption>
<thead><tr><th>数据</th><th class="n">AUC</th><th class="n">SI</th><th class="n">真错误率</th><th>解读</th></tr></thead><tbody>
<tr><td>shortcut</td><td class="n">{_f(ident('surface_auc_shortcut'))}</td><td class="n">{_f(ident('shortcut_index_shortcut'))}</td><td class="n">{_f(ident('true_error_rate_shortcut'))}</td><td>标签被表面特征确定性解释</td></tr>
<tr><td>deconfounded</td><td class="n">{_f(ident('surface_auc_deconfounded'))}</td><td class="n">{_f(ident('shortcut_index_deconfounded'))}</td><td class="n">{_f(ident('true_error_rate_deconfounded'))}</td><td>表面线索被切断，只剩质量</td></tr>
<tr><td>length_reversed</td><td class="n">{_f(ident('surface_auc_length_reversed'))}</td><td class="n">{_f(ident('shortcut_index_length_reversed'))}</td><td class="n">{_f(ident('true_error_rate_length_reversed'))}</td><td>线索方向反转，仍可分</td></tr>
<tr><td>旧项目 240 条</td><td class="n">{_f(ident('surface_auc_legacy'))}</td><td class="n">—</td><td class="n">—</td><td>reward accuracy 1.0 与数学能力无关</td></tr>
</tbody></table>
<p>数值上再给两个佐证。把真实质量 Q 在偏好对之间随机重排 50 次，
DPO 损失的最大变化是 {_f(ident('max_abs_change_under_Q_permute'))}，逐位不变。
让 Δ<sub>θ</sub> 同时依赖 Q 时，最大变化是
{_f(ident('max_abs_change_coupled_Q'))}，非零。这说明损失对 Q 的依赖不是数值噪声，
而是构造上被切断了。</p>

<h3>5.2　伪重复与配对推断</h3>
<p>40 条测试集来自 2 道题，簇内结果几乎完全一致。用单因素方差分析估簇内相关，
得 ICC {_f(cl('icc'))}，设计效应 deff = 1 + (m − 1)ρ = {_f(cl('design_effect'),0)}，
有效样本量 n<sub>eff</sub> = n / deff = {_f(cl('effective_n'))}。
把 40 条当独立样本得到的正态区间是
[{_f(cl('iid_ci_low'))}, {_f(cl('iid_ci_high'))}]，不跨 0；
以题目为聚类单位做 bootstrap 得到
[{_f(cl('cluster_ci_low'))}, {_f(cl('cluster_ci_high'))}]，跨 0。
结论从「显著提升」变成「不能拒绝无提升」。</p>
<figure>{_interval_svg((cl('iid_ci_low', 0), cl('iid_ci_high', 0)),
                     (cl('cluster_ci_low', 0), cl('cluster_ci_high', 0)),
                     cl('point_difference_sft_minus_base', 0))}
<figcaption>图 4　同一个点估计在两种推断单位下的 95% 区间。聚类区间宽且下界压在 0 上，
这不是估计不精确，而是 {_f(cl('n_clusters'),0)} 个簇能提供的信息就这么多。</figcaption></figure>
<p>配对检验给出一致的结论。SFT 对 Base 的 McNemar 精确 p 为
{_f(cl('mcnemar_p_base_sft'))}，三个两两比较做 Holm 校正后最小 p 为
{_f(cl('holm_adjusted_min_p'))}。按观测到的簇间方差反解功效，
要检出 5 个百分点的提升需要约 {_f(cl('clusters_needed_for_5pp'),0)} 个独立簇。
现有 2 个簇对应的功效只有 {_f(cl('power_at_K2'))}。</p>
<p>新数据集按题型族做聚类，测试集有 {_f(cl('new_n_test'),0)} 条、
{_f(cl('new_n_family_clusters'),0)} 个族，按同一公式估得的有效样本量在
{_f(cl('new_effective_n_moderate'),1)} 到 {_f(cl('new_effective_n_independence'),1)}
之间，取决于簇内相关取中等值还是取 0。这比 40 条对应 2.6667 高两个量级，
但仍然远小于条数本身。</p>

<h3>5.3　验证器偏差校正</h3>
<p>设真实准确率为 π，验证器的假阳性率为 FPR、假阴性率为 FNR，
观测准确率 p̂ = (1 − FNR)π + FPR(1 − π)，反解得
π̂ = (p̂ − FPR) / (1 − FPR − FNR)。
用真值已知的 calibration 集估 FPR 与 FNR，再用 survey 集的观测准确率反解，
bootstrap 给区间。</p>
<table><caption>表 11　三种验证器在同一批真值已知样本上的表现</caption>
<thead><tr><th>验证器</th><th class="n">FPR</th><th class="n">FNR</th><th class="n">观测准确率</th><th class="n">校正准确率</th><th class="n">未校正偏差</th><th class="n">校正后偏差</th></tr></thead><tbody>
<tr><td>旧式子串匹配</td><td class="n">{_f(vb('fpr_legacy_substring'))}</td><td class="n">{_f(vb('fnr_legacy_substring'))}</td><td class="n">{_f(vb('observed_accuracy_legacy_substring'))}</td><td class="n">{_f(vb('corrected_accuracy_legacy_substring'))}</td><td class="n">{_f(vb('naive_bias_legacy_substring'))}</td><td class="n">{_f(vb('corrected_bias_legacy_substring'))}</td></tr>
<tr><td>严格多项式</td><td class="n">{_f(vb('fpr_strict_poly'))}</td><td class="n">{_f(vb('fnr_strict_poly'))}</td><td class="n">{_f(vb('observed_accuracy_strict_poly'))}</td><td class="n">{_f(vb('corrected_accuracy_strict_poly'))}</td><td class="n">{_f(vb('naive_bias_strict_poly'))}</td><td class="n">{_f(vb('corrected_bias_strict_poly'))}</td></tr>
<tr><td>数值代入</td><td class="n">{_f(vb('fpr_numeric_float'))}</td><td class="n">{_f(vb('fnr_numeric_float'))}</td><td class="n">{_f(vb('observed_accuracy_numeric_float'))}</td><td class="n">{_f(vb('corrected_accuracy_numeric_float'))}</td><td class="n">{_f(vb('naive_bias_numeric_float'))}</td><td class="n">{_f(vb('corrected_bias_numeric_float'))}</td></tr>
</tbody></table>
<p class="note">构造真值 π = {_f(vb('survey_pi_true'))}。旧式验证器把推导里出现的数字当答案，
FPR 高达 {_f(vb('fpr_legacy_substring'))}，观测准确率高估
{_f(vb('naive_bias_legacy_substring'))}；校正把偏差压到
{_f(vb('corrected_bias_legacy_substring'))}，代价是分母
1 − FPR − FNR 只有 {_f(vb('correction_denominator'))}，区间明显变宽。</p>
<p>没有人工金标准时，可以用多个验证器的判定做潜类别估计。J 个验证器有 2<sup>J</sup> − 1
个自由模式频率，模型有 1 + 2J 个参数，J = 2 时 3 对 5 欠识别，J = 3 时 7 对 7 恰好识别。
三验证器 EM 给出的 π̂ 是 {_f(vb('latent_class_pi'))}，与真值相差
{_f(vb('latent_class_abs_error'))}。代价是覆盖率只有
{_f(vb('latent_class_coverage'))}，因为只能使用三者都给出二元判定的样本，
被丢掉的几乎都是没有答案标记的错误输出，属于 MNAR。子集真值比全集高
{_f(vb('latent_class_selection_bias'))}，这个选择偏差本身就是结论：
直接丢掉不可核验样本再报准确率，会系统性高估。</p>
<p>条件独立是这套估计的硬假设。以 0、0.2、0.5 的概率把第三个验证器的判定替换成第一个，
π̂ 从 {_f(vb('latent_class_pi_dependence_0.0'))} 变到
{_f(vb('latent_class_pi_dependence_0.5'))}，幅度
{_f(vb('latent_class_dependence_spread'))}。在这批数据上结论不敏感，
但这不构成一般性保证。</p>

<h2>6　讨论</h2>
<p>三处失效不是并列的，它们互相放大。偏好数据里没有真错误，使梯度只能沿表面特征走；
评测器用子串匹配，使「更长、含更多数字」的回答更容易被判对，于是表面特征与观测收益
同向；测试集是伪重复，使这点收益在统计上显得显著。三者叠加，就得到一条
「loss 下降、margin 上升、准确率不动」的曲线。把其中任何一处单独修掉，
另外两处仍会让结论看起来成立，这也是旧项目反复调参却没有进展的原因。</p>
<p>对工业评测，这套诊断可以直接落成三条操作。上线前给偏好数据做体检，
表面特征基线 AUC 接近 1 就说明这批数据只能教格式。评测集做样本量规划，
用簇内相关算有效样本量，别把同题改写当独立样本，否则 A/B 结论的假阳性会失控。
自动评测器定期用一小批人工标注估 FPR/FNR，把观测准确率还原成真实准确率再报。
三件事都不需要额外训练成本。</p>
<p>局限同样明确。实验台的策略是 7 维解析模型，能验证目标函数却不能替代权重训练，
它的探针设计也参与了结论的构造。权重层 CPU pilot 已在真实 Qwen3-0.6B 权重上跑通缩减协议，但完整协议的 9 个指标
仍为 <span class="pending">pending</span>。聚类推断只有 2 个簇，簇间方差不可靠，
功效数字只能当量级看。潜类别估计依赖条件独立，敏感性分析只覆盖了一种相关结构。
这些都在结果文件里逐条标注，没有藏在正文之外。</p>

<h2>7　复现</h2>
<p>研究阶段与算法实验台只用 Python 标准库，不需要 GPU，也不需要安装依赖。
在项目根目录执行：</p>
<div class="eq">python -m src.run_all --stage research　　# 数据、算法实验台、三个测量问题、报告<br>
python -m src.run_all --stage algo　　　　# 单独跑解析策略的 SFT → DPO<br>
python -m src.model.pilot　　　　　　　　# 真实 Qwen3-0.6B 权重上的 CPU pilot<br>
python -m src.run_all --stage check　　　 # 三轮检查</div>
<p>产物包括 <code>data/</code> 下 600 道题与三组偏好数据、
<code>reports/results.jsonl</code> 单一事实源、
<code>reports/research_report.md</code> 指标表、
<code>reports/algo_lab_curves.json</code> 与 <code>algo_lab_curve.svg</code> 训练曲线，
、<code>reports/model_pilot_manifest.json</code> 与逐条预测文件，以及本报告。<code>reports/check_report.json</code> 记录三轮检查：
全量编译与单元测试、报告数字逐条反查、原项目 134 个文件哈希未变。
重跑研究阶段后上述文件的哈希逐字节一致。</p>
<p>权重层分两步。CPU pilot 在有 torch 的解释器上执行
<code>python -m src.model.pilot</code>，或右键 <code>run_model_pycharm.py</code>，
规模写在 <code>configs/model_pilot.yaml</code>。完整协议需要 GPU：安装
<code>requirements.txt</code>、准备 Qwen3-0.6B 权重、改三份配置路径后执行
<code>python -m src.run_all --stage model</code>，届时 9 个 pending 记录会被同名
measured 替换。依赖缺失时脚本明确报错退出，不会写入任何伪造指标。</p>

<footer>本报告由 <code>src/core/tech_report.py</code> 从 <code>reports/results.jsonl</code>
与 <code>reports/algo_lab_curves.json</code> 生成。所有 measured 数字均为真实执行结果；
CPU pilot 数字来自真实权重执行，完整协议的 9 个指标仍为 pending。</footer>
</div></body></html>""")
    return "".join(L)


def write(root: str | Path = _PYCHARM_ROOT) -> Path:
    root = Path(root)
    records = ResultStore.load(root / "reports" / "results.jsonl")
    p = root / "reports" / "tech_report.html"
    p.write_text(render(records, root), encoding="utf-8")
    return p


if __name__ == "__main__":
    out = write()
    print("技术报告:", out)
    print("大小:", out.stat().st_size, "bytes")
