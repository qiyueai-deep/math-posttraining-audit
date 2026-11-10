"""三态数学答案验证器。

旧实现的两处致命缺陷在这里被显式禁止:
  1. exact() 用 `gold in pred` 子串包含判对 -> 已删除, 不允许任何形式的包含匹配。
  2. symbolic() 在 SymPy 解析失败时静默回退到字符串匹配 -> 改为 UNVERIFIABLE。

判定只有三种结果: correct / incorrect / unverifiable。
unverifiable 既不算对也不算错, 单独统计并在偏差校正中显式建模。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import re
from dataclasses import dataclass, asdict
from fractions import Fraction

from src.core.expressions import ParseError, extract_answer, parse_answer_value, values_equal

CORRECT = "correct"
INCORRECT = "incorrect"
UNVERIFIABLE = "unverifiable"


@dataclass
class Verdict:
    verdict: str
    reason: str
    extracted: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class StrictVerifier:
    """主验证器: 受限语法精确解析 + 稀疏多项式精确相等。"""

    name = "strict_poly"

    def verify(self, text: str, gold: str) -> Verdict:
        got = extract_answer(text)
        if got is None:
            return Verdict(UNVERIFIABLE, "no explicit final-answer marker")
        try:
            parse_answer_value(gold)  # gold 本身必须可核验, 否则整题不可核验
        except ParseError as e:
            return Verdict(UNVERIFIABLE, f"gold not parseable: {e}", got)
        try:
            ok = values_equal(gold, got)
        except ParseError as e:
            return Verdict(UNVERIFIABLE, f"prediction not parseable: {e}", got)
        return Verdict(CORRECT if ok else INCORRECT, "exact polynomial equality", got)


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?")
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_TAIL = re.compile(r"(?:最终答案|答案)[:：]?\s*(.{0,80})")


class NumericVerifier:
    """第二验证器: 独立的抽取逻辑 + 浮点数值比较, 用于 latent class 估计。

    与 StrictVerifier 使用不同的答案抽取路径和不同的等价判定路径。
    两者并非严格条件独立(共享题面语言), 报告里会做条件独立假设的敏感性分析。
    """

    name = "numeric_float"
    points = (-3.0, -1.0, 0.0, 1.0, 2.0, 5.0)
    tol = 1e-6

    def _extract(self, text: str) -> str | None:
        boxed = _BOXED.findall(text or "")
        if boxed:
            return boxed[-1]
        tail = _TAIL.findall(text or "")
        return tail[-1] if tail else None

    @staticmethod
    def _numbers(s: str) -> list[float]:
        out = []
        for m in _NUM.findall(s):
            m = m.replace(" ", "")
            try:
                out.append(float(Fraction(m)))
            except (ValueError, ZeroDivisionError):
                pass
        return out

    def verify(self, text: str, gold: str) -> Verdict:
        got = self._extract(text)
        if got is None:
            return Verdict(UNVERIFIABLE, "no numeric final answer found")
        g, p = self._numbers(gold), self._numbers(got)
        if not g or not p:
            return Verdict(UNVERIFIABLE, "no numeric token", got)
        # 只比较紧随答案标记的那一段, 忽略推导正文里的数字
        p = p[-len(g):] if len(p) >= len(g) else p
        if len(p) != len(g):
            return Verdict(INCORRECT, "answer arity mismatch", got)
        ok = all(abs(a - b) <= self.tol * max(1.0, abs(a), abs(b)) for a, b in zip(g, p))
        return Verdict(CORRECT if ok else INCORRECT, "floating point comparison", got)


class LegacyLikeVerifier:
    """研究对照: 逐行复刻旧项目的宽松判定, 只用于测量偏差, 绝不用于主评测。

    旧 evaluate.py 的 exact() 先比较抽取出的最终答案, 再做整段输出的子串包含判定,
    且没有 UNVERIFIABLE 状态。它正是子贡献 3 要校正的有偏测量工具。
    """

    name = "legacy_substring"

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+|[,，。；;。.]", "", s).lower()

    def verify(self, text: str, gold: str) -> Verdict:
        m = re.search(r"(?:最终答案|答案)\s*[:：]?\s*([^\n。；;]+)", text or "")
        got = m.group(1).strip() if m else (text or "").strip()
        ans = self._norm(gold)
        pred = self._norm(text or "")
        ok = (ans == self._norm(got)) or (ans in pred)
        return Verdict(CORRECT if ok else INCORRECT,
                       "legacy substring containment (biased by construction)", got)


VERIFIERS = {
    "strict_poly": StrictVerifier(),
    "numeric_float": NumericVerifier(),
    "legacy_substring": LegacyLikeVerifier(),
}


def verify(text: str, gold: str, verifier: str = "strict_poly") -> Verdict:
    return VERIFIERS[verifier].verify(text, gold)


def binary(v: Verdict) -> int | None:
    """把三态压成二元判定, 供混淆矩阵使用。unverifiable 返回 None。"""
    if v.verdict == CORRECT:
        return 1
    if v.verdict == INCORRECT:
        return 0
    return None
