"""严格表达式解析与精确比较，零第三方依赖。

只接受一个受限的算术/整多项式语法。解析失败时明确抛 ParseError,
由 verifier 记为 UNVERIFIABLE, 绝不猜测, 绝不回退字符串包含匹配。

值域用稀疏多项式 Poly: monomial -> Fraction。
monomial 是 ((var, exponent), ...) 的规范序元组, 常量项为 ()。
这样 (x-2)^2-1 与 x^2-4x+3 能精确判等, 5/6 与 10/12 也能精确判等。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import re
from fractions import Fraction

__all__ = [
    "ParseError", "Poly", "normalize", "parse", "parse_answer_value",
    "extract_answer", "values_equal",
]


class ParseError(ValueError):
    """表达式无法在本语法内解析。调用方必须把它当作不可核验, 而不是错误答案。"""


Poly = dict[tuple[tuple[str, int], ...], Fraction]


# ---------------------------------------------------------------- 预处理

_LATEX_FRAC = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_LATEX_SQRT = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_LATEX_CMD = re.compile(r"\\(?:left|right|mathrm|text|displaystyle|,|;|!|quad)")
# 只认明确的答案标记。绝不把正文里的「结果为 ...」「答案是 ... 无需推导」之外的
# 自然语言片段当答案, 更不做 gold-in-pred 的包含匹配。
_ANSWER_RE = re.compile(r"(?:最终答案|答案)\s*(?:是|为)?\s*[:：=]?\s*([^\n。；;！!？?]+)")


def normalize(s: str) -> str:
    """把 LaTeX / 中文排版噪声归一化为受限语法。不做任何语义猜测。"""
    if s is None:
        raise ParseError("empty input")
    s = str(s).strip()
    if not s:
        raise ParseError("empty input")
    s = s.replace("\u2212", "-").replace("\u00d7", "*").replace("\u00f7", "/")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("$", "").replace("（", "(").replace("）", ")")
    s = s.replace("\\[", "").replace("\\]", "")
    # \boxed{...} 与 \frac{a}{b} 必须在其它命令清理之前展开
    s = _LATEX_BOXED.sub(r"(\1)", s)
    for _ in range(4):
        new = _LATEX_FRAC.sub(r"((\1)/(\2))", s)
        if new == s:
            break
        s = new
    s = _LATEX_SQRT.sub(r"((\1)^(1/2))", s)
    s = _LATEX_CMD.sub("", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"\s+", "", s)
    s = s.replace("**", "^")
    return s


def extract_answer(text: str) -> str | None:
    """抽取最终答案表达式。找不到明确答案标记时返回 None。

    与旧实现的关键差异: 不把整段输出当答案, 不做 '答案在输出里出现过就算对'。
    """
    if not text:
        return None
    boxed = _LATEX_BOXED.findall(text)
    if boxed and boxed[-1].strip():
        return boxed[-1].strip()
    hits = _ANSWER_RE.findall(text)
    if not hits:
        return None
    tail = hits[-1].strip()
    # 只取紧随分隔符后的第一个表达式片段, 后面的自然语言一律丢弃
    # 英文逗号是列表答案的分隔符(如特征值 1,-5), 不能当作句子边界切掉。
    tail = re.split(r"[，、。;；\n]|(?<=\d)\s*(?=[一-鿿])", tail)[0]
    tail = tail.strip().rstrip("。.，,；;:：")
    # 剥掉 "x=4" 之类的变量名前缀
    m = re.fullmatch(r"[A-Za-z]\s*=\s*(.+)", tail)
    if m:
        tail = m.group(1)
    return tail or None


# ---------------------------------------------------------------- 多项式代数

def _const(c: Fraction) -> Poly:
    return {(): c} if c else {}


def _var(name: str) -> Poly:
    return {((name, 1),): Fraction(1)}


def _add(a: Poly, b: Poly) -> Poly:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + v
    return {k: v for k, v in out.items() if v}


def _neg(a: Poly) -> Poly:
    return {k: -v for k, v in a.items()}


def _mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            merged: dict[str, int] = dict(ka)
            for name, e in kb:
                merged[name] = merged.get(name, 0) + e
            key = tuple(sorted((n, e) for n, e in merged.items() if e))
            out[key] = out.get(key, Fraction(0)) + va * vb
    return {k: v for k, v in out.items() if v}


def _int_pow(a: Poly, n: int) -> Poly:
    if n < 0 or n > 24:
        raise ParseError("exponent out of supported range")
    out = _const(Fraction(1))
    for _ in range(n):
        out = _mul(out, a)
    return out


def _is_const(a: Poly) -> bool:
    return all(k == () for k in a)


def _div(a: Poly, b: Poly) -> Poly:
    # 只允许除以非零常数。有理函数不在支持范围内, 明确拒绝而不是近似。
    if not _is_const(b) or () not in b:
        raise ParseError("division by a non-constant expression")
    d = b[()]
    if d == 0:
        raise ParseError("division by zero")
    return {k: v / d for k, v in a.items()}


# ---------------------------------------------------------------- 递归下降解析

_TOKEN = re.compile(r"\s*(\d+\.\d+|\d+|[A-Za-z_][A-Za-z_0-9]*|[()+\-*/^,])")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _tokenize(s: str) -> list[str]:
    out, pos = [], 0
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ParseError(f"unexpected character at {pos}: {s[pos]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[str]):
        self.t = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str:
        if self.i >= len(self.t):
            raise ParseError("unexpected end of expression")
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, tok: str) -> None:
        got = self.take()
        if got != tok:
            raise ParseError(f"expected {tok!r}, got {got!r}")

    def parse(self) -> Poly:
        v = self.expr()
        if self.i != len(self.t):
            raise ParseError(f"trailing tokens: {self.t[self.i:]}")
        return v

    def expr(self) -> Poly:
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            rhs = self.term()
            v = _add(v, rhs) if op == "+" else _add(v, _neg(rhs))
        return v

    def term(self) -> Poly:
        v = self.power()
        while True:
            p = self.peek()
            if p in ("*", "/"):
                self.take()
                rhs = self.power()
                v = _mul(v, rhs) if p == "*" else _div(v, rhs)
            elif p is not None and (_IDENT.fullmatch(p) or p == "("):
                # 隐式乘法: 4x 与 2(x+1), 数学书写习惯
                v = _mul(v, self.power())
            else:
                break
        return v

    def power(self) -> Poly:
        v = self.unary()
        if self.peek() == "^":
            self.take()
            # 指数必须是括号里的有理常数或正整数, 且化简后结果为多项式
            e = self.exponent()
            if not _is_const(e) or () not in e:
                raise ParseError("symbolic exponent")
            q = e[()]
            if q.denominator not in (1, 2):
                raise ParseError("unsupported exponent")
            if q.denominator == 2:
                # 只允许完全平方根, 否则明确拒绝
                root = _exact_sqrt(v)
                if root is None:
                    raise ParseError("non-perfect square under sqrt")
                v = root
                if q != Fraction(1, 2):
                    v = _int_pow(v, q.numerator)
            else:
                v = _int_pow(v, int(q))
        return v

    def exponent(self) -> Poly:
        sign = 1
        if self.peek() == "-":
            self.take()
            sign = -1
        if self.peek() == "(":
            self.take()
            v = self.expr()
            self.expect(")")
            return _neg(v) if sign < 0 else v
        return self.atom()

    def unary(self) -> Poly:
        if self.peek() == "-":
            self.take()
            return _neg(self.unary())
        if self.peek() == "+":
            self.take()
            return self.unary()
        return self.atom()

    def atom(self) -> Poly:
        tok = self.take()
        if tok == "(":
            v = self.expr()
            self.expect(")")
            return v
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            return _const(Fraction(tok))
        if _IDENT.fullmatch(tok):
            # 只接受短数学符号。否则 "not an expression at all" 会被
            # normalize 压成一个长标识符, 被误当成合法符号变量。
            if len(tok) > 2:
                raise ParseError(f"identifier is not a math symbol: {tok!r}")
            return _var(tok)
        raise ParseError(f"unexpected atom {tok!r}")


def _exact_sqrt(a: Poly) -> Poly | None:
    """只处理可分平方根: 常数完全平方, 或单项式的完全平方。"""
    if not a:
        return {}
    if len(a) == 1 and _is_const(a):
        c = a[()]
        if c < 0:
            return None
        n, d = c.numerator, c.denominator
        rn, rd = _isqrt_exact(n), _isqrt_exact(d)
        if rn is None or rd is None:
            return None
        return _const(Fraction(rn, rd))
    if len(a) == 1:
        (key, coef), = a.items()
        if coef < 0 or any(e % 2 for _, e in key):
            return None
        rn, rd = _isqrt_exact(coef.numerator), _isqrt_exact(coef.denominator)
        if rn is None or rd is None:
            return None
        return {tuple(sorted((n, e // 2) for n, e in key if e // 2)): Fraction(rn, rd)}
    return None


def _isqrt_exact(n: int) -> int | None:
    import math
    r = math.isqrt(n)
    return r if r * r == n else None


def parse(s: str) -> Poly:
    """把字符串解析为规范多项式。任何不确定性都抛 ParseError。"""
    return _Parser(_tokenize(normalize(s))).parse()


def _parse_one(s: str) -> Poly:
    return parse(s)


def parse_answer_value(s: str) -> list[Poly]:
    """答案可以是单个表达式, 也可以是逗号分隔的列表(如特征值 1,2)。"""
    s = normalize(s)
    parts = [p for p in s.split(",") if p]
    if not parts:
        raise ParseError("empty answer")
    return [_parse_one(p) for p in parts]


def values_equal(gold: str, pred: str) -> bool:
    """精确相等判定。列表按无序多重集合比较。"""
    g, p = parse_answer_value(gold), parse_answer_value(pred)
    if len(g) != len(p):
        return False
    # 单项式字典不可 hash, 用排序后的规范串做多重集合比较
    def key(v: Poly) -> str:
        return repr(sorted((k, str(c)) for k, c in v.items()))
    return sorted(map(key, g)) == sorted(map(key, p))
