"""极简扁平 YAML 读写，纯标准库。

核心层不依赖 PyYAML，这样在 PyCharm 里随便选一个 Python 3.10+ 解释器，
不用 pip install 任何东西就能右键跑通研究阶段和三轮检查。
只支持本项目配置里出现的形态: 标量、布尔、整数、浮点数、方括号列表、注释。
"""

from __future__ import annotations

import re
from pathlib import Path


def load(path: str | Path) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if not s or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k, v = k.strip(), v.strip()
        if not v:
            continue
        if v.startswith("[") and v.endswith("]"):
            out[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        elif v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        elif re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
        elif re.fullmatch(r"-?\d+\.\d+", v):
            out[k] = float(v)
        else:
            out[k] = v.strip("'\"")
    return out


def dump(data: dict, path: str | Path) -> None:
    lines = []
    for k, v in data.items():
        if isinstance(v, list):
            s = "[" + ", ".join(str(x) for x in v) + "]"
        elif isinstance(v, bool):
            s = "true" if v else "false"
        elif isinstance(v, (int, float)):
            s = str(v)
        else:
            s = str(v)
            if ":" in s or " " in s or not s:
                s = "'" + s.replace("'", "''") + "'"
        lines.append(f"{k}: {s}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
