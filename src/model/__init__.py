"""模型层(可选依赖 torch/transformers/peft/trl)。

本机若无这些依赖, 所有模型脚本都应给出明确报错与安装指引,
不得静默跳过, 更不得写入伪造指标。
"""

from __future__ import annotations

import importlib.util

REQUIRED = ("torch", "transformers", "peft", "trl", "datasets")


def check_deps(raise_on_missing: bool = True) -> dict[str, bool]:
    state = {m: importlib.util.find_spec(m) is not None for m in REQUIRED}
    if raise_on_missing and not all(state.values()):
        missing = [m for m, ok in state.items() if not ok]
        raise RuntimeError(
            "模型阶段依赖缺失: " + ", ".join(missing)
            + "。请执行 pip install -r requirements.txt (GPU 环境), "
            + "或改用 python -m src.run_research 只运行无需模型的研究阶段。"
        )
    return state
