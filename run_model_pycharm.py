"""PyCharm 右键入口：在真实 Qwen3-0.6B 权重上跑 CPU pilot。

当前解释器若没有 torch，会自动寻找本机已有的 Python 3.12 与旧项目里
已经装好的 CPU 版依赖（只读引用，不修改原项目），再用子进程执行
`python -m src.model.pilot`。

这只跑缩减规模的 pilot，规模写在 configs/model_pilot.yaml。
完整 256-token 协议仍需 GPU，9 个 model_stage 指标保持 pending。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEPS = ROOT.parent / "math-posttraining-project" / ".python-deps"
PY312_CANDIDATES = [
    Path(r"C:\Users\24399\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"),
    Path(r"C:\Users\24399\.cache\codex-runtimes\codex-primary-runtime.previous-bb757e04-2db3-4d08-be14-ed358004f9cf\dependencies\python\python.exe"),
]


def find_python() -> tuple[list[str], dict]:
    if importlib.util.find_spec("torch") is not None:
        return [sys.executable, "-m", "src.model.pilot"], dict(os.environ)
    if not DEPS.is_dir():
        raise SystemExit(
            "当前解释器没有 torch，且找不到已有的 CPU 依赖目录：\n  " + str(DEPS)
            + "\n请安装 requirements.txt，或改用有 torch 的解释器。")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DEPS) + os.pathsep + env.get("PYTHONPATH", "")
    for py in PY312_CANDIDATES:
        if not py.is_file():
            continue
        probe = subprocess.run([str(py), "-c", "import torch, transformers, peft"],
                               env=env, capture_output=True, text=True)
        if probe.returncode == 0:
            return [str(py), "-m", "src.model.pilot"], env
    raise SystemExit("找不到能加载 torch 的 Python 解释器。已检查："
                     + ", ".join(str(p) for p in PY312_CANDIDATES))


def main() -> int:
    cmd, env = find_python()
    print("=" * 78)
    print("Qwen3-0.6B 权重层 CPU pilot（缩减协议，不是正式评测）")
    print("解释器:", cmd[0])
    print("配置  :", ROOT / "configs" / "model_pilot.yaml")
    print("依赖  :", DEPS if importlib.util.find_spec("torch") is None else "当前解释器")
    print("预计  : 8 到 12 分钟，进度会逐条打印")
    print("=" * 78, flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
