"""PyCharm 一键三轮检查，右键 Run 即可，不需要任何参数。

等价于在项目根目录执行 python -m tests.run_three_round_checks。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def main() -> int:
    print("=" * 68)
    print("三轮检查: 语法与单元测试 / 数字一致性 / 需求与原项目完整性")
    print("项目根目录:", ROOT)
    print("=" * 68, flush=True)
    rc = subprocess.run([sys.executable, "-m", "tests.run_three_round_checks"],
                        cwd=ROOT, check=False).returncode
    print("完整结果:", ROOT / "reports" / "check_report.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
