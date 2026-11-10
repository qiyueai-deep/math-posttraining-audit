"""PyCharm 主入口，右键 Run 即可，不需要任何参数。

默认跑研究阶段(零第三方依赖, 约 8 秒)。用 --stage 切换:
    research  生成数据 + 算法实验台 + 三个子贡献, 产出报告
    algo      单独演示 SFT -> DPO 算法实验台并出训练曲线
    check     三轮检查
    model     模型训练与评测(需要 GPU 环境与 requirements.txt)
    all       全部

脚本会主动把工作目录切到项目根, 因此 PyCharm 的 Working directory
无论设成什么都不会出现找不到 data/ 或 reports/ 的问题。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

STAGES = {
    "research": ([sys.executable, "-m", "src.run_research"],
                 "生成参数化数据 + 算法实验台 + 三个子贡献 + 技术报告"),
    "algo": ([sys.executable, str(ROOT / "algo_lab_pycharm.py")],
             "现场跑 SFT→DPO 算法实验台并出训练曲线"),
    "check": ([sys.executable, "-m", "tests.run_three_round_checks"],
              "三轮检查"),
    "model": ([sys.executable, "-m", "src.run_all", "--stage", "model"],
              "模型训练、严格评测与 KL 诊断"),
    "all": ([sys.executable, "-m", "src.run_all", "--stage", "all"],
            "研究阶段 + 模型阶段 + 三轮检查"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="PyCharm 一键入口")
    ap.add_argument("--stage", choices=tuple(STAGES), default="research",
                    help="默认 research, 不填参数直接右键 Run 就是它")
    a = ap.parse_args()
    cmd, desc = STAGES[a.stage]
    print("=" * 68)
    print("项目根目录:", ROOT)
    print("当前阶段:", a.stage, "-", desc)
    print("执行命令:", " ".join(cmd))
    print("=" * 68, flush=True)
    rc = subprocess.run(cmd, cwd=ROOT, check=False).returncode
    print("=" * 68)
    if a.stage in ("research", "all"):
        print("报告位置:")
        print("  ", ROOT / "reports" / "research_report.md")
        print("  ", ROOT / "reports" / "resume_bullets.md")
        print("  ", ROOT / "reports" / "resume_rewrite.md")
        print("  ", ROOT / "reports" / "interview_guide.md")
        print("  ", ROOT / "reports" / "algorithm_interview.md")
        print("  ", ROOT / "reports" / "algo_lab_curve.svg")
    if a.stage in ("check", "all"):
        print("检查结果:", ROOT / "reports" / "check_report.json")
    print("退出码:", rc)
    print("=" * 68)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
