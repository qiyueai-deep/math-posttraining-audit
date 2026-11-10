"""PyCharm 右键入口：现场跑一遍后训练算法实验台（零第三方依赖，无需 GPU）。

真实执行 SFT 最大似然训练与 DPO 梯度下降，打印三组偏好对照、beta 恒等式、
参考模型锚点对照，并生成 reports/algo_lab_curve.svg。

注意：这里训练的是 7 维可解析梯度的 log-linear 策略，用来验证目标函数与
优化行为，不是 Qwen3-0.6B 的 LoRA 权重。它不写 results.jsonl；
要生成正式报告请右键 run_pycharm.py（默认 research 阶段）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.core.results import ResultStore  # noqa: E402
from src.model import algo_lab  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("后训练算法实验台  SFT MLE -> DPO 梯度下降  (CPU, 零第三方依赖)")
    print("项目根目录:", ROOT)
    print("=" * 78, flush=True)

    rows = [json.loads(l) for name in ("train", "valid", "test")
            for l in (ROOT / "data" / f"{name}.jsonl").read_text(
                encoding="utf-8").splitlines()]
    store = ResultStore(ROOT)
    out = algo_lab.run(store, ROOT, rows, seed=20251109)

    chk = algo_lab.self_test()
    print()
    print(f"梯度校验  解析梯度 vs 中心有限差分: 最大绝对误差 "
          f"{chk['gradient_max_abs_error']:.2e}，KL(θ||θ)={chk['kl_self']:.1f}")
    print(f"探针 {out['n_probe']} 题，SFT 用 {out['sft']['n_train']} 条指令，"
          f"SFT 后主探针准确率 {out['sft_scores']['accuracy']:.4f}")
    print()
    print("三组偏好对照（reward accuracy 相同，下游行为不同）")
    print("-" * 78)
    print(f"{'组':<20}{'reward acc':>12}{'margin':>10}{'hack rate':>12}"
          f"{'θ_quality':>12}{'θ_length':>12}{'KL':>10}")
    for g, r in out["groups"].items():
        print(f"{g:<20}{r['reward_accuracy']:>12.4f}{r['reward_margin']:>10.4f}"
              f"{r['probe_hack_rate']:>12.4f}{r['theta_quality']:>12.4f}"
              f"{r['theta_length']:>12.4f}{r['kl']:>10.4f}")
    print("-" * 78)
    print(f"beta 恒等式  固定策略 margin/beta 变异系数 "
          f"{out['margin_over_beta_cv_fixed_theta']:.4f}；"
          f"真实训练后 {out['margin_over_beta_cv_after_training']:.4f}")
    print(f"参考模型锚点  ref=SFT 时 KL {out['anchor']['correct_kl']:.4f}，"
          f"锚错到 base 时 KL {out['anchor']['wrong_kl']:.4f}")
    print()
    print("曲线数据:", ROOT / "reports" / "algo_lab_curves.json")
    print("曲线图  :", ROOT / "reports" / "algo_lab_curve.svg")
    print()
    print("边界：以上是解析策略上的算法机制验证，不是 Qwen3 权重训练结果。")
    print("正式报告与单一事实源请右键 run_pycharm.py；算法岗问答见")
    print("reports/algorithm_interview.md")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
