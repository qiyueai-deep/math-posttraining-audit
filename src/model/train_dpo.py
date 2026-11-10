"""修正版 DPO: 显式冻结 SFT 参考模型, 不再把参考模型参数置空。

修复旧项目的致命问题 6: 旧 train_dpo.py 与 sweep_beta.py 都把参考模型参数置空,
而 model 是 PeftModel。TRL 在该情形下会 disable adapter, 于是 KL 锚点是
未训练的 base, 不是 SFT。旧目录里 runs/qwen3-dpo/ref/ 保存了 SFT adapter
的副本, 但训练代码从未引用它, 属于「看起来做对了」。

这里用独立的基座实例加载冻结的 SFT adapter 作为 ref_model, 并在训练前
逐参数断言 ref 与 policy 初始状态一致。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import argparse
import json
import random
import shutil
from pathlib import Path

from src.core.config import load as load_config


def build_models(base_model: str, sft_adapter: str, torch_dtype=None):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    # policy 与 ref 必须使用互不共享权重的基座实例
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype),
        sft_adapter, is_trainable=True)
    ref = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype),
        sft_adapter, is_trainable=False)
    ref.requires_grad_(False)
    ref.eval()
    return policy, ref


def assert_ref_matches_policy(policy, ref) -> dict:
    import torch
    p = dict(policy.named_parameters())
    r = dict(ref.named_parameters())
    shared = sorted(set(p) & set(r))
    if not shared:
        raise AssertionError("policy and ref share no named parameters")
    bad = [k for k in shared if not torch.allclose(p[k].detach().cpu(),
                                                   r[k].detach().cpu(), atol=1e-6)]
    if bad:
        raise AssertionError(f"ref differs from SFT policy at init: {bad[:5]}")
    return {"checked_parameters": len(shared), "ok": True}


def train(config_path: str, preference_group: str = "deconfounded",
          beta_override: float | None = None) -> dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from src.model import check_deps
    check_deps()

    root = Path(__file__).resolve().parents[2]
    p = Path(config_path).resolve()
    c = load_config(p)
    seed = int(c["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    def local(v):
        q = Path(str(v))
        return str(q if q.is_absolute() else root / q)

    base = local(c["base_model"])
    sft = local(c["sft_adapter"])
    out = Path(local(c["output_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    if not Path(base, "config.json").exists():
        raise SystemExit(f"基座权重目录不完整: {base}")
    if not Path(sft, "adapter_config.json").exists():
        raise SystemExit(f"SFT adapter 不存在: {sft}，请先运行 train_sft.py")

    # 偏好数据只保留 DPOTrainer 需要的三列, 避免 TRL 对额外列的处理差异
    pref_file = root / "data" / f"preferences_{preference_group}.jsonl"
    if not pref_file.exists():
        raise SystemExit(f"偏好文件不存在: {pref_file}")
    ds = load_dataset("json", data_files={"train": str(pref_file)})["train"]
    ds = ds.select_columns(["prompt", "chosen", "rejected"])
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy, ref = build_models(base, sft)
    check = assert_ref_matches_policy(policy, ref)
    beta = float(beta_override if beta_override is not None else c["beta"])
    args = DPOConfig(
        output_dir=str(out), num_train_epochs=int(c["num_train_epochs"]),
        learning_rate=float(c["learning_rate"]), beta=beta,
        max_length=int(c["max_length"]), max_prompt_length=int(c["max_prompt_length"]),
        per_device_train_batch_size=int(c["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(c["gradient_accumulation_steps"]),
        logging_steps=1, save_strategy="no",
        bf16=torch.cuda.is_available(), report_to="none", seed=seed)
    trainer = DPOTrainer(model=policy, ref_model=ref, args=args,
                         train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))

    # 保存真正被使用的 ref 快照, 而不是摆设
    ref_dir = out / "ref"
    ref_dir.mkdir(exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        src = Path(sft) / name
        if src.exists():
            shutil.copy2(src, ref_dir / name)

    hist = trainer.state.log_history
    margins = [h["rewards/margins"] for h in hist if "rewards/margins" in h]
    # margin 与 beta 成正比是恒等式, 报告去量纲的 margin/beta 才有意义
    tail = max(1, len(margins) // 5)
    tail_margins = margins[-tail:]
    summary = {
        "final_loss": hist[-1].get("loss") if hist else None,
        "final_reward_margin": margins[-1] if margins else None,
        "dimensionless_margin_over_beta": (sum(tail_margins) / tail / beta
                                          if tail_margins else None),
        "beta": beta, "preference_group": preference_group,
        "ref_model_explicit": True, "ref_matches_sft_at_init": check,
        "n_train": len(ds),
    }
    (out / "training_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--preference_group", default="deconfounded",
                    choices=("deconfounded", "shortcut", "length_reversed"))
    ap.add_argument("--beta", type=float, default=None)
    a = ap.parse_args()
    print(json.dumps(train(a.config, a.preference_group, a.beta),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
