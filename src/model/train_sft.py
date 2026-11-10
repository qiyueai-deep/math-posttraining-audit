"""修正版 LoRA-SFT: 手写训练循环, 只对 assistant completion 计算损失。

不用 TRL SFTTrainer 的隐式模板行为, 避免不同版本 assistant_only_loss
语义变化带来的静默差异。保存 adapter、tokenizer、配置与数据哈希, 保证可复现。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import argparse
import hashlib
import json
import random
from pathlib import Path

from src.core.config import load as load_config


def train(config_path: str) -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from src.model import check_deps
    from src.model.teacher_forced import build_labels
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

    train_file, valid_file, model_name = (local(c["train_file"]),
                                         local(c["valid_file"]),
                                         local(c["model_name"]))
    if not Path(model_name, "config.json").exists():
        raise SystemExit(f"基座权重目录不完整: {model_name}")
    out = Path(local(c["output_dir"]))
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    lora = LoraConfig(r=int(c["lora_r"]), lora_alpha=int(c["lora_alpha"]),
                      lora_dropout=float(c["lora_dropout"]),
                      target_modules=c.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(c["learning_rate"]))
    rows = [json.loads(l) for l in Path(train_file).read_text(encoding="utf-8").splitlines()]
    valid_rows = [json.loads(l) for l in Path(valid_file).read_text(encoding="utf-8").splitlines()]
    bs = int(c["per_device_train_batch_size"])
    accum = int(c["gradient_accumulation_steps"])
    max_len = int(c["max_seq_length"])

    def make_batch(batch):
        seqs = [build_labels(tok, r["messages"]) for r in batch]
        L = min(max(len(s[0]) for s in seqs), max_len)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y, _) in enumerate(seqs):
            x, y = x[:L], y[:L]
            ids[i, :len(x)] = torch.tensor(x)
            lab[i, :len(y)] = torch.tensor(y)
            attn[i, :len(x)] = 1
        if int((lab[:, 1:] != -100).sum()) == 0:
            raise RuntimeError("batch has no completion tokens; prompt mask is wrong")
        return ids, lab, attn

    def flush_grads():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

    history, valid_history = [], []
    for epoch in range(int(c["num_train_epochs"])):
        random.Random(seed + epoch).shuffle(rows)
        pending = False
        for step in range(0, len(rows), bs):
            ids, lab, attn = make_batch(rows[step:step + bs])
            out_m = model(input_ids=ids, attention_mask=attn, labels=lab)
            (out_m.loss / accum).backward()
            pending = True
            if (step // bs + 1) % accum == 0:
                flush_grads()
                pending = False
            history.append({"epoch": epoch, "step": step // bs,
                            "loss": float(out_m.loss.detach())})
        if pending:
            flush_grads()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for vstep in range(0, len(valid_rows), bs):
                ids, lab, attn = make_batch(valid_rows[vstep:vstep + bs])
                o = model(input_ids=ids, attention_mask=attn, labels=lab)
                n = int((lab[:, 1:] != -100).sum())
                total += float(o.loss.detach()) * n
                count += n
        model.train()
        valid_history.append({"epoch": epoch, "masked_valid_loss": total / count,
                              "n_completion_tokens": count})
    model.save_pretrained(out)
    tok.save_pretrained(out)
    digest = hashlib.sha256(Path(train_file).read_bytes()).hexdigest()
    (out / "training_manifest.json").write_text(json.dumps({
        "config": str(p), "seed": seed, "train_sha256": digest,
        "valid_sha256": hashlib.sha256(Path(valid_file).read_bytes()).hexdigest(),
        "n_train": len(rows), "n_valid": len(valid_rows),
        "assistant_only_loss": True,
        "history": history[-10:], "valid_history": valid_history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(out), "n_steps": len(history),
            "final_loss": history[-1]["loss"],
            "final_masked_valid_loss": valid_history[-1]["masked_valid_loss"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    a = ap.parse_args()
    print(json.dumps(train(a.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
