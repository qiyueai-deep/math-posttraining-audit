"""assistant-only teacher-forced 指标。

修复旧项目的致命问题 3: 旧 evaluate_teacher_forced.py 直接
`labels = input_ids.clone()`, 对整段对话(含 system + user)算 loss 和 token
accuracy。测试题题干重复 20 遍, 模型只要记住题干就能把 loss 大幅拉低,
与数学能力无关。

这里用 offset_mapping 在字符级精确定位 prompt 边界, 只对 assistant 的
completion token 计算 NLL 与 token accuracy, 并按 token 加权而非样本等权。
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
import math
from pathlib import Path


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    kw = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kw)


def build_labels(tokenizer, messages: list[dict]) -> tuple[list[int], list[int], int]:
    """返回 (input_ids, labels, n_completion_tokens), labels 里 prompt 段为 -100。

    只依赖 tokenizer 的接口, 不依赖 torch, 因此可以用假 tokenizer 做单元测试。
    """
    full = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt = _apply_chat_template(tokenizer, messages[:2], add_generation_prompt=True)
    if not full.startswith(prompt):
        raise ValueError("chat template mismatch: full text does not start with prompt text")
    enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    cut = len(prompt)
    labels = [tid if off[0] >= cut else -100 for tid, off in zip(ids, offsets)]
    n_completion = sum(1 for x in labels if x != -100)
    if n_completion == 0:
        raise ValueError("no completion tokens; refusing to report a prompt-only loss")
    return ids, labels, n_completion


def masked_nll(model, input_ids, labels, torch, F):
    """token 加权的 completion NLL 与 token accuracy。"""
    ids = torch.tensor([input_ids], dtype=torch.long)
    lab = torch.tensor([labels], dtype=torch.long)
    attn = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    targets = lab[:, 1:]
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1), ignore_index=-100,
                            reduction="sum")
    n = int((targets != -100).sum())
    if n == 0:
        raise ValueError("empty completion after shift")
    loss = float(total.detach()) / n
    with torch.no_grad():
        pred = logits.argmax(-1)
        hit = int(((pred == targets) & (targets != -100)).sum())
    return {"nats": loss, "n_tokens": n, "correct_tokens": hit}


def _load(path: str, base_model: str | None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    is_adapter = Path(path, "adapter_config.json").exists()
    root = base_model if is_adapter else path
    tok = AutoTokenizer.from_pretrained(path if Path(path).exists() else root)
    model = AutoModelForCausalLM.from_pretrained(root, torch_dtype=torch.float32)
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path)
    model.eval()
    return tok, model


def score(path: str, test_file: str, base_model: str | None = None) -> dict:
    import torch
    import torch.nn.functional as F
    from src.model import check_deps
    check_deps()
    tok, model = _load(path, base_model)
    rows = [json.loads(l) for l in Path(test_file).read_text(encoding="utf-8").splitlines()]
    total_nats = total_tokens = correct_tokens = 0
    for r in rows:
        ids, labels, _ = build_labels(tok, r["messages"])
        v = masked_nll(model, ids, labels, torch, F)
        total_nats += v["nats"] * v["n_tokens"]
        total_tokens += v["n_tokens"]
        correct_tokens += v["correct_tokens"]
    if total_tokens == 0:
        raise ValueError("no completion tokens across the test set")
    loss = total_nats / total_tokens
    return {
        "n": len(rows),
        "teacher_forced_loss": loss,
        "perplexity": math.exp(min(loss, 20)),
        "token_accuracy": correct_tokens / total_tokens,
        "completion_tokens": total_tokens,
        "prompt_masked": True,
        "token_weighted": True,
    }


class _FakeTokenizer:
    """仅用于 --self-test: 用字符当 token, 验证 prompt mask 的边界正确性。"""

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        text = "".join(m["role"] + ":" + m["content"] + "\n" for m in messages)
        return text + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        ids = [ord(c) % 251 for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return {"input_ids": ids, "offset_mapping": offsets} if return_offsets_mapping else {"input_ids": ids}


def self_test() -> dict:
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "ANSWER"},
    ]
    ids, labels, n = build_labels(tok, messages)
    assert len(ids) == len(labels)
    assert any(x == -100 for x in labels), "prompt tokens must be masked"
    assert n == len("ANSWER\n"), f"unexpected completion length: {n}"
    assert all(x != -100 for x in labels[-n:])
    return {"n_total_tokens": len(ids), "n_masked": len(ids) - n,
            "n_completion": n, "ok": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=False)
    ap.add_argument("--sft_dir")
    ap.add_argument("--dpo_dir")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    if not a.base_model:
        raise SystemExit("--base_model is required unless --self-test")
    result = {}
    for name, path in (("base", a.base_model), ("sft", a.sft_dir), ("dpo", a.dpo_dir)):
        if path and Path(path).exists():
            result[name] = score(path, a.test_file, a.base_model)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
