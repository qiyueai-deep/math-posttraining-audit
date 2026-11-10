"""序列级 KL 的蒙特卡洛估计与长度漂移诊断。

修复旧项目的致命问题 5: 旧 beta 扫描把 reward margin 随 beta 线性增长
当成发现, 其实 margin = beta * (delta_logpi_theta - delta_logpi_ref) 是恒等式。
这里直接估计真正的 KL(pi_theta || pi_ref), 用按 prompt 聚类的 bootstrap
给置信区间, 并报告长度漂移与格式率, 作为 reward hacking 的可观测量。
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
import random
from collections import defaultdict
from pathlib import Path

from src.model.teacher_forced import _apply_chat_template


def _completion_logprob(model, input_ids, prompt_len, torch, F):
    """给定完整序列, 返回 completion 部分的总 log p。"""
    ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, :-1, :]
    tgt = torch.tensor([input_ids[1:]], dtype=torch.long, device=logits.device)
    lp = F.log_softmax(logits.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(lp[0, prompt_len - 1:].sum())


def estimate_kl(policy, ref, tok, prompts: list[str], n_samples: int = 8,
                max_new_tokens: int = 128, seed: int = 20251109,
                B: int = 2000) -> dict:
    import torch
    import torch.nn.functional as F
    from src.model import check_deps
    check_deps()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    by_prompt: dict[int, list[float]] = defaultdict(list)
    lengths, formats = [], []
    policy.eval()
    ref.eval()
    for pi, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt")
        pids = enc["input_ids"][0].tolist()
        prompt_len = len(pids)
        for _ in range(n_samples):
            with torch.no_grad():
                gen = policy.generate(**{k: v.to(policy.device) for k, v in enc.items()},
                                      do_sample=True, top_p=0.95, temperature=1.0,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=tok.eos_token_id)
            full = gen[0].tolist()
            text = tok.decode(full[prompt_len:], skip_special_tokens=True)
            lengths.append(len(text))
            formats.append(int("最终答案" in text))
            lp_t = _completion_logprob(policy, full, prompt_len, torch, F)
            lp_r = _completion_logprob(ref, full, prompt_len, torch, F)
            by_prompt[pi].append(lp_t - lp_r)
    all_ratios = [v for vs in by_prompt.values() for v in vs]
    kl = sum(all_ratios) / len(all_ratios)
    keys = list(by_prompt)
    boots = []
    for _ in range(B):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        vals = [v for k in pick for v in by_prompt[k]]
        boots.append(sum(vals) / len(vals))
    boots.sort()
    return {
        "kl_estimate": kl,
        "ci_low": boots[int(0.025 * B)],
        "ci_high": boots[min(B - 1, int(0.975 * B))],
        "n_prompts": len(prompts), "n_samples_per_prompt": n_samples,
        "n_sequences": len(all_ratios), "bootstrap_B": B,
        "mean_output_chars": sum(lengths) / len(lengths),
        "format_rate": sum(formats) / len(formats),
        "cluster_unit": "prompt",
    }


def run_from_dirs(base_model: str, sft_dir: str, dpo_dir: str,
                  test_file: str, n_prompts: int = 20, seed: int = 20251109,
                  max_new_tokens: int = 128) -> dict:
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from src.model import check_deps
    check_deps()
    tok = AutoTokenizer.from_pretrained(base_model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype), dpo_dir)
    ref = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype), sft_dir)
    ref.requires_grad_(False)
    rows = [json.loads(l) for l in Path(test_file).read_text(encoding="utf-8").splitlines()]
    rows = rows[:n_prompts]
    prompts = [_apply_chat_template(tok, r["messages"][:2], add_generation_prompt=True)
               for r in rows]
    return estimate_kl(policy, ref, tok, prompts, seed=seed,
                       max_new_tokens=max_new_tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--sft_dir", required=True)
    ap.add_argument("--dpo_dir", required=True)
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--n_prompts", type=int, default=20)
    a = ap.parse_args()
    print(json.dumps(run_from_dirs(a.base_model, a.sft_dir, a.dpo_dir,
                                   a.test_file, a.n_prompts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
