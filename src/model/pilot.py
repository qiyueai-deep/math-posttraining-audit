"""Qwen3-0.6B 权重层的 CPU pilot：在真实权重上把后训练链路跑一遍。

这一层回答「模型层代码到底能不能跑」，它不是正式评测协议。
正式协议在 configs/eval.yaml（256 new tokens、完整 303 题测试集、四个泛化子集上的
配对推断），本机 CPU 跑不完，reports/results.jsonl 里 9 个 model_stage 指标
因此始终保持 pending_model_run，本脚本绝不替换它们。

pilot 用 configs/model_pilot.yaml 里明确写出的缩减规模，在真实 Qwen3-0.6B 权重上
真实执行：LoRA-SFT、显式冻结 SFT 参考模型的 DPO、三组偏好对照、
assistant-only masked NLL、严格三态验证器的生成评测、序列级 KL 的 MC 估计。
所有数字写入 results.jsonl 的 model_pilot 实验，note 里标注缩减口径。
"""

from __future__ import annotations

# PyCharm 右键直接运行本文件时, 把项目根加入 sys.path
import sys
from pathlib import Path
_PYCHARM_ROOT = Path(__file__).resolve().parents[2]
if str(_PYCHARM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYCHARM_ROOT))

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from src.core.config import load as load_config
from src.core.results import (ResultStore, append_records, render_all,
                              verify_consistency)
from src.core.verifier import CORRECT, VERIFIERS
from src.model.teacher_forced import _apply_chat_template, build_labels

NOTE = ("CPU pilot on real Qwen3-0.6B weights; reduced protocol "
        "(80 new tokens, 10 eval items, LoRA r=4); "
        "full 256-token protocol stays pending_model_run; "
        "these numbers only prove the pipeline executes, not model quality")


def _load_base(path: str, torch):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)


def _log(msg: str) -> None:
    print(f"[pilot] {msg}", flush=True)


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _stratified_eval(rows: list[dict], per_subset: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 17)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["subset"]].append(r)
    out = []
    for key in sorted(by):
        pool = list(by[key])
        rng.shuffle(pool)
        out.extend(pool[:per_subset])
    return out


def _system(train_rows: list[dict]) -> str:
    return train_rows[0]["messages"][0]["content"]


def _pair_messages(pair: dict, system: str) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": pair["prompt"]}]


def _seq(tok, messages: list[dict], answer: str, max_len: int):
    msgs = list(messages[:2]) + [{"role": "assistant", "content": answer}]
    ids, labels, n = build_labels(tok, msgs)
    return ids[:max_len], labels[:max_len], n


def _completion_logp(model, ids, labels, torch, F, keep_graph: bool):
    """用 build_labels 的 mask 精确取 completion 部分的 log p。"""
    x = torch.tensor([ids], dtype=torch.long)
    ctx = torch.enable_grad() if keep_graph else torch.no_grad()
    with ctx:
        logits = model(input_ids=x).logits[:, :-1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        tgt = torch.tensor([ids[1:]], dtype=torch.long)
        pick = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        m = torch.tensor([labels[i + 1] != -100 for i in range(len(ids) - 1)],
                         dtype=torch.bool)
        return pick[m].sum()


def _masked_nll(model, tok, rows, torch, max_len: int) -> float:
    total, ntok = 0.0, 0
    with torch.no_grad():
        for r in rows:
            ids, labels, n = build_labels(tok, r["messages"])
            ids, labels = ids[:max_len], labels[:max_len]
            n = sum(1 for x in labels if x != -100)
            if n == 0:
                continue
            out = model(input_ids=torch.tensor([ids], dtype=torch.long),
                        labels=torch.tensor([labels], dtype=torch.long))
            total += float(out.loss.detach()) * n
            ntok += n
    return total / max(ntok, 1)


def _generate(model, tok, rows, torch, max_new_tokens: int, tag: str,
              root: Path) -> dict:
    tok.padding_side = "left"
    prompts = [_apply_chat_template(tok, r["messages"][:2],
                                    add_generation_prompt=True) for r in rows]
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    cut = enc["input_ids"].shape[1]
    items = []
    for r, g in zip(rows, gen):
        text = tok.decode(g[cut:], skip_special_tokens=True)
        v = VERIFIERS["strict_poly"].verify(text, r["answer"])
        items.append({"id": r["id"], "subset": r["subset"], "family": r["family"],
                      "gold": r["answer"], "pred": text, "verdict": v.verdict,
                      "extracted": v.extracted, "reason": v.reason,
                      "pred_chars": len(text)})
    (root / "reports" / f"pilot_predictions_{tag}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in items) + "\n",
        encoding="utf-8")
    n = len(items)
    return {
        "correct_rate": sum(x["verdict"] == CORRECT for x in items) / n,
        "unverifiable_rate": sum(x["verdict"] == "unverifiable" for x in items) / n,
        "mean_pred_chars": sum(x["pred_chars"] for x in items) / n,
        "n": n,
    }


def _ref_logps(ref, tok, pairs, system, torch, F, max_len):
    out = []
    for p in pairs:
        msgs = _pair_messages(p, system)
        ci, cl, _ = _seq(tok, msgs, p["chosen"], max_len)
        ri, rl, _ = _seq(tok, msgs, p["rejected"], max_len)
        lc = float(_completion_logp(ref, ci, cl, torch, F, False))
        lr = float(_completion_logp(ref, ri, rl, torch, F, False))
        out.append((lc - lr, (ci, cl), (ri, rl)))
    return out


def _policy_margin(policy, tok, cached, torch, F, beta: float,
                    n_probe_pairs: int = 3) -> tuple[float, float]:
    """margin 只在固定的前几对上测, 避免每次评估都遍历全部偏好对。"""
    losses, margins = [], []
    for ref_diff, (ci, cl), (ri, rl) in cached[:n_probe_pairs]:
        lc = _completion_logp(policy, ci, cl, torch, F, False)
        lr = _completion_logp(policy, ri, rl, torch, F, False)
        d = (lc - lr) - ref_diff
        margins.append(beta * float(d))
        losses.append(-math.log(1.0 / (1.0 + math.exp(-beta * float(d)))) if beta * float(d) > -30
                      else -beta * float(d))
    return (sum(margins) / len(margins), sum(losses) / len(losses))


def _kl_mc(policy, ref, tok, prompts, torch, F, n_samples: int,
           max_new_tokens: int, seed: int, B: int = 1000) -> dict:
    rng = random.Random(seed + 5)
    torch.manual_seed(seed)
    vals = []
    policy.eval()
    ref.eval()
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt")
        plen = enc["input_ids"].shape[1]
        for _ in range(n_samples):
            with torch.no_grad():
                g = policy.generate(**enc, do_sample=True, top_p=0.95,
                                    temperature=1.0, max_new_tokens=max_new_tokens,
                                    pad_token_id=tok.eos_token_id)
            ids = g[0].tolist()
            labels = [-100] * plen + ids[plen:]
            with torch.no_grad():
                lp = float(_completion_logp(policy, ids, labels, torch, F, False))
                lr = float(_completion_logp(ref, ids, labels, torch, F, False))
            vals.append(lp - lr)
    mean = sum(vals) / max(len(vals), 1)
    boots = []
    for _ in range(B):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(s) / len(s))
    boots.sort()
    return {"kl": mean, "ci_low": boots[int(0.025 * B)],
            "ci_high": boots[min(B - 1, int(0.975 * B))], "n": len(vals)}


def run(root: Path, config_path: Path) -> dict:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    from src.model import check_deps
    from src.model.train_dpo import assert_ref_matches_policy

    check_deps()
    t0 = time.perf_counter()
    c = load_config(config_path)
    root = Path(root)
    seed = int(c["seed"])
    beta = float(c["beta"])
    max_len = int(c["max_seq_length"])
    out_dir = root / "runs" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    train_rows = _rows(root / "data" / "train.jsonl")
    test_rows = _rows(root / "data" / "test.jsonl")
    eval_rows = _stratified_eval(test_rows, int(c["n_eval_per_subset"]), seed)
    system = _system(train_rows)
    groups = ("shortcut", "deconfounded", "length_reversed")
    pairs = {g: _rows(root / "data" / f"preferences_{g}.jsonl")[:int(c["n_pairs_per_group"])]
             for g in groups}

    recs = []
    def record(metric, value, unit="", n=None, note=""):
        recs.append({"experiment": "model_pilot", "metric": metric,
                     "value": value, "unit": unit, "n": n, "seed": seed,
                     "ci_low": None, "ci_high": None, "status": "measured",
                     "note": note or NOTE, "source": "src/model/pilot.py"})

    _log(f"config: {config_path.name}, eval items={len(eval_rows)}, "
         f"pairs/group={len(pairs['deconfounded'])}, max_new_tokens={c['max_new_tokens']}")
    tok = AutoTokenizer.from_pretrained(c["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = _load_base(c["base_model"], torch)
    base.eval()
    ids0, lab0, n0 = build_labels(tok, train_rows[0]["messages"])
    assert n0 > 0
    record("mask_self_test_completion_tokens", n0, unit="tokens")
    _log(f"base loaded, mask self-test ok ({n0} completion tokens)")

    # ---- Base ----
    base_loss = _masked_nll(base, tok, eval_rows, torch, max_len)
    base_gen = _generate(base, tok, eval_rows, torch,
                         int(c["max_new_tokens"]), "base", root)
    record("test_loss_masked_base", base_loss, unit="nats", n=len(eval_rows))
    record("generative_accuracy_base", base_gen["correct_rate"], unit="rate", n=len(eval_rows))
    record("unverifiable_rate_base", base_gen["unverifiable_rate"], unit="rate", n=len(eval_rows))
    _log(f"base: loss={base_loss:.4f} acc={base_gen['correct_rate']:.4f} "
         f"unverifiable={base_gen['unverifiable_rate']:.4f}")

    # ---- SFT ----
    lora = LoraConfig(r=int(c["lora_r"]), lora_alpha=int(c["lora_alpha"]),
                      lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM")
    sft = get_peft_model(base, lora)
    opt = torch.optim.AdamW([p for p in sft.parameters() if p.requires_grad],
                           lr=float(c["sft_lr"]))
    sample_pool = train_rows[:int(c["n_train_items"])]
    sft_losses = []
    sft.train()
    for step in range(int(c["n_sft_steps"])):
        r = sample_pool[step % len(sample_pool)]
        ids, labels, n = build_labels(tok, r["messages"])
        ids, labels = ids[:max_len], labels[:max_len]
        if not any(x != -100 for x in labels):
            continue
        out = sft(input_ids=torch.tensor([ids], dtype=torch.long),
                  labels=torch.tensor([labels], dtype=torch.long))
        (out.loss).backward()
        torch.nn.utils.clip_grad_norm_([p for p in sft.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad()
        sft_losses.append(float(out.loss.detach()))
    sft_dir = out_dir / "sft"
    sft.save_pretrained(sft_dir)
    record("sft_steps", len(sft_losses), unit="steps")
    record("sft_loss_first", sft_losses[0], unit="nats")
    record("sft_loss_last", sft_losses[-1], unit="nats")
    sft.eval()
    sft_loss = _masked_nll(sft, tok, eval_rows, torch, max_len)
    sft_gen = _generate(sft, tok, eval_rows, torch,
                        int(c["max_new_tokens"]), "sft", root)
    record("test_loss_masked_sft", sft_loss, unit="nats", n=len(eval_rows))
    record("generative_accuracy_sft", sft_gen["correct_rate"], unit="rate", n=len(eval_rows))
    record("unverifiable_rate_sft", sft_gen["unverifiable_rate"], unit="rate", n=len(eval_rows))
    _log(f"sft: {len(sft_losses)} steps, loss {sft_losses[0]:.4f}->{sft_losses[-1]:.4f}, "
         f"test loss={sft_loss:.4f} acc={sft_gen['correct_rate']:.4f}")
    del sft, base, opt
    gc.collect()

    # ---- 显式 ref: 独立基座实例 + 逐参数断言, 然后缓存 ref log-ratio ----
    policy0 = PeftModel.from_pretrained(_load_base(c["base_model"], torch),
                                        sft_dir, is_trainable=True)
    ref = PeftModel.from_pretrained(_load_base(c["base_model"], torch),
                                    sft_dir, is_trainable=False)
    ref.requires_grad_(False)
    ref.eval()
    checked = assert_ref_matches_policy(policy0, ref)
    record("ref_assert_checked_parameters", checked["checked_parameters"], unit="params")
    record("ref_is_frozen_sft_adapter", 1, unit="bool")
    _log(f"ref assertion ok ({checked['checked_parameters']} shared parameters)")
    cached = {}
    for g in groups:
        cached[g] = _ref_logps(ref, tok, pairs[g], system, torch, F, max_len)
    _log("reference log-ratios cached for all three groups")
    del ref, policy0
    gc.collect()

    # ---- 三组 DPO ----
    for g in groups:
        policy = PeftModel.from_pretrained(_load_base(c["base_model"], torch),
                                           sft_dir, is_trainable=True)
        dopt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                 lr=float(c["dpo_lr"]))
        policy.eval()
        m0, l0 = _policy_margin(policy, tok, cached[g], torch, F, beta)
        policy.train()
        for step in range(int(c["n_dpo_steps"])):
            ref_diff, (ci, cl), (ri, rl) = cached[g][step % len(cached[g])]
            lc = _completion_logp(policy, ci, cl, torch, F, True)
            lr = _completion_logp(policy, ri, rl, torch, F, True)
            d = (lc - lr) - ref_diff
            t = beta * d
            loss = F.softplus(-t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            dopt.step()
            dopt.zero_grad()
        policy.eval()
        m1, l1 = _policy_margin(policy, tok, cached[g], torch, F, beta)
        record(f"dpo_loss_before_{g}", l0, unit="nats")
        record(f"dpo_loss_after_{g}", l1, unit="nats")
        record(f"reward_margin_before_{g}", m0, unit="value")
        record(f"reward_margin_after_{g}", m1, unit="value")
        record(f"reward_margin_gain_{g}", m1 - m0, unit="value")
        loss_g = _masked_nll(policy, tok, eval_rows, torch, max_len)
        gen_g = _generate(policy, tok, eval_rows, torch,
                          int(c["max_new_tokens"]), f"dpo_{g}", root)
        record(f"test_loss_masked_dpo_{g}", loss_g, unit="nats", n=len(eval_rows))
        record(f"generative_accuracy_dpo_{g}", gen_g["correct_rate"],
               unit="rate", n=len(eval_rows))
        record(f"unverifiable_rate_dpo_{g}", gen_g["unverifiable_rate"],
               unit="rate", n=len(eval_rows))
        record(f"mean_pred_chars_dpo_{g}", gen_g["mean_pred_chars"], unit="chars")
        policy.save_pretrained(out_dir / f"dpo_{g}")
        _log(f"{g}: margin {m0:.4f}->{m1:.4f}, loss {l0:.4f}->{l1:.4f}, "
             f"acc={gen_g['correct_rate']:.4f}, unverifiable={gen_g['unverifiable_rate']:.4f}")
        del policy, dopt
        gc.collect()

    # ---- 序列级 KL 的 MC 估计 (主组 deconfounded) ----
    main = PeftModel.from_pretrained(_load_base(c["base_model"], torch),
                                     out_dir / "dpo_deconfounded", is_trainable=False)
    ref2 = PeftModel.from_pretrained(_load_base(c["base_model"], torch),
                                     sft_dir, is_trainable=False)
    prompts = [_apply_chat_template(tok, r["messages"][:2], add_generation_prompt=True)
               for r in eval_rows[:int(c["n_kl_prompts"])]]
    kl = _kl_mc(main, ref2, tok, prompts, torch, F,
                int(c["n_kl_samples"]), int(c["kl_max_new_tokens"]), seed)
    record("sequence_kl_mc", kl["kl"], unit="nats", n=kl["n"])
    record("sequence_kl_n", kl["n"], unit="samples")
    recs.append({"experiment": "model_pilot", "metric": "sequence_kl_ci_low",
                 "value": kl["ci_low"], "unit": "nats", "n": kl["n"], "seed": seed,
                 "ci_low": None, "ci_high": None, "status": "measured",
                 "note": NOTE, "source": "src/model/pilot.py"})
    recs.append({"experiment": "model_pilot", "metric": "sequence_kl_ci_high",
                 "value": kl["ci_high"], "unit": "nats", "n": kl["n"], "seed": seed,
                 "ci_low": None, "ci_high": None, "status": "measured",
                 "note": NOTE, "source": "src/model/pilot.py"})
    _log(f"KL mc={kl['kl']:.4f} [{kl['ci_low']:.4f}, {kl['ci_high']:.4f}] n={kl['n']}")
    del main, ref2
    gc.collect()

    # ---- 规模、环境与产物 ----
    elapsed = time.perf_counter() - t0
    record("elapsed_seconds", round(elapsed, 2), unit="seconds")
    for k in ("n_sft_steps", "n_dpo_steps", "n_pairs_per_group", "n_eval_per_subset",
              "max_new_tokens", "beta", "lora_r"):
        record(f"config_{k}", c[k], unit="value" if not k.startswith("n_") else "count")
    merged = append_records(root, recs)
    manifest = {
        "note": NOTE,
        "base_model": c["base_model"],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
        "seed": seed,
        "config": c,
        "n_eval_items": len(eval_rows),
        "eval_subsets": sorted({r["subset"] for r in eval_rows}),
        "n_pilot_records": len(recs),
        "elapsed_seconds": round(elapsed, 2),
        "adapters": {p.name: hashlib.sha256(
            (p / "adapter_model.safetensors").read_bytes()).hexdigest()[:16]
            for p in sorted(out_dir.iterdir()) if p.is_dir()
            and (p / "adapter_model.safetensors").exists()},
        "formal_protocol_still_pending": True,
    }
    (root / "reports" / "model_pilot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path = root / "reports" / "experiment_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    render_all(root, meta)
    check = verify_consistency(root, meta)
    _log(f"done in {elapsed:.1f}s, {len(recs)} pilot records, "
         f"{len(merged)} total records, consistency={check['ok']}")
    return {"manifest": manifest, "n_records": len(recs), "consistency": check,
            "base": {"loss": base_loss, **base_gen},
            "sft": {"loss": sft_loss, **sft_gen}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_pilot.yaml")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = run(root, Path(a.config) if Path(a.config).is_absolute() else root / a.config)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("n_records", "base", "sft", "consistency")},
                     ensure_ascii=False, indent=2))
    return 0 if out["consistency"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
