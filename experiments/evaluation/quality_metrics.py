from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

PPL_REFERENCE_MODEL = "Qwen/Qwen2.5-1.5B"

class PPLEvaluator:


    def __init__(
        self,
        model_name: str = PPL_REFERENCE_MODEL,
        device: Optional[str] = None,
        max_len: int = 1024,
        revision: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        load_kwargs = dict(trust_remote_code=True)
        if revision is not None:
            load_kwargs["revision"] = revision

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, **load_kwargs,
            ).to(self.device)
        except RuntimeError:
            # CUDA OOM fallback: CPU
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, **load_kwargs,
            ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    def to_device(self, device: str):
        self.model = self.model.to(device)
        self.device = device
        return self

    def compute(self, text: str) -> Dict:
        if not text:
            return {
                "ppl": float('inf'), "mean_entropy": None, "mean_surprisal": None,
                "num_tokens": 0, "token_entropies": [], "token_surprisals": [],
            }
        enc = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
            max_length=self.max_len, truncation=True,
        ).to(self.device)
        input_ids = enc["input_ids"]
        num_tokens = int(input_ids.shape[1])
        if num_tokens < 2:
            return {
                "ppl": float('inf'), "mean_entropy": None, "mean_surprisal": None,
                "num_tokens": num_tokens, "token_entropies": [None] * num_tokens,
                "token_surprisals": [None] * num_tokens,
            }

        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            logits = self.model(input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        probs = torch.exp(log_probs)
        ent = (-probs * log_probs).sum(dim=-1).squeeze(0)
        target_ids = input_ids[:, 1:]
        token_logp = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1),
        ).squeeze(-1).squeeze(0)
        surprisal = -token_logp

        mean_ent = float(ent.mean().item())
        mean_surp = float(surprisal.mean().item())
        ppl = float(math.exp(mean_surp))

        return {
            "ppl": ppl,
            "mean_entropy": mean_ent,
            "mean_surprisal": mean_surp,
            "num_tokens": num_tokens,
            "token_entropies": [None] + ent.detach().cpu().tolist(),
            "token_surprisals": [None] + surprisal.detach().cpu().tolist(),
        }


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(text: str, n: int = 1, tokenizer=None) -> float:

    if tokenizer is not None:
        try:
            tokens = tokenizer.tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return float(len(set(grams)) / len(grams))


def self_bleu(text: str, n_max: int = 4, tokenizer=None) -> float:

    if tokenizer is not None:
        try:
            tokens = tokenizer.tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()
    if len(tokens) < n_max:
        return 0.0
    repeats = 0.0
    counted = 0
    for n in range(1, n_max + 1):
        grams = _ngrams(tokens, n)
        if not grams:
            continue
        unique = len(set(grams))
        repeated_frac = 1.0 - (unique / len(grams))
        repeats += repeated_frac
        counted += 1
    return float(repeats / counted) if counted > 0 else 0.0


def length_metrics(text: str, tokenizer=None) -> Dict:

    char_count = len(text)
    if tokenizer is not None:
        try:
            tokens = tokenizer.tokenize(text)
            token_count = len(tokens)
        except Exception:
            token_count = len(text.split())
    else:
        token_count = len(text.split())
    sentence_marks = ".!?。！？؟"  
    sent_count = sum(1 for c in text if c in sentence_marks)
    return {
        "char_count": int(char_count),
        "token_count": int(token_count),
        "sentence_count": int(sent_count),
    }

def compute_quality_metrics(
    text: str,
    ppl_evaluator: Optional[PPLEvaluator] = None,
    text_tokenizer=None,
) -> Dict:

    out = {
        "ppl": float('nan'),
        "mean_entropy_nats": float('nan'),
        "mean_surprisal_nats": float('nan'),
        "num_tokens_ppl_ref": 0,
    }
    if ppl_evaluator is not None and text:
        ppl_result = ppl_evaluator.compute(text)
        out["ppl"] = ppl_result["ppl"]
        out["mean_entropy_nats"] = (
            ppl_result["mean_entropy"]
            if ppl_result["mean_entropy"] is not None
            else float('nan')
        )
        out["mean_surprisal_nats"] = (
            ppl_result["mean_surprisal"]
            if ppl_result["mean_surprisal"] is not None
            else float('nan')
        )
        out["num_tokens_ppl_ref"] = ppl_result["num_tokens"]

    out["self_bleu"] = self_bleu(text, n_max=4, tokenizer=text_tokenizer) if text else float('nan')
    out["distinct_1"] = distinct_n(text, 1, tokenizer=text_tokenizer) if text else float('nan')
    out["distinct_2"] = distinct_n(text, 2, tokenizer=text_tokenizer) if text else float('nan')
    out["distinct_3"] = distinct_n(text, 3, tokenizer=text_tokenizer) if text else float('nan')

    out.update(length_metrics(text, tokenizer=text_tokenizer))
    return out
