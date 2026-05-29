from __future__ import annotations

from typing import Callable, Dict, List, Optional
import numpy as np

from .detection_metrics import compute_detection_metrics


def _safe_mean(vals: List[float]) -> float:
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float('nan')


def _safe_median(vals: List[float]) -> float:
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else float('nan')


def aggregate_records(
    records: List[dict],
    filter_fn: Optional[Callable[[dict], bool]] = None,
    setup_label: str = "all",
) -> Dict:

    if filter_fn is not None:
        records = [r for r in records if filter_fn(r)]
    n = len(records)
    out: Dict = {"setup_label": setup_label, "num_records": n}
    if n == 0:
        return out

    pos = [r["watermarked_score"] for r in records
           if r.get("watermarked_score") is not None]
    neg_uwm = [r["unwatermarked_score"] for r in records
               if r.get("unwatermarked_score") is not None]
    neg_hum = [r["human_score"] for r in records
               if r.get("human_score") is not None]

    setup_a = compute_detection_metrics(pos, neg_uwm)
    setup_b = compute_detection_metrics(pos, neg_hum)

    out["detection_vs_unwatermarked"] = setup_a
    out["detection_vs_human"] = setup_b

    for source in ("watermarked", "unwatermarked", "human"):
        prefix = f"{source}_quality"
        for key in ("ppl", "mean_entropy_nats", "mean_surprisal_nats",
                    "self_bleu", "distinct_1", "distinct_2", "distinct_3",
                    "char_count", "token_count", "sentence_count"):
            full_key = f"{prefix}_{key}"
            vals = [r.get(full_key) for r in records]
            out[f"avg_{full_key}"] = _safe_mean([v for v in vals if v is not None])
            out[f"med_{full_key}"] = _safe_median([v for v in vals if v is not None])

    ppl_w = [r.get("watermarked_quality_ppl") for r in records]
    ppl_u = [r.get("unwatermarked_quality_ppl") for r in records]
    deltas = []
    for w, u in zip(ppl_w, ppl_u):
        if (w is not None and u is not None
                and np.isfinite(w) and np.isfinite(u)):
            deltas.append(w - u)
    out["avg_ppl_delta"] = _safe_mean(deltas) if deltas else float('nan')
    out["med_ppl_delta"] = _safe_median(deltas) if deltas else float('nan')

    out["avg_generation_time_sec"] = _safe_mean(
        [r.get("watermarked_generation_time") for r in records])
    out["avg_unwm_generation_time_sec"] = _safe_mean(
        [r.get("unwatermarked_generation_time") for r in records])
    out["avg_detection_time_sec"] = _safe_mean(
        [r.get("watermarked_detection_time") for r in records])

    return out

def length_bucketed(records: List[dict], buckets=(50, 100, 150, 200, 250)) -> Dict:

    out = {}
    for i, hi in enumerate(buckets):
        lo = 0 if i == 0 else buckets[i - 1]
        label = f"len_{lo}_{hi}"
        out[label] = aggregate_records(
            records,
            filter_fn=lambda r, lo=lo, hi=hi: (
                lo <= (r.get("watermarked_quality_token_count") or 0) < hi
            ),
            setup_label=label,
        )
    out[f"len_{buckets[-1]}_inf"] = aggregate_records(
        records,
        filter_fn=lambda r: (r.get("watermarked_quality_token_count") or 0) >= buckets[-1],
        setup_label=f"len_{buckets[-1]}_inf",
    )
    return out


def by_instruction_presence(records: List[dict]) -> Dict:
    return {
        "with_instruction": aggregate_records(
            records,
            filter_fn=lambda r: bool(r.get("meta", {}).get("has_instruction")),
            setup_label="with_instruction",
        ),
        "without_instruction": aggregate_records(
            records,
            filter_fn=lambda r: not bool(r.get("meta", {}).get("has_instruction")),
            setup_label="without_instruction",
        ),
    }


def by_temperature(records: List[dict]) -> Dict:
    temps = sorted(set(
        float(r.get("meta", {}).get("temperature"))
        for r in records
        if r.get("meta", {}).get("temperature") is not None
    ))
    out = {}
    for t in temps:
        out[f"temperature_{t}"] = aggregate_records(
            records,
            filter_fn=lambda r, t=t: float(r.get("meta", {}).get("temperature", -1)) == t,
            setup_label=f"temperature_{t}",
        )
    return out
