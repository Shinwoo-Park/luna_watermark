from __future__ import annotations

import os
import gc
import json
import time
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dataset import load_wiki_dataset, load_dataset, WikiRecord
from experiments.model_config import (
    LANG_MODEL_CONFIGS, get_generation_kwargs,
    load_tokenizer, load_model, configure_model_for_tokenizer,
    build_input_ids, extract_generated_text,
    DEFAULT_TEMPERATURE,
)
from experiments.evaluation import (
    compute_detection_metrics, compute_quality_metrics,
    PPLEvaluator, PPL_REFERENCE_MODEL, aggregate_records,
)

def json_safe(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):

        f = float(obj)
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return json_safe(obj.item())
        return [json_safe(x) for x in obj.detach().cpu().tolist()]

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    return obj


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda_memory(verbose: bool = False, label: str = ""):

    if torch.cuda.is_available() and verbose:
        before_alloc = torch.cuda.memory_allocated() / (1024**3)
        before_reserved = torch.cuda.memory_reserved() / (1024**3)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if torch.cuda.is_available() and verbose:
        after_alloc = torch.cuda.memory_allocated() / (1024**3)
        after_reserved = torch.cuda.memory_reserved() / (1024**3)
        free_b, total_b = torch.cuda.mem_get_info()
        free_gib = free_b / (1024**3)
        total_gib = total_b / (1024**3)
        tag = f" [{label}]" if label else ""
        print(
            f"  GPU mem{tag}: alloc {before_alloc:.2f} → {after_alloc:.2f} GiB; "
            f"reserved {before_reserved:.2f} → {after_reserved:.2f} GiB; "
            f"system free {free_gib:.2f}/{total_gib:.2f} GiB",
            flush=True,
        )


def append_summary_csv(csv_path: str, row: Dict):

    import pandas as pd
    df_row = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode='a', header=False, index=False,
                      encoding='utf-8-sig', float_format='%.10f')
    else:
        df_row.to_csv(csv_path, mode='w', header=True, index=False,
                      encoding='utf-8-sig', float_format='%.10f')


def flatten_summary(summary: Dict, prefix: str = "") -> Dict:

    out = {}
    for k, v in summary.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_summary(v, prefix=f"{key}_"))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v

    return out


def build_watermarker(
    algorithm: str,
    config_path: str,
    transformers_config,
):
    from watermark.auto_watermark import AutoWatermark
    return AutoWatermark.load(
        algorithm_name=algorithm,
        algorithm_config=config_path,
        transformers_config=transformers_config,
    )


def _generate_with_input_ids(watermarker, inputs: dict, watermarked: bool) -> str:

    from transformers import LogitsProcessorList
    import torch

    config = watermarker.config
    gen_kwargs = config.gen_kwargs
    model = config.generation_model
    tokenizer = config.generation_tokenizer

    ctx = (torch.inference_mode()
           if hasattr(torch, "inference_mode") else torch.no_grad())

    if watermarked:
        has_lp = (hasattr(watermarker, "logits_processor")
                  or hasattr(watermarker, "logits_processor_template"))
        if not has_lp:
            decoded_prompt = tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=True
            )
            with ctx:
                return watermarker.generate_watermarked_text(decoded_prompt)

        if hasattr(watermarker, "logits_processor_template"):
            logits_processor = watermarker.logits_processor_template(config)
            processors = LogitsProcessorList([logits_processor])
        else:
            lp = watermarker.logits_processor
            if hasattr(lp, "reset_state"):
                lp.reset_state()
            else:
                if hasattr(lp, "state"):
                    lp.state = None
            processors = LogitsProcessorList([lp])

        with ctx:
            out_ids = model.generate(
                **inputs,
                logits_processor=processors,
                **gen_kwargs,
            )
    else:
        with ctx:
            out_ids = model.generate(
                **inputs,
                **gen_kwargs,
            )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

def _truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:

    if max_tokens <= 0 or not text:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def process_one_record(
    record: WikiRecord,
    watermarker,
    tokenizer,
    lang: str,
    device: str,
    ppl_evaluator: Optional[PPLEvaluator],
    text_tokenizer_for_metrics,
    algorithm: str,
    skip_quality: bool = False,
    max_detection_tokens: int = 256,
) -> Dict:

    import torch

    instruction = record.instruction
    prompt_text = record.prompt
    human_text = record.text
    user_content = f"{instruction}\n\n{prompt_text}".strip()

    inputs = build_input_ids(record.to_dict(), tokenizer, lang, device=device)
    encoded_prompt_str = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    t0 = time.perf_counter()
    wm_full = _generate_with_input_ids(watermarker, inputs, watermarked=True)
    wm_gen_time = time.perf_counter() - t0
    wm_text = extract_generated_text(wm_full, record.to_dict(), tokenizer, lang)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    uwm_full = _generate_with_input_ids(watermarker, inputs, watermarked=False)
    uwm_gen_time = time.perf_counter() - t0
    uwm_text = extract_generated_text(uwm_full, record.to_dict(), tokenizer, lang)

    import numpy as np

    def _normalize_det(d):
        if not isinstance(d, dict):
            return d
        out = {}
        for k, v in d.items():
            if isinstance(v, np.generic):           
                out[k] = v.item()
            elif isinstance(v, np.ndarray) and v.ndim == 0:
                out[k] = v.item()
            else:
                out[k] = v
        return out

    def _fix_polarity(d, algorithm):

        INVERTED_ALGOS = ()  
        if not isinstance(d, dict) or "score" not in d:
            return d
        if algorithm in INVERTED_ALGOS:
            try:
                d["score"] = 1.0 - float(d["score"])
            except (TypeError, ValueError):
                pass
        return d

    def _free_gpu():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _free_gpu()

    wm_text_eval  = _truncate_text_to_tokens(wm_text,  tokenizer, max_detection_tokens)
    uwm_text_eval = _truncate_text_to_tokens(uwm_text, tokenizer, max_detection_tokens)
    hum_text_eval = _truncate_text_to_tokens(human_text, tokenizer, max_detection_tokens)

    t0 = time.perf_counter()
    wm_det = _fix_polarity(
        _normalize_det(watermarker.detect_watermark(wm_text_eval, return_dict=True)),
        algorithm,
    )
    wm_det_time = time.perf_counter() - t0

    _free_gpu()
    t0 = time.perf_counter()
    uwm_det = _fix_polarity(
        _normalize_det(watermarker.detect_watermark(uwm_text_eval, return_dict=True)),
        algorithm,
    )
    uwm_det_time = time.perf_counter() - t0

    _free_gpu()
    t0 = time.perf_counter()
    hum_det = _fix_polarity(
        _normalize_det(watermarker.detect_watermark(hum_text_eval, return_dict=True)),
        algorithm,
    )
    hum_det_time = time.perf_counter() - t0
    _free_gpu()

    if skip_quality:
        wm_q = uwm_q = hum_q = {}
    else:
        wm_q  = compute_quality_metrics(wm_text_eval,  ppl_evaluator, text_tokenizer_for_metrics)
        uwm_q = compute_quality_metrics(uwm_text_eval, ppl_evaluator, text_tokenizer_for_metrics)
        hum_q = compute_quality_metrics(hum_text_eval, ppl_evaluator, text_tokenizer_for_metrics)
    out = {
        "meta": {
            "lang": record.lang,
            "title": record.title,
            "has_instruction": bool(record.instruction.strip()),
            "instruction_len": len(record.instruction),
            "prompt_len": len(record.prompt),
            "human_text_len": len(record.text),
        },
        "instruction": record.instruction,
        "prompt": record.prompt,
        "human_text": record.text,
        "watermarked_text": wm_text,
        "unwatermarked_text": uwm_text,
        "watermarked_score": wm_det.get("score"),
        "watermarked_is_watermarked": wm_det.get("is_watermarked"),
        "unwatermarked_score": uwm_det.get("score"),
        "unwatermarked_is_watermarked": uwm_det.get("is_watermarked"),
        "human_score": hum_det.get("score"),
        "human_is_watermarked": hum_det.get("is_watermarked"),
        "watermarked_detection_extras": {
            k: v for k, v in wm_det.items()
            if k not in ("score", "is_watermarked")
        },
        "watermarked_generation_time": wm_gen_time,
        "unwatermarked_generation_time": uwm_gen_time,
        "watermarked_detection_time": wm_det_time,
        "unwatermarked_detection_time": uwm_det_time,
        "human_detection_time": hum_det_time,
    }
    for src, q in (("watermarked", wm_q), ("unwatermarked", uwm_q), ("human", hum_q)):
        for k, v in q.items():
            if isinstance(v, list):
                continue
            out[f"{src}_quality_{k}"] = v

    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True,
                   help="LUNA")
    p.add_argument("--config", default=None,
                   help="Path to algorithm config JSON. "
                        "Default: config/{algorithm}.json (or config/{algorithm}_{lang}_k{k}.json "
                        "if it exists)")
    p.add_argument("--lang", required=True,
                   choices=("en", "zh", "ko", "ja", "de", "ar"))
    p.add_argument("--k-primary", type=int, default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset", default="wiki", choices=("wiki", "news"))
    p.add_argument("--output-dir", default="results")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--skip", type=int, default=0,
                   help="Skip first N records.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None,
                   help="cuda | cpu | auto. Default: auto.")
    p.add_argument("--skip-ppl", action="store_true",
                   help="Skip PPL/quality computation (debug mode).")
    p.add_argument("--ppl-model", default=None,
                   help=f"Override PPL reference model. Default: {PPL_REFERENCE_MODEL}.")
    p.add_argument("--ppl-device", default=None,
                   choices=["cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + dataset paths without loading models.")
    p.add_argument("--gpu-mem-verbose", action="store_true",
                   help="Print GPU memory at key cleanup points (startup, "
                        "post-LM-load, post-watermarker, pre-PPL, end). "
                        "Useful to diagnose OOM patterns.")
    p.add_argument("--max-detection-tokens", type=int, default=256)
    args = p.parse_args()

    set_seed(args.seed)

    clear_cuda_memory(verbose=args.gpu_mem_verbose, label="startup")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    DEFAULT_K = {"en": 2, "ko": 4, "zh": 4, "ja": 4, "de": 3, "ar": 4}
    if args.config is not None:
        config_path = args.config
    else:
        k = args.k_primary or DEFAULT_K[args.lang]
        candidate_lang_k = f"config/{args.algorithm}_{args.lang}_k{k}.json"
        candidate_plain = f"config/{args.algorithm}.json"
        if os.path.exists(candidate_lang_k):
            config_path = candidate_lang_k
        elif os.path.exists(candidate_plain):
            config_path = candidate_plain
        else:
            raise FileNotFoundError(
                f"No config found. Tried:\n  {candidate_lang_k}\n  {candidate_plain}"
            )

    dataset_path = os.path.join(args.data_dir, f"{args.dataset}_{args.lang}.jsonl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    model_key = LANG_MODEL_CONFIGS[args.lang].model_id.split("/")[-1]
    stem = f"{args.lang}_{args.dataset}_{config_stem}_{model_key}_T{args.temperature}"

    print(f"=" * 70)
    print(f"  Watermark experiment")
    print(f"=" * 70)
    print(f"  Algorithm    : {args.algorithm}")
    print(f"  Config       : {config_path}")
    print(f"  Language     : {args.lang}")
    print(f"  Model        : {LANG_MODEL_CONFIGS[args.lang].model_id}")
    print(f"  Dataset      : {dataset_path} (family={args.dataset})")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Num samples  : {args.num_samples or 'all'}")
    print(f"  Skip         : {args.skip}")
    print(f"  Device       : {device}")
    print(f"  Seed         : {args.seed}")
    print(f"  Output stem  : {stem}")

    if args.dry_run:
        recs = load_dataset(args.data_dir, args.lang, dataset=args.dataset,
                            n=1, skip=args.skip)
        print(f"  ✓ Dataset readable; first record title: {recs[0].title!r}")
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"  ✓ Config readable; algorithm_name: {cfg.get('algorithm_name')}")
        print(f"  Dry run complete.")
        return

    records = load_dataset(
        args.data_dir, args.lang, dataset=args.dataset,
        n=args.num_samples, skip=args.skip,
    )
    print(f"  Loaded {len(records)} records", flush=True)

    import time as _time
    t_stage = _time.time()

    print(f"\n[1/3] Loading generation model + tokenizer "
          f"({LANG_MODEL_CONFIGS[args.lang].model_id}) ...", flush=True)
    clear_cuda_memory(verbose=args.gpu_mem_verbose, label="pre-LM-load")
    tokenizer = load_tokenizer(args.lang)
    print(f"      tokenizer loaded ({_time.time() - t_stage:.1f}s)", flush=True)
    t_step = _time.time()
    gen_model = load_model(args.lang, device=device)
    configure_model_for_tokenizer(gen_model, tokenizer)
    print(f"      model loaded     ({_time.time() - t_step:.1f}s, "
          f"stage total {_time.time() - t_stage:.1f}s)", flush=True)
    clear_cuda_memory(verbose=args.gpu_mem_verbose, label="post-LM-load")
    t_stage = _time.time()

    print(f"[2/3] Building watermarker ({args.algorithm}) ...", flush=True)
    from utils.transformers_config import TransformersConfig
    gen_kwargs = get_generation_kwargs(args.lang, temperature=args.temperature)

    KNOWN_TC_ARGS = {
        "max_new_tokens", "temperature", "do_sample", "min_length",
        "no_repeat_ngram_size", "vocab_size",
    }
    tc_explicit = {k: v for k, v in gen_kwargs.items() if k in KNOWN_TC_ARGS}
    tc_extra    = {k: v for k, v in gen_kwargs.items() if k not in KNOWN_TC_ARGS}

    transformers_config = TransformersConfig(
        model=gen_model, tokenizer=tokenizer, device=device,
        **tc_explicit,
    )

    if tc_extra:
        if not hasattr(transformers_config, "gen_kwargs") or transformers_config.gen_kwargs is None:
            transformers_config.gen_kwargs = {}
        transformers_config.gen_kwargs.update(tc_extra)

    watermarker = build_watermarker(args.algorithm, config_path, transformers_config)
    print(f"      watermarker ready ({_time.time() - t_stage:.1f}s)", flush=True)
    clear_cuda_memory(verbose=args.gpu_mem_verbose, label="post-watermarker")
    t_stage = _time.time()

    ppl_target_device = None  
    if args.skip_ppl:
        ppl_evaluator = None
        print(f"[3/3] Skipping PPL reference (--skip-ppl)", flush=True)
    else:
        ppl_target_device = args.ppl_device or device
        print(f"[3/3] Loading PPL reference model "
              f"({args.ppl_model or PPL_REFERENCE_MODEL}) on cpu (will move to "
              f"{ppl_target_device} during phase 2) ...",
              flush=True)
        ppl_evaluator = PPLEvaluator(
            model_name=args.ppl_model or PPL_REFERENCE_MODEL,
            device="cpu",
        )
        print(f"      PPL evaluator ready on cpu ({_time.time() - t_stage:.1f}s)",
              flush=True)
        clear_cuda_memory(verbose=args.gpu_mem_verbose, label="post-PPL-load")
    t_loop_start = _time.time()
    print(f"\n→ Starting per-record loop ({len(records)} record"
          f"{'s' if len(records) != 1 else ''}) ...", flush=True)

    all_records = []
    pending_quality = []   
    pbar = tqdm(
        records,
        desc=f"  [P1 gen+det] {args.algorithm}/{args.lang}",

        file=sys.stdout,
        mininterval=0.0,
        miniters=1,
        ascii=False,
    )
    for record in pbar:
        try:
            rec_out = process_one_record(
                record=record,
                watermarker=watermarker,
                tokenizer=tokenizer,
                lang=args.lang,
                device=device,
                ppl_evaluator=None,        
                text_tokenizer_for_metrics=tokenizer,
                algorithm=args.algorithm,  
                skip_quality=True,         
                max_detection_tokens=args.max_detection_tokens,
            )
            rec_out["meta"]["temperature"] = args.temperature
            rec_out["meta"]["algorithm"] = args.algorithm
            rec_out["meta"]["dataset"] = args.dataset
            rec_out["meta"]["config_path"] = config_path
            rec_out["meta"]["max_detection_tokens"] = args.max_detection_tokens
            all_records.append(rec_out)

            pending_quality.append({
                "rec_idx": len(all_records) - 1,
                "wm_text": _truncate_text_to_tokens(
                    rec_out.get("watermarked_text", ""),
                    tokenizer, args.max_detection_tokens),
                "uwm_text": _truncate_text_to_tokens(
                    rec_out.get("unwatermarked_text", ""),
                    tokenizer, args.max_detection_tokens),
                "human_text": _truncate_text_to_tokens(
                    record.text, tokenizer, args.max_detection_tokens),
            })
        except Exception as e:

            import traceback
            tb = traceback.format_exc()
            err_line = f"  WARN: record failed ({record.title}): {type(e).__name__}: {e}"
            print(err_line, file=sys.stderr, flush=True)
            print(err_line, flush=True)

            tb_lines = [ln for ln in tb.strip().split("\n") if ln.strip()]
            for ln in tb_lines[-3:]:
                print(f"      {ln}", file=sys.stderr, flush=True)

            if "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower():
                clear_cuda_memory(verbose=args.gpu_mem_verbose, label="post-OOM-recovery")
            continue
        finally:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not args.skip_ppl and pending_quality:

        print(f"\n[P2 prep] Offloading gen_model to CPU before PPL phase ...",
              flush=True)
        t_off = _time.time()
        try:
            gen_model.cpu()
        except Exception as e:

            print(f"      gen_model.cpu() failed ({e!r}), falling back to del",
                  flush=True)
            del watermarker
            del gen_model
        clear_cuda_memory(verbose=args.gpu_mem_verbose, label="post-gen-offload")
        print(f"      offload + cache clear done ({_time.time() - t_off:.1f}s)",
              flush=True)


        if ppl_target_device and ppl_target_device != "cpu":
            t_move = _time.time()
            print(f"[P2 prep] Moving PPL model cpu → {ppl_target_device} ...",
                  flush=True)
            try:
                ppl_evaluator.to_device(ppl_target_device)
                print(f"      PPL on {ppl_target_device} "
                      f"({_time.time() - t_move:.1f}s)",
                      flush=True)
            except Exception as e:

                print(f"      PPL → {ppl_target_device} failed ({e!r}); "
                      f"continuing with PPL on CPU",
                      flush=True)
                ppl_evaluator.to_device("cpu")
            clear_cuda_memory(verbose=args.gpu_mem_verbose, label="post-PPL-onto-GPU")


        from experiments.evaluation.quality_metrics import compute_quality_metrics
        pbar2 = tqdm(
            pending_quality,
            desc=f"  [P2 PPL+qual] {args.algorithm}/{args.lang}",
            file=sys.stdout,
            mininterval=0.0,
            miniters=1,
            ascii=False,
        )
        for entry in pbar2:
            ridx = entry["rec_idx"]
            try:
                wm_q  = compute_quality_metrics(entry["wm_text"],  ppl_evaluator, tokenizer)
                uwm_q = compute_quality_metrics(entry["uwm_text"], ppl_evaluator, tokenizer)
                hum_q = compute_quality_metrics(entry["human_text"], ppl_evaluator, tokenizer)

                rec = all_records[ridx]
                for src, q in (("watermarked", wm_q), ("unwatermarked", uwm_q),
                               ("human", hum_q)):
                    for k, v in q.items():
                        if isinstance(v, list):
                            continue
                        rec[f"{src}_quality_{k}"] = v
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                err_line = (f"  WARN: PPL/quality failed for record {ridx}: "
                            f"{type(e).__name__}: {e}")
                print(err_line, file=sys.stderr, flush=True)
                print(err_line, flush=True)
                tb_lines = [ln for ln in tb.strip().split("\n") if ln.strip()]
                for ln in tb_lines[-3:]:
                    print(f"      {ln}", file=sys.stderr, flush=True)
                if "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower():
                    clear_cuda_memory(verbose=args.gpu_mem_verbose, label="post-PPL-OOM")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    elif args.skip_ppl:

        from experiments.evaluation.quality_metrics import compute_quality_metrics
        for entry in pending_quality:
            ridx = entry["rec_idx"]
            try:
                wm_q  = compute_quality_metrics(entry["wm_text"],  None, tokenizer)
                uwm_q = compute_quality_metrics(entry["uwm_text"], None, tokenizer)
                hum_q = compute_quality_metrics(entry["human_text"], None, tokenizer)
                rec = all_records[ridx]
                for src, q in (("watermarked", wm_q), ("unwatermarked", uwm_q),
                               ("human", hum_q)):
                    for k, v in q.items():
                        if isinstance(v, list):
                            continue
                        rec[f"{src}_quality_{k}"] = v
            except Exception as e:
                print(f"  WARN: text-quality failed for record {ridx}: {e}",
                      flush=True)


    records_path = os.path.join(args.output_dir, f"records_{stem}.jsonl")
    with open(records_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(json_safe(r), ensure_ascii=False) + "\n")
    print(f"\n  Records saved → {records_path}")


    if not all_records:
        msg = (
            f"\n  ✗ All {len(records)} records failed for "
            f"{args.algorithm} × {args.lang}. No summary computed.\n"
            f"    See record-level errors above for the root cause."
        )
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        sys.exit(2)

    summary = aggregate_records(all_records, setup_label=stem)
    summary["meta"] = {
        "algorithm": args.algorithm,
        "lang": args.lang,
        "dataset": args.dataset,
        "model": LANG_MODEL_CONFIGS[args.lang].model_id,
        "temperature": args.temperature,
        "num_samples": len(all_records),
        "config_path": config_path,
        "ppl_model": args.ppl_model or PPL_REFERENCE_MODEL if not args.skip_ppl else None,
    }
    summary_path = os.path.join(args.output_dir, f"summary_{stem}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False, default=str)
    print(f"  Summary saved → {summary_path}")


    flat = flatten_summary(summary)
    csv_path = os.path.join(args.output_dir, "summary_all.csv")
    append_summary_csv(csv_path, flat)
    print(f"  Appended to    → {csv_path}")

    print()
    print("=" * 70)
    setup_a = summary.get("detection_vs_unwatermarked", {})
    setup_b = summary.get("detection_vs_human", {})
    if setup_a:
        print(f"  Detection (vs unwatermarked):")
        print(f"    AUROC       : {setup_a.get('auroc', float('nan')):.4f}")
        print(f"    TPR@FPR=1%  : {setup_a.get('tpr_at_1_fpr', float('nan')):.4f}")
        print(f"    TPR@FPR=5%  : {setup_a.get('tpr_at_5_fpr', float('nan')):.4f}")
        print(f"    EER         : {setup_a.get('eer', float('nan')):.4f}")
        print(f"    Best F1     : {setup_a.get('best_f1', float('nan')):.4f}")
    if setup_b:
        print(f"  Detection (vs human-written):")
        print(f"    AUROC       : {setup_b.get('auroc', float('nan')):.4f}")
        print(f"    TPR@FPR=1%  : {setup_b.get('tpr_at_1_fpr', float('nan')):.4f}")
        print(f"    TPR@FPR=5%  : {setup_b.get('tpr_at_5_fpr', float('nan')):.4f}")
        print(f"    EER         : {setup_b.get('eer', float('nan')):.4f}")
    if not args.skip_ppl:
        print(f"  Quality:")
        print(f"    PPL (wm)    : {summary.get('avg_watermarked_quality_ppl'):.3f}")
        print(f"    PPL (uwm)   : {summary.get('avg_unwatermarked_quality_ppl'):.3f}")
        print(f"    PPL (human) : {summary.get('avg_human_quality_ppl'):.3f}")
        print(f"    Δ PPL       : {summary.get('avg_ppl_delta'):.3f}")
    print(f"  Timing:")
    print(f"    Gen wm      : {summary.get('avg_generation_time_sec'):.3f} s/record")
    print(f"    Gen uwm     : {summary.get('avg_unwm_generation_time_sec'):.3f} s/record")
    print(f"    Detection   : {summary.get('avg_detection_time_sec'):.3f} s/record")
    print("=" * 70)

    try: del watermarker
    except NameError: pass
    try: del gen_model
    except NameError: pass
    try: del tokenizer
    except NameError: pass
    try: del ppl_evaluator
    except NameError: pass
    clear_cuda_memory(verbose=args.gpu_mem_verbose, label="exit")


if __name__ == "__main__":
    main()
