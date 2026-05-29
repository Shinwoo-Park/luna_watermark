from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_TEMPERATURE   = 0.7
ABLATION_TEMPERATURES = [0.5, 0.7, 1.0]

MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 200   

BASE_GENERATION_KWARGS: dict = dict(
    do_sample          = True,
    temperature        = DEFAULT_TEMPERATURE,  
    top_p              = 0.95,  
    top_k              = 0,       
    max_new_tokens     = MAX_NEW_TOKENS,
    min_new_tokens     = MIN_NEW_TOKENS,
    repetition_penalty = 1.0,
)

SMALL_MODEL_OVERRIDE: dict = dict(
    repetition_penalty = 1.1,
)

@dataclass
class LangModelConfig:
    lang:               str
    model_id:           str
    size_b:             float       
    is_small:           bool        
    pad_side:           str  = "left"
    pad_as_eos:         bool = False
    gen_override:       dict = field(default_factory=dict)
    system_prompt:      str  = ""   
    use_chat_template:  bool = True
    chat_template_kwargs: dict = field(default_factory=dict)
    revision:           Optional[str] = None
    notes:              str  = ""

LANG_MODEL_CONFIGS: Dict[str, LangModelConfig] = {

    # ── English: Llama-3.2-1B-Instruct ────────────────────────────────────
    "en": LangModelConfig(
        lang      = "en",
        model_id  = "meta-llama/Llama-3.2-1B-Instruct",
        size_b    = 1.0,
        is_small  = False,
        pad_side  = "left",
        pad_as_eos= True,
        gen_override = dict(),
        system_prompt = "You are a helpful writing assistant. Follow the user's instruction precisely.",
    ),

    # ── Korean: EXAONE-3.5-2.4B-Instruct ──────────────────────────────────
    "ko": LangModelConfig(
        lang      = "ko",
        model_id  = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        revision  = "8e6fc27",  
        size_b    = 2.4,
        is_small  = False,  
        pad_side  = "left",
        pad_as_eos= True,    
        gen_override = dict(),
        system_prompt = "You are EXAONE model from LG AI Research, a helpful assistant.",
        use_chat_template = True,
        chat_template_kwargs = {},
    ),

    # ── Chinese: Qwen2.5-0.5B-Instruct ────────────────────────────────────
    "zh": LangModelConfig(
        lang      = "zh",
        model_id  = "Qwen/Qwen2.5-0.5B-Instruct",
        size_b    = 0.5,
        is_small  = True,    
        pad_side  = "left",
        pad_as_eos= False,   
        gen_override = dict(),
        system_prompt = "你是一位专业的写作助手，请准确地按照用户的指示完成任务。",
        use_chat_template = True,
        chat_template_kwargs = {},  
    ),

    "ja": LangModelConfig(
        lang      = "ja",
        model_id  = "sbintuitions/sarashina2.2-3b-instruct-v0.1",
        size_b    = 3.0,
        is_small  = False,
        pad_side  = "left",
        pad_as_eos= True,
        gen_override = dict(),
        system_prompt = "あなたは優秀なライティングアシスタントです。ユーザーの指示に正確に従ってください。",
    ),

    "de": LangModelConfig(
        lang      = "de",
        model_id  = "utter-project/EuroLLM-1.7B-Instruct",
        size_b    = 1.7,
        is_small  = False,
        pad_side  = "left",
        pad_as_eos= True,
        gen_override = dict(),
        system_prompt = "Du bist ein hilfreicher Schreibassistent. Befolge die Anweisungen des Benutzers genau.",
    ),

    "ar": LangModelConfig(
        lang      = "ar",
        model_id  = "inceptionai/jais-family-1p3b-chat",
        size_b    = 1.3,
        is_small  = False,
        pad_side  = "left",
        pad_as_eos= False,  
        gen_override = dict(),
        system_prompt = "أنت مساعد كتابة محترف. اتبع تعليمات المستخدم بدقة.",
        use_chat_template = True,
    ),
}

def get_generation_kwargs(
    lang: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:

    cfg    = LANG_MODEL_CONFIGS[lang]
    kwargs = {**BASE_GENERATION_KWARGS}
    if cfg.is_small:
        kwargs.update(SMALL_MODEL_OVERRIDE)
    if cfg.gen_override:
        kwargs.update(cfg.gen_override)
    kwargs["temperature"] = temperature
    return kwargs

def load_tokenizer(lang: str) -> AutoTokenizer:
    cfg = LANG_MODEL_CONFIGS[lang]
    kwargs = dict(
        trust_remote_code=True,
        padding_side=cfg.pad_side,
    )
    if cfg.revision is not None:
        kwargs["revision"] = cfg.revision
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **kwargs)
    if tokenizer.pad_token is None:
        if cfg.pad_as_eos:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_model(lang: str, device: str = "cuda") -> AutoModelForCausalLM:
    cfg   = LANG_MODEL_CONFIGS[lang]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = dict(
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    if cfg.revision is not None:
        kwargs["revision"] = cfg.revision
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    return model


def configure_model_for_tokenizer(model, tokenizer):
    if tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        gen_eos = model.generation_config.eos_token_id
        if not isinstance(gen_eos, list):
            model.generation_config.eos_token_id = tokenizer.eos_token_id

def build_input_ids(
    record: dict,
    tokenizer: AutoTokenizer,
    lang: str,
    device: str = "cuda",
) -> dict:

    cfg          = LANG_MODEL_CONFIGS[lang]
    instruction  = record.get("instruction", "")
    prompt_text  = record.get("prompt", "")
    user_content = f"{instruction}\n\n{prompt_text}".strip()

    if cfg.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages: List[dict] = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": user_content})

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **cfg.chat_template_kwargs,
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)
            return inputs
        except Exception:
            pass

    text   = user_content
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return inputs

def extract_generated_text(
    full_decoded_text: str,
    record: dict,
    tokenizer: AutoTokenizer,
    lang: str,
) -> str:

    text = full_decoded_text or ""
    prompt_text = record.get("prompt", "")
    instruction = record.get("instruction", "")

    if prompt_text and prompt_text in text:
        idx = text.rfind(prompt_text)
        return text[idx + len(prompt_text):].strip()

    if prompt_text and len(prompt_text) >= 80:
        prompt_tail = prompt_text[-80:]
        if prompt_tail in text:
            idx = text.rfind(prompt_tail)
            return text[idx + len(prompt_tail):].strip()

    if instruction and len(instruction) >= 60:
        ins_tail = instruction[-60:]
        if ins_tail in text:
            idx = text.rfind(ins_tail)
            tail_text = text[idx + len(ins_tail):]
            if prompt_text and prompt_text in tail_text:
                p_idx = tail_text.rfind(prompt_text)
                return tail_text[p_idx + len(prompt_text):].strip()
            return tail_text.strip()

    markers = [
        "<|im_start|>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "[/INST]",
        "<|assistant|>",
    ]
    for marker in markers:
        if marker in text:
            return text.split(marker, 1)[1].strip()

    return text.strip()
