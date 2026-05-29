from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional


SUPPORTED_LANGS = ("en", "zh", "ko", "ja", "de", "ar")
SUPPORTED_DATASETS = ("wiki", "news")


@dataclass
class WikiRecord:

    lang: str
    title: str
    instruction: str
    prompt: str
    text: str

    @classmethod
    def from_dict(cls, d: dict) -> "WikiRecord":
        return cls(
            lang        = str(d.get("lang", "")),
            title       = str(d.get("title", "")),
            instruction = str(d.get("instruction", "")),
            prompt      = str(d.get("prompt", "")),
            text        = str(d.get("text", "")),
        )

    def to_dict(self) -> dict:
        return {
            "lang": self.lang,
            "title": self.title,
            "instruction": self.instruction,
            "prompt": self.prompt,
            "text": self.text,
        }


def load_dataset(
    data_dir: str,
    lang: str,
    dataset: str = "wiki",
    n: Optional[int] = None,
    skip: int = 0,
) -> List[WikiRecord]:

    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported lang: {lang}. Must be one of {SUPPORTED_LANGS}")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}. Must be one of {SUPPORTED_DATASETS}")

    path = os.path.join(data_dir, f"{dataset}_{lang}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: List[WikiRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < skip:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at line {i + 1} of {path}: {e}")
            rec = WikiRecord.from_dict(d)
            records.append(rec)
            if n is not None and len(records) >= n:
                break

    return records

def load_wiki_dataset(
    data_dir: str,
    lang: str,
    n: Optional[int] = None,
    skip: int = 0,
) -> List[WikiRecord]:

    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported lang: {lang}. Must be one of {SUPPORTED_LANGS}")

    path = os.path.join(data_dir, f"wiki_{lang}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: List[WikiRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < skip:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at line {i + 1} of {path}: {e}")
            rec = WikiRecord.from_dict(d)
            records.append(rec)
            if n is not None and len(records) >= n:
                break

    return records


def stream_wiki_dataset(
    data_dir: str,
    lang: str,
    n: Optional[int] = None,
    skip: int = 0,
) -> Iterator[WikiRecord]:

    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported lang: {lang}")
    path = os.path.join(data_dir, f"wiki_{lang}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < skip:
                continue
            d = json.loads(line)
            yield WikiRecord.from_dict(d)
            yielded += 1
            if n is not None and yielded >= n:
                break


def dataset_summary(data_dir: str) -> dict:

    summary = {"per_lang_count": {}, "missing": [], "total": 0}
    for lang in SUPPORTED_LANGS:
        path = os.path.join(data_dir, f"wiki_{lang}.jsonl")
        if not os.path.exists(path):
            summary["missing"].append(lang)
            summary["per_lang_count"][lang] = 0
            continue
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        summary["per_lang_count"][lang] = n
        summary["total"] += n
    return summary

def load_culturax_records(
    culturax_dir: str,
    lang: str,
    n: int = 100,
    min_chars: int = 300,
    max_chars: int = 4000,
    seed: int = 42,
) -> Optional[List[str]]:

    import random
    path = os.path.join(culturax_dir, f"culturax_{lang}_20k.jsonl")
    if not os.path.exists(path):
        return None

    valid_texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (rec.get("text") or "").strip()
            if min_chars <= len(text) <= max_chars:
                valid_texts.append(text)

    if not valid_texts:
        return None

    rng = random.Random(seed)
    if len(valid_texts) > n:
        valid_texts = rng.sample(valid_texts, n)
    return valid_texts
