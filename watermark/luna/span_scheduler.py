import torch
import numpy as np
from typing import List, Tuple, Optional

from .pos_tagger import (
    POSTagger,
    SPACED_LANGUAGES,         
    UNSPACED_LANGUAGES,
    DROP_TRAILING_LANGUAGES,  
)
from .lambda_lookup import LambdaLookup

def lambda_to_m(
    lam: float,
    tau1: float, tau2: float,
    m_min: int, m_mid: int, m_max: int,
    mode: str = "tier",
) -> int:

    if mode == "linear":
        if lam <= tau1:
            return m_min
        if lam >= tau2:
            return m_max
        if tau2 - tau1 <= 1e-12:
            return m_mid
        frac = (lam - tau1) / (tau2 - tau1)
        m = m_min + int(round((m_max - m_min) * frac))
        return max(m_min, min(m_max, m))
    if lam < tau1:
        return m_min
    if lam < tau2:
        return m_mid
    return m_max

class IncrementalSpanScheduler:
    def __init__(
        self,
        tagger: POSTagger,
        lambda_lookup: LambdaLookup,
        tokenizer,
        lang: str,
        k_primary: int,
        m_min: int, m_mid: int, m_max: int,
        tau1: float, tau2: float,
        lambda_default: float = 0.5,
        lam_mode: str = "tier",
        drop_trailing_tag: Optional[bool] = None,
    ):
        self.tagger = tagger
        self.lambda_lookup = lambda_lookup
        self.tokenizer = tokenizer
        self.lang = lang
        self.k_primary = k_primary
        self.m_min, self.m_mid, self.m_max = m_min, m_mid, m_max
        self.tau1, self.tau2 = tau1, tau2
        self.lambda_default = lambda_default
        self.lam_mode = lam_mode
        self._drop_trailing_override = drop_trailing_tag

        self._cached_text: str = ""
        self._cached_tags: List[str] = []
        self._cached_m_t: int = m_mid
        self._cached_lambda: float = lambda_default

    def _should_drop_trailing(self) -> bool:
        if self._drop_trailing_override is not None:
            return bool(self._drop_trailing_override)
        return self.lang in DROP_TRAILING_LANGUAGES

    def reset(self) -> None:
        self._cached_text = ""
        self._cached_tags = []
        self._cached_m_t = self.m_mid
        self._cached_lambda = self.lambda_default

    def get_m_and_weight(
        self,
        prefix_token_ids: torch.LongTensor,
        batch_idx: int = 0,
    ) -> Tuple[int, float, float]:

        ids = prefix_token_ids[batch_idx].cpu().tolist()
        prefix_text = self.tokenizer.decode(ids, skip_special_tokens=True)

        if self._is_safe_to_reuse_cache(prefix_text):
            return self._cached_m_t, self._cached_lambda, self._cached_lambda
        tags = [t.pos_fine for t in self.tagger.tag(prefix_text)]
        effective_tags = tags
        if self._should_drop_trailing() and prefix_text:
            if not prefix_text[-1].isspace():
                if effective_tags:
                    effective_tags = effective_tags[:-1]
        ctx_len = self.k_primary - 1
        if len(effective_tags) >= ctx_len:
            ctx_tags = effective_tags[-ctx_len:] if ctx_len > 0 else []
            lam, _k_used, _n = self.lambda_lookup.get(ctx_tags)
        else:
            lam = self.lambda_default

        m_t = lambda_to_m(
            lam, self.tau1, self.tau2,
            self.m_min, self.m_mid, self.m_max,
            mode=self.lam_mode,
        )

        self._cached_text = prefix_text
        self._cached_tags = tags
        self._cached_m_t = m_t
        self._cached_lambda = lam

        return m_t, lam, lam

    def _is_safe_to_reuse_cache(self, new_text: str) -> bool:

        if not self._cached_text:
            return False
        if not new_text.startswith(self._cached_text):
            return False
        appended = new_text[len(self._cached_text):]
        if not appended:
            return True
        if self.lang in UNSPACED_LANGUAGES:
            return False
        if any(c.isspace() for c in appended):
            return False
        return True

class BatchTokenPOSAligner:
    def __init__(
        self,
        tagger: POSTagger,
        lambda_lookup: LambdaLookup,
        tokenizer,
        lang: str,
        k_primary: int,
        m_min: int, m_mid: int, m_max: int,
        tau1: float, tau2: float,
        lambda_default: float = 0.5,
        lam_mode: str = "tier",
        drop_trailing_tag: Optional[bool] = None,
    ):
        self.tagger = tagger
        self.lambda_lookup = lambda_lookup
        self.tokenizer = tokenizer
        self.lang = lang
        self.k_primary = k_primary
        self.m_min, self.m_mid, self.m_max = m_min, m_mid, m_max
        self.tau1, self.tau2 = tau1, tau2
        self.lambda_default = lambda_default
        self.lam_mode = lam_mode
        self._drop_trailing_override = drop_trailing_tag

    def _should_drop_trailing(self) -> bool:
        if self._drop_trailing_override is not None:
            return bool(self._drop_trailing_override)
        return self.lang in DROP_TRAILING_LANGUAGES

    def _cum_char_offsets(self, ids: List[int]) -> List[int]:
        offsets = [0]
        for i in range(1, len(ids) + 1):
            decoded = self.tokenizer.decode(ids[:i], skip_special_tokens=True)
            offsets.append(len(decoded))
        return offsets

    def _word_end_chars_and_tags(
        self, full_text: str
    ) -> Tuple[List[int], List[str]]:
        tagged = self.tagger.tag(full_text)
        end_chars: List[int] = []
        tags: List[str] = []
        cursor = 0
        for tt in tagged:
            if tt.end_char is not None:
                end = int(tt.end_char)
            else:
                idx = full_text.find(tt.text, cursor)
                if idx == -1:
                    idx = cursor
                end = idx + len(tt.text)
            end_chars.append(end)
            tags.append(tt.pos_fine)
            cursor = end
        return end_chars, tags

    def align(
        self,
        encoded_text: torch.LongTensor,
        ngram_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        B, T = encoded_text.shape
        N = T - ngram_len + 1

        m_t_arr = np.full((B, N), self.m_mid, dtype=np.int64)
        w_arr = np.full((B, N), self.lambda_default, dtype=np.float64)
        if N <= 0:
            return m_t_arr, w_arr

        for b in range(B):
            ids = encoded_text[b].cpu().tolist()
            cum_chars = self._cum_char_offsets(ids)
            full_text = self.tokenizer.decode(ids, skip_special_tokens=True)

            try:
                pos_end_chars, pos_tags = self._word_end_chars_and_tags(full_text)
            except Exception:
                continue

            ctx_len = self.k_primary - 1
            pos_pointer = 0
            for p in range(N):
                seq_p = p + ngram_len - 1
                if seq_p < len(cum_chars):
                    char_end = cum_chars[seq_p]
                else:
                    char_end = cum_chars[-1]
                while (pos_pointer < len(pos_end_chars)
                       and pos_end_chars[pos_pointer] <= char_end):
                    pos_pointer += 1

                completed_count = pos_pointer

                if (self._should_drop_trailing()
                        and 0 < char_end <= len(full_text)
                        and not full_text[char_end - 1].isspace()
                        and completed_count > 0
                        and pos_end_chars[completed_count - 1] == char_end):
                    completed_count -= 1

                if completed_count >= ctx_len:
                    if ctx_len > 0:
                        ctx_tags = pos_tags[completed_count - ctx_len:completed_count]
                    else:
                        ctx_tags = []
                    lam, _k_used, _n = self.lambda_lookup.get(ctx_tags)
                else:
                    lam = self.lambda_default

                m_t = lambda_to_m(
                    lam, self.tau1, self.tau2,
                    self.m_min, self.m_mid, self.m_max,
                    mode=self.lam_mode,
                )
                m_t_arr[b, p] = m_t
                w_arr[b, p] = lam

        return m_t_arr, w_arr
