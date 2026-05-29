
import math
import torch
import numpy as np
from functools import partial
from typing import Tuple

from transformers import LogitsProcessorList

from ..synthid.synthid import (
    SynthIDConfig,
    SynthIDUtils,
    SynthIDLogitsProcessor,
    SynthID,
)
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization


class SynthIDStochasticConfig(SynthIDConfig):

    def initialize_parameters(self) -> None:
        super().initialize_parameters()

        self.budget_B = float(self.config_dict.get('budget_B', 268_451_848.0))

        self.stochastic_salt = int(self.config_dict.get('stochastic_salt', 12345))

        self.z_threshold = float(self.config_dict.get('z_threshold', 4.0))

        log2B = math.log2(self.budget_B)
        self.m_floor = int(math.floor(log2B))
        self.m_ceil = int(math.ceil(log2B))
        if self.m_floor == self.m_ceil:
            self.p_ceil = 0.0
        else:
            denom = (2 ** self.m_ceil) - (2 ** self.m_floor)
            self.p_ceil = (self.budget_B - (2 ** self.m_floor)) / denom

            self.p_ceil = float(np.clip(self.p_ceil, 0.0, 1.0))

        max_depth = len(self.keys)
        if self.m_ceil > max_depth:
            raise ValueError(
                f"m_ceil ({self.m_ceil}) exceeds len(keys) ({max_depth}). "
                f"Increase keys list or lower budget_B."
            )
        if self.m_floor < 0:
            raise ValueError(f"m_floor ({self.m_floor}) is negative. Increase budget_B.")

    @property
    def algorithm_name(self) -> str:
        return 'SynthIDStochastic'

class SynthIDStochasticUtils(SynthIDUtils):

    def update_scores_variable_depth(
        self,
        scores: torch.FloatTensor,
        g_values: torch.FloatTensor,
        m_t: int,
    ) -> torch.FloatTensor:

        _, _, max_depth = g_values.shape
        if not (0 <= m_t <= max_depth):
            raise ValueError(f"m_t={m_t} out of range [0, {max_depth}]")
        device = scores.device

        if m_t == 0:
            log_probs = torch.log_softmax(scores, dim=1)
            return log_probs

        probs = torch.softmax(scores, dim=1)
        for i in range(m_t):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(dim=1, keepdim=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
        )
        return log_probs


class LayerSumZDetector:


    def __init__(self):
        pass

    def detect(
        self,
        g_values: np.ndarray,        
        position_mask: np.ndarray,   
        m_t_per_pos: np.ndarray,     
        weights_per_pos: np.ndarray, 
    ) -> np.ndarray:
        B, T, max_depth = g_values.shape

        layer_idx = np.arange(max_depth, dtype=np.int64).reshape(1, 1, max_depth)
        active_layer = (layer_idx < m_t_per_pos[:, :, None]).astype(np.float64)
        layer_mask = active_layer * position_mask[:, :, None].astype(np.float64)

        centered = g_values.astype(np.float64) - 0.5

        weights_3d = weights_per_pos[:, :, None].astype(np.float64)
        weighted = centered * layer_mask * weights_3d
        numerator = weighted.sum(axis=(1, 2))  

        m_t_active = layer_mask.sum(axis=2)  
        var_term = (0.25 * m_t_active * (weights_per_pos.astype(np.float64) ** 2)).sum(axis=1)  
        denom = np.sqrt(np.maximum(var_term, 1e-12))

        return numerator / denom


class SynthIDStochasticLogitsProcessor(SynthIDLogitsProcessor):

    def __init__(self, config: SynthIDStochasticConfig, utils: SynthIDStochasticUtils):
        super().__init__(config, utils)
        self._m_history = []

    def reset_state(self) -> None:
        self.state = None
        self._m_history = []

    @staticmethod
    def _hash_to_m_t(
        hash_value: int,
        salt: int,
        m_floor: int,
        m_ceil: int,
        p_ceil: float,
    ) -> int:

        if m_floor == m_ceil:
            return m_floor
        mixed = (int(hash_value) ^ int(salt)) & ((1 << 63) - 1)

        u = (mixed & ((1 << 53) - 1)) / float(1 << 53)
        return m_ceil if u < p_ceil else m_floor

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        scores_processed = scores / self.config.temperature
        batch_size, vocab_size = scores.shape

        if self.config.top_k > 0:
            top_k_result = torch.topk(scores_processed, k=self.config.top_k, dim=1)
            scores_top_k = top_k_result.values
            top_k_indices = top_k_result.indices
        else:
            scores_top_k = scores_processed
            top_k_indices = torch.stack([
                torch.arange(vocab_size, device=self.device)
                for _ in range(batch_size)
            ])

        if self.state is None:
            self.state = {
                "context": torch.zeros(
                    (batch_size, self.ngram_len - 1),
                    dtype=torch.int64, device=self.device,
                ),
                "context_history": torch.zeros(
                    (batch_size, self.context_history_size),
                    dtype=torch.int64, device=self.device,
                ),
                "num_calls": 0,
            }

        if self.state["num_calls"] > 0:
            self.state["context"] = torch.cat(
                (self.state["context"], input_ids[:, -1:]), dim=1
            )[:, 1:]
        self.state["num_calls"] += 1

        ngram_keys, hash_context = self._compute_keys(self.state["context"], top_k_indices)
        g_values = self.sample_g_values(ngram_keys)

        m_ts = []
        for b in range(batch_size):
            m_t = self._hash_to_m_t(
                hash_value=int(hash_context[b].item()),
                salt=self.config.stochastic_salt,
                m_floor=self.config.m_floor,
                m_ceil=self.config.m_ceil,
                p_ceil=self.config.p_ceil,
            )
            m_ts.append(m_t)

        m_t_active = m_ts[0]
        self._m_history.append(m_t_active)

        updated_top_k = self.utils.update_scores_variable_depth(
            scores_top_k, g_values, m_t_active
        )

        hash_context_col = hash_context[:, None]
        is_repeated = (self.state["context_history"] == hash_context_col).any(
            dim=1, keepdim=True
        )
        self.state["context_history"] = torch.cat(
            (hash_context_col, self.state["context_history"]), dim=1
        )[:, :-1]

        if self.config.top_k > 0:
            full_scores = scores_processed.clone()
            full_scores.scatter_(1, top_k_indices, updated_top_k)
            updated_scores = full_scores
        else:
            updated_scores = updated_top_k

        return torch.where(is_repeated, scores, updated_scores)

class SynthIDStochastic(SynthID):

    def __init__(
        self,
        algorithm_config: "str | SynthIDStochasticConfig",
        transformers_config: TransformersConfig | None = None,
        *args, **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = SynthIDStochasticConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SynthIDStochasticConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be a path string or SynthIDStochasticConfig instance"
            )

        self.utils = SynthIDStochasticUtils(self.config)
        self.logits_processor = SynthIDStochasticLogitsProcessor(self.config, self.utils)
        self.detector = LayerSumZDetector()

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        self.logits_processor.reset_state()

        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        return self.config.generation_tokenizer.batch_decode(
            encoded_watermarked_text, skip_special_tokens=True
        )[0]

    def _compute_hash_contexts(self, encoded_text: torch.LongTensor) -> torch.LongTensor:

        contexts = encoded_text[:, :-1].unfold(
            dimension=1, size=self.config.ngram_len - 1, step=1
        )
        B, N, _ = contexts.shape
        results = torch.empty((B, N), dtype=torch.long, device=self.config.device)
        ones = torch.ones(B, device=self.config.device, dtype=torch.long)
        for p in range(N):
            results[:, p] = self.utils.accumulate_hash(ones, contexts[:, p, :])
        return results

    def _compute_m_t_per_pos(
        self, encoded_text: torch.LongTensor
    ) -> Tuple[np.ndarray, np.ndarray]:

        hash_contexts = self._compute_hash_contexts(encoded_text)  
        B, N = hash_contexts.shape
        m_t_arr = np.zeros((B, N), dtype=np.int64)
        for b in range(B):
            for p in range(N):
                m_t_arr[b, p] = SynthIDStochasticLogitsProcessor._hash_to_m_t(
                    hash_value=int(hash_contexts[b, p].item()),
                    salt=self.config.stochastic_salt,
                    m_floor=self.config.m_floor,
                    m_ceil=self.config.m_ceil,
                    p_ceil=self.config.p_ceil,
                )
        weights = np.ones((B, N), dtype=np.float64)
        return m_t_arr, weights

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.config.device)

        if encoded_text.shape[1] < self.config.ngram_len:
            return (
                {"is_watermarked": False, "score": 0.0}
                if return_dict
                else (False, 0.0)
            )

        g_values = self.logits_processor.compute_g_values(encoded_text)  

        eos_mask = self.logits_processor.compute_eos_token_mask(
            input_ids=encoded_text,
            eos_token_id=self.config.generation_tokenizer.eos_token_id,
        )[:, self.config.ngram_len - 1:]
        context_repetition_mask = self.logits_processor.compute_context_repetition_mask(
            input_ids=encoded_text
        )
        combined_mask = (context_repetition_mask * eos_mask).cpu().numpy()

        m_t_per_pos, weights_per_pos = self._compute_m_t_per_pos(encoded_text)

        g_values_np = g_values.cpu().numpy()
        z_scores = self.detector.detect(g_values_np, combined_mask, m_t_per_pos, weights_per_pos)
        z_score = float(z_scores[0])

        is_watermarked = z_score > self.config.z_threshold
        if return_dict:
            return {
                "is_watermarked": bool(is_watermarked),
                "score": z_score,
                "m_t_mean": float(np.mean(m_t_per_pos[combined_mask > 0]))
                            if combined_mask.sum() > 0 else 0.0,
            }
        return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.config.device)

        decoded_tokens = []
        highlight_values = []
        if encoded_text.shape[1] < self.config.ngram_len:
            for tid in encoded_text[0]:
                decoded_tokens.append(self.config.generation_tokenizer.decode(tid.item()))
                highlight_values.append(0.0)
            return DataForVisualization(decoded_tokens, highlight_values)

        g_values = self.logits_processor.compute_g_values(encoded_text)[0].cpu().numpy()
        m_t_per_pos, _ = self._compute_m_t_per_pos(encoded_text)
        m_t_per_pos = m_t_per_pos[0]

        prefix_len = self.config.ngram_len - 1
        for i in range(encoded_text.shape[1]):
            decoded_tokens.append(self.config.generation_tokenizer.decode(encoded_text[0, i].item()))
            if i < prefix_len:
                highlight_values.append(0.0)
            else:
                p = i - prefix_len
                m_t = int(m_t_per_pos[p])
                if m_t == 0:
                    highlight_values.append(0.0)
                else:
                    layer_mean = float(np.mean(g_values[p, :m_t]))
                    highlight_values.append(layer_mean - 0.5)
        return DataForVisualization(decoded_tokens, highlight_values)
