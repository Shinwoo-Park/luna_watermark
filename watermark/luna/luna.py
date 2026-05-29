import math
import torch
import numpy as np
from functools import partial
from typing import Tuple, Optional

from transformers import LogitsProcessorList

from ..synthid.synthid import (
    SynthIDConfig,
    SynthIDUtils,
    SynthIDLogitsProcessor,
    SynthID,
)
from ..synthid_stochastic.synthid_stochastic import (
    SynthIDStochasticUtils,  
    LayerSumZDetector,        
)
from .pos_tagger import get_tagger, POSTagger
from .lambda_lookup import LambdaLookup
from .span_scheduler import IncrementalSpanScheduler, BatchTokenPOSAligner

from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization

DEFAULT_K_PRIMARY = {
    "en": 2,
    "zh": 4,
    "ko": 4,
    "ja": 4,
    "de": 3,
    "ar": 4,
}

class LUNAConfig(SynthIDConfig):

    def initialize_parameters(self) -> None:
        super().initialize_parameters()

        self.lang = str(self.config_dict.get("lang", "en"))
        if self.lang not in DEFAULT_K_PRIMARY:
            raise ValueError(f"Unsupported language: {self.lang}")
        self.k_primary = int(self.config_dict.get("k_primary", DEFAULT_K_PRIMARY[self.lang]))

        self.lambda_dir = str(self.config_dict.get("lambda_dir", "data"))

        self.m_min = int(self.config_dict.get("m_min", 5))
        self.m_mid = int(self.config_dict.get("m_mid", 15))
        self.m_max = int(self.config_dict.get("m_max", 30))

        self.tau1: Optional[float] = self.config_dict.get("tau1", None)
        self.tau2: Optional[float] = self.config_dict.get("tau2", None)

        self.z_threshold = float(self.config_dict.get("z_threshold", 4.0))
        self.lambda_default = float(self.config_dict.get("lambda_default", 0.5))

        self.dirichlet_alpha = float(self.config_dict.get("dirichlet_alpha", 0.01))
        self.min_freq = int(self.config_dict.get("min_freq", 1))

        self.lam_mode = str(self.config_dict.get("lam_mode", "tier"))
        if self.lam_mode not in ("tier", "linear"):
            raise ValueError(
                f"lam_mode must be 'tier' or 'linear'; got {self.lam_mode!r}"
            )

        self.drop_trailing_tag: Optional[bool] = self.config_dict.get(
            "drop_trailing_tag", None,
        )

        if not (self.m_min <= self.m_mid <= self.m_max):
            raise ValueError(
                f"Require m_min <= m_mid <= m_max; got ({self.m_min}, {self.m_mid}, {self.m_max})"
            )
        if self.m_max > len(self.keys):
            raise ValueError(
                f"m_max ({self.m_max}) exceeds len(keys) ({len(self.keys)}). "
                f"Increase keys list."
            )
        if self.k_primary < 2:
            raise ValueError(f"k_primary must be >= 2; got {self.k_primary}")

    @property
    def algorithm_name(self) -> str:
        return "LUNA"

class LUNAUtils(SynthIDStochasticUtils):
    pass

class LUNALogitsProcessor(SynthIDLogitsProcessor):

    def __init__(
        self,
        config: LUNAConfig,
        utils: LUNAUtils,
        scheduler: IncrementalSpanScheduler,
    ):
        super().__init__(config, utils)
        self.scheduler = scheduler

        self._m_history = []
        self._lambda_history = []

    def reset_state(self) -> None:
        self.state = None
        self._m_history = []
        self._lambda_history = []
        self.scheduler.reset()

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

        m_t_active, lam, _w = self.scheduler.get_m_and_weight(input_ids, batch_idx=0)
        self._m_history.append(m_t_active)
        self._lambda_history.append(lam)

        updated_top_k = self.utils.update_scores_variable_depth(
            scores_top_k, g_values, m_t_active
        )

        hash_context_col = hash_context[:, None]
        is_repeated = (self.state["context_history"] == hash_context_col).any(
            dim=1, keepdim=True,
        )
        self.state["context_history"] = torch.cat(
            (hash_context_col, self.state["context_history"]), dim=1,
        )[:, :-1]

        if self.config.top_k > 0:
            full_scores = scores_processed.clone()
            full_scores.scatter_(1, top_k_indices, updated_top_k)
            updated_scores = full_scores
        else:
            updated_scores = updated_top_k

        return torch.where(is_repeated, scores, updated_scores)

class LUNA(SynthID):

    def __init__(
        self,
        algorithm_config: "str | LUNAConfig",
        transformers_config: TransformersConfig | None = None,
        *args, **kwargs,
    ) -> None:
        # Config
        if isinstance(algorithm_config, str):
            self.config = LUNAConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, LUNAConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be a path string or LUNAConfig instance"
            )

        self.lambda_lookup = LambdaLookup(
            lang=self.config.lang,
            k_primary=self.config.k_primary,
            lambda_dir=self.config.lambda_dir,
            lambda_default=self.config.lambda_default,
            min_freq=self.config.min_freq,
        )

        if self.config.tau1 is None or self.config.tau2 is None:
            tau1_auto, tau2_auto = self.lambda_lookup.compute_percentile_thresholds(
                p_low=25.0, p_high=75.0, weighted=True,
            )
            self.config.tau1 = tau1_auto if self.config.tau1 is None else self.config.tau1
            self.config.tau2 = tau2_auto if self.config.tau2 is None else self.config.tau2
        if not (self.config.tau1 < self.config.tau2):
            raise ValueError(
                f"Require tau1 < tau2; got ({self.config.tau1}, {self.config.tau2}). "
                f"This can happen if the λ distribution is highly degenerate."
            )

        self.tagger: POSTagger = get_tagger(self.config.lang)

        self._gen_scheduler = IncrementalSpanScheduler(
            tagger=self.tagger,
            lambda_lookup=self.lambda_lookup,
            tokenizer=self.config.generation_tokenizer,
            lang=self.config.lang,
            k_primary=self.config.k_primary,
            m_min=self.config.m_min,
            m_mid=self.config.m_mid,
            m_max=self.config.m_max,
            tau1=self.config.tau1,
            tau2=self.config.tau2,
            lambda_default=self.config.lambda_default,
            lam_mode=self.config.lam_mode,
            drop_trailing_tag=self.config.drop_trailing_tag,
        )
        self._det_aligner = BatchTokenPOSAligner(
            tagger=self.tagger,
            lambda_lookup=self.lambda_lookup,
            tokenizer=self.config.generation_tokenizer,
            lang=self.config.lang,
            k_primary=self.config.k_primary,
            m_min=self.config.m_min,
            m_mid=self.config.m_mid,
            m_max=self.config.m_max,
            tau1=self.config.tau1,
            tau2=self.config.tau2,
            lambda_default=self.config.lambda_default,
            lam_mode=self.config.lam_mode,
            drop_trailing_tag=self.config.drop_trailing_tag,
        )

        self.utils = LUNAUtils(self.config)
        self.logits_processor = LUNALogitsProcessor(self.config, self.utils, self._gen_scheduler)
        self.detector = LayerSumZDetector()

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        self.logits_processor.reset_state()
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs,
        )
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
        ).to(self.config.device)
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        return self.config.generation_tokenizer.batch_decode(
            encoded_watermarked_text, skip_special_tokens=True,
        )[0]

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
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
            input_ids=encoded_text,
        )
        combined_mask = (context_repetition_mask * eos_mask).cpu().numpy()

        m_t_per_pos, weights_per_pos = self._det_aligner.align(
            encoded_text, ngram_len=self.config.ngram_len,
        )

        g_values_np = g_values.float().cpu().numpy()
        z_scores = self.detector.detect(g_values_np, combined_mask, m_t_per_pos, weights_per_pos)
        z_score = float(z_scores[0])

        is_watermarked = z_score > self.config.z_threshold
        if return_dict:
            valid_count = int(combined_mask.sum())
            return {
                "is_watermarked": bool(is_watermarked),
                "score": z_score,
                "m_t_mean": float(np.mean(m_t_per_pos[combined_mask > 0]))
                            if valid_count > 0 else 0.0,
                "lambda_mean": float(np.mean(weights_per_pos[combined_mask > 0]))
                               if valid_count > 0 else 0.0,
                "num_valid_positions": valid_count,
            }
        return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        encoded_text = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(self.config.device)

        decoded_tokens = []
        highlight_values = []

        if encoded_text.shape[1] < self.config.ngram_len:
            for tid in encoded_text[0]:
                decoded_tokens.append(self.config.generation_tokenizer.decode(tid.item()))
                highlight_values.append(0.0)
            return DataForVisualization(decoded_tokens, highlight_values)

        g_values = self.logits_processor.compute_g_values(encoded_text)[0].float().cpu().numpy()
        m_t_per_pos, weights_per_pos = self._det_aligner.align(
            encoded_text, ngram_len=self.config.ngram_len,
        )
        m_t_per_pos = m_t_per_pos[0]
        weights_per_pos = weights_per_pos[0]

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
                    weighted = (layer_mean - 0.5) * float(weights_per_pos[p])
                    highlight_values.append(weighted)
        return DataForVisualization(decoded_tokens, highlight_values)
