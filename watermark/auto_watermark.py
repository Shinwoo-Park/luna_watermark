import torch
import importlib
from typing import List
from watermark.auto_config import AutoConfig

WATERMARK_MAPPING_NAMES = {
    'SynthID': 'watermark.synthid.SynthID',
    'LUNA': 'watermark.luna.LUNA',
    'SynthIDStochastic': 'watermark.synthid_stochastic.SynthIDStochastic',
}

def watermark_name_from_alg_name(name):
    if name in WATERMARK_MAPPING_NAMES:
        return WATERMARK_MAPPING_NAMES[name]
    raise ValueError(f"Invalid algorithm name: {name}")


class AutoWatermark:
    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    def load(algorithm_name, algorithm_config=None, transformers_config=None, *args, **kwargs):
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_config = AutoConfig.load(
            algorithm_name, transformers_config,
            algorithm_config_path=algorithm_config, **kwargs,
        )
        return watermark_class(watermark_config)


vllm_supported_methods = []


class AutoWatermarkForVLLM:
    def __init__(self, algorithm_name, algorithm_config, transformers_config):
        if algorithm_name not in vllm_supported_methods:
            raise NotImplementedError(
                f"vllm integrating currently supports {vllm_supported_methods}, but got {algorithm_name}"
            )
        self.watermark = AutoWatermark.load(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            transformers_config=transformers_config,
        )

    def __call__(self, prompt_tokens, generated_tokens, scores):
        if len(prompt_tokens) == 0:
            return scores
        input_ids = torch.LongTensor(
            prompt_tokens + generated_tokens
        ).to(self.watermark.config.device)[None, :]
        scores = scores[None, :]
        scores = self.watermark.logits_processor(input_ids, scores)
        return scores[0, :]

    def get_data_for_visualization(self, text):
        return self.watermark.get_data_for_visualization(text)

    def detect_watermark(self, text):
        if isinstance(text, list):
            return [self.watermark.detect_watermark(_) for _ in text]
        return self.watermark.detect_watermark(text)
