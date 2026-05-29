import importlib
from typing import Dict, Optional, Any
from utils.transformers_config import TransformersConfig

CONFIG_MAPPING_NAMES = {
    'SynthID': 'watermark.synthid.SynthIDConfig',
    'LUNA': 'watermark.luna.LUNAConfig',
    'SynthIDStochastic': 'watermark.synthid_stochastic.SynthIDStochasticConfig',
}

def config_name_from_alg_name(name: str) -> Optional[str]:
    if name in CONFIG_MAPPING_NAMES:
        return CONFIG_MAPPING_NAMES[name]
    raise ValueError(f"Invalid algorithm name: {name}")

class AutoConfig:
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.load(algorithm_name, **kwargs)` method."
        )

    @classmethod
    def load(
        cls,
        algorithm_name: str,
        transformers_config: TransformersConfig,
        algorithm_config_path: str = None,
        **kwargs,
    ) -> Any:
        config_name = config_name_from_alg_name(algorithm_name)
        if config_name is None:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
        module_name, class_name = config_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        if algorithm_config_path is None:
            algorithm_config_path = f'config/{algorithm_name}.json'
        return config_class(algorithm_config_path, transformers_config, **kwargs)
