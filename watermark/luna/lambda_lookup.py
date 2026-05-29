import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple


CTX_SEP = "|"


class LambdaLookup:

    def __init__(
        self,
        lang: str,
        k_primary: int,
        lambda_dir: str,
        lambda_default: float = 0.5,
        min_freq: int = 1,
    ):
        self.lang = lang
        self.k_primary = k_primary
        self.lambda_dir = lambda_dir
        self.lambda_default = float(lambda_default)
        self.min_freq = int(min_freq)

        self._tables: Dict[int, Dict[str, Dict]] = {}
        self._tagsets: Dict[int, str] = {}
        self._analyzers: Dict[int, str] = {}
        self._loaded_ks: List[int] = []

        for k in range(k_primary, 1, -1):
            path = os.path.join(lambda_dir, f"lambda_{lang}_k{k}.json")
            if not os.path.exists(path):
                if k == k_primary:
                    raise FileNotFoundError(
                        f"Primary lambda file not found: {path}\n"
                        f"Run estimate_lambda.py to produce it, or set lambda_dir / k_primary correctly."
                    )
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("lang") != lang:
                raise ValueError(f"Lang mismatch in {path}: expected '{lang}', got '{data.get('lang')}'")
            if data.get("k") != k:
                raise ValueError(f"k mismatch in {path}: expected {k}, got {data.get('k')}")
            self._tables[k] = data["m_phi"]
            self._tagsets[k] = data.get("tagset", "")
            self._analyzers[k] = data.get("analyzer", "")
            self._loaded_ks.append(k)

        if not self._loaded_ks:
            raise RuntimeError(f"No lambda tables loaded for lang={lang}")

        self._tau_cache: Optional[Tuple[float, float]] = None

    def get(
        self,
        context_tags: List[str],
    ) -> Tuple[float, int, int]:

        if not context_tags:
            return self.lambda_default, 0, 0

        for k in self._loaded_ks:
            ctx_len = k - 1
            if ctx_len > len(context_tags):
                continue
            ctx = context_tags[-ctx_len:] if ctx_len > 0 else []
            key = CTX_SEP.join(ctx)
            entry = self._tables[k].get(key)
            if entry is None:
                continue
            n = int(entry.get("n", 0))
            if n < self.min_freq:
                continue
            return float(entry["m"]), k, n

        return self.lambda_default, 0, 0

    def compute_percentile_thresholds(
        self,
        p_low: float = 25.0,
        p_high: float = 75.0,
        weighted: bool = True,
    ) -> Tuple[float, float]:

        if self._tau_cache is not None and (p_low, p_high) == (25.0, 75.0):
            return self._tau_cache

        primary = self._tables[self.k_primary]
        lambdas = np.array([v["m"] for v in primary.values()], dtype=np.float64)
        if weighted:
            weights = np.array([v["n"] for v in primary.values()], dtype=np.float64)
            order = np.argsort(lambdas)
            lambdas_s = lambdas[order]
            weights_s = weights[order]
            cum = np.cumsum(weights_s)
            total = cum[-1] if cum[-1] > 0 else 1.0
            cum_norm = cum / total

            def _pct(p):
                idx = np.searchsorted(cum_norm, p / 100.0)
                idx = min(idx, len(lambdas_s) - 1)
                return float(lambdas_s[idx])

            tau1 = _pct(p_low)
            tau2 = _pct(p_high)
        else:
            tau1 = float(np.percentile(lambdas, p_low))
            tau2 = float(np.percentile(lambdas, p_high))

        if (p_low, p_high) == (25.0, 75.0):
            self._tau_cache = (tau1, tau2)
        return tau1, tau2

    @property
    def loaded_ks(self) -> List[int]:
        return list(self._loaded_ks)

    @property
    def tagset(self) -> str:
        return self._tagsets.get(self.k_primary, "")

    @property
    def analyzer(self) -> str:
        return self._analyzers.get(self.k_primary, "")

    def __repr__(self) -> str:
        return (f"LambdaLookup(lang={self.lang}, k_primary={self.k_primary}, "
                f"loaded_ks={self._loaded_ks}, tagset={self.tagset})")
