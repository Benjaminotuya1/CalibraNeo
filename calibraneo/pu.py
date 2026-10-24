"""Optional positive-unlabeled calibration utilities.

These functions are not applied to the current verified binary labels. They
are provided for future datasets where positives are known but negatives are
unlabeled and a class-prior estimate is available.
"""

from __future__ import annotations

import numpy as np


def elkan_noto_correct(
    positive_probability: np.ndarray,
    positive_capture_rate: float,
) -> np.ndarray:
    """Estimate P(y=1|x) from P(s=1|x) using the Elkan-Noto correction."""
    if not 0 < positive_capture_rate <= 1:
        raise ValueError("positive_capture_rate must be in (0, 1]")
    values = np.asarray(positive_probability, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("positive_probability must be within [0, 1]")
    return np.clip(values / positive_capture_rate, 0.0, 1.0)


def pu_sensitivity(
    positive_probability: np.ndarray,
    capture_rates: list[float] | tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Return corrected probabilities across plausible capture-rate assumptions."""
    return {rate: elkan_noto_correct(positive_probability, rate) for rate in capture_rates}
