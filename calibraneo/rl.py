"""Small, reproducible tabular RL environment for cassette selection.

This is a research baseline, not a claim that RL is biologically superior.
The environment makes the action/state/reward definition explicit so it can
be compared with greedy and integer-programming baselines.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def train_q_policy(
    candidates: pd.DataFrame,
    capacity: int,
    dominant_loh_risk: float,
    *,
    episodes: int = 500,
    seed: int = 42,
    learning_rate: float = 0.15,
    discount: float = 0.90,
    epsilon: float = 0.15,
) -> list[int]:
    required = {"hla_clean", "CalibraNeo_Prob"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing RL columns: {', '.join(sorted(missing))}")
    if capacity < 1 or len(candidates) < capacity:
        raise ValueError("capacity must be positive and no larger than candidates")
    rng = np.random.default_rng(seed)
    frame = candidates.reset_index(drop=True)
    alleles = sorted(frame["hla_clean"].astype(str).unique())
    allele_index = {allele: i for i, allele in enumerate(alleles)}
    dominant = str(frame["hla_clean"].value_counts().index[0])
    risks = np.array([dominant_loh_risk if h == dominant else 0.10 for h in frame["hla_clean"]])
    values = frame["CalibraNeo_Prob"].to_numpy(float) * (1.0 - risks)
    q: dict[tuple[tuple[int, ...], tuple[int, ...]], np.ndarray] = defaultdict(
        lambda: np.zeros(len(frame), dtype=float)
    )

    def state(selected: tuple[int, ...], counts: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return selected, counts

    for _ in range(episodes):
        selected: tuple[int, ...] = ()
        counts = tuple(0 for _ in alleles)
        for _slot in range(capacity):
            available = [i for i in range(len(frame)) if i not in selected]
            current = state(selected, counts)
            if rng.random() < epsilon:
                action = int(rng.choice(available))
            else:
                action = max(available, key=lambda i: (q[current][i], -i))
            allele = allele_index[str(frame.loc[action, "hla_clean"])]
            reward = values[action] * np.exp(-0.45 * counts[allele])
            next_counts = list(counts)
            next_counts[allele] += 1
            next_selected = selected + (action,)
            next_state = state(next_selected, tuple(next_counts))
            future = 0.0 if len(next_selected) == capacity else np.max(q[next_state])
            q[current][action] += learning_rate * (
                reward + discount * future - q[current][action]
            )
            selected, counts = next_selected, tuple(next_counts)

    start = state((), tuple(0 for _ in alleles))
    selected: list[int] = []
    counts = [0 for _ in alleles]
    for _slot in range(capacity):
        available = [i for i in range(len(frame)) if i not in selected]
        action = max(available, key=lambda i: (q[state(tuple(selected), tuple(counts))][i], -i))
        selected.append(action)
        counts[allele_index[str(frame.loc[action, "hla_clean"])]] += 1
    return selected
