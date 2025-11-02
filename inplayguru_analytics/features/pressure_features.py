"""Feature engineering helpers for modelling attacking pressure and goal probability.

The InplayGuru live feed exposes a rich set of on-ball and off-ball metrics (dangerous
attacks, possession, momentum gauges, etc.). Most metrics are volatile at the minute
level, therefore we smooth them using exponentially weighted averages to reduce noise
before feeding them into a predictive model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

import math


@dataclass
class PressureFeatures:
    """Aggregated features describing which team is exerting pressure."""

    minute: int
    pressure_score: float
    momentum_index: float
    attack_velocity: float
    shot_conversion_gap: float

    def as_dict(self) -> MutableMapping[str, float]:
        return {
            "minute": float(self.minute),
            "pressure_score": self.pressure_score,
            "momentum_index": self.momentum_index,
            "attack_velocity": self.attack_velocity,
            "shot_conversion_gap": self.shot_conversion_gap,
        }


def _exp_weighted_average(values: Sequence[float], alpha: float = 0.6) -> float:
    if not values:
        return 0.0
    weight = 1.0
    total_weight = 0.0
    acc = 0.0
    for value in reversed(values):
        acc += value * weight
        total_weight += weight
        weight *= alpha
    return acc / total_weight if total_weight else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_pressure_features(history: Iterable[Mapping[str, float]]) -> List[PressureFeatures]:
    """Convert a list of metric snapshots into model-friendly features.

    Parameters
    ----------
    history:
        Iterable of chronological metric snapshots (oldest first). Each snapshot should
        contain at least the following keys (matching
        :func:`InplayGuruClient.normalize_metric_payload`):

        ``minute`` (int), ``dangerous_attacks`` (float), ``total_attacks`` (float),
        ``possession_home`` (float), ``possession_away`` (float), ``shots_on_target``
        (float), ``shots_total`` (float) and ``xg`` (float).
    """

    history_list = list(history)
    window_size = 5

    features: List[PressureFeatures] = []
    dangerous_history: List[float] = []
    attacks_history: List[float] = []
    xg_history: List[float] = []
    shots_on_target_history: List[float] = []
    shots_total_history: List[float] = []

    for snapshot in history_list:
        minute = int(snapshot.get("minute", len(features)))
        dangerous = float(snapshot.get("dangerous_attacks", 0.0))
        attacks = float(snapshot.get("total_attacks", 0.0))
        shots_on_target = float(snapshot.get("shots_on_target", 0.0))
        shots_total = float(snapshot.get("shots_total", 0.0))
        xg = float(snapshot.get("xg", 0.0))
        possession_home = float(snapshot.get("possession_home", 50.0))
        possession_away = float(snapshot.get("possession_away", 50.0))

        dangerous_history.append(dangerous)
        attacks_history.append(attacks)
        xg_history.append(xg)
        shots_on_target_history.append(shots_on_target)
        shots_total_history.append(shots_total)

        if len(dangerous_history) > window_size:
            dangerous_history.pop(0)
            attacks_history.pop(0)
            xg_history.pop(0)
            shots_on_target_history.pop(0)
            shots_total_history.pop(0)

        # Pressure score emphasises recent dangerous attacks and xG spikes.
        pressure_score = _exp_weighted_average(dangerous_history) * 0.4
        pressure_score += _exp_weighted_average(attacks_history) * 0.2
        pressure_score += _exp_weighted_average(xg_history) * 0.4

        # Momentum index measures possession tilt and attacking efficiency.
        possession_gap = possession_home - possession_away
        momentum_index = math.tanh(0.03 * possession_gap) + 0.5 * math.tanh(
            0.5 * _safe_ratio(dangerous, max(attacks, 1.0))
        )

        # Attack velocity tracks change in dangerous attacks over the sliding window.
        if len(dangerous_history) >= 2:
            attack_velocity = dangerous_history[-1] - dangerous_history[0]
        else:
            attack_velocity = 0.0

        # Shot conversion gap highlights periods of pressure without goals.
        shot_conversion_gap = _safe_ratio(shots_on_target, max(shots_total, 1.0))
        shot_conversion_gap -= _safe_ratio(shots_on_target_history[0], max(shots_total_history[0], 1.0))

        features.append(
            PressureFeatures(
                minute=minute,
                pressure_score=pressure_score,
                momentum_index=momentum_index,
                attack_velocity=attack_velocity,
                shot_conversion_gap=shot_conversion_gap,
            )
        )

    return features


__all__ = ["PressureFeatures", "compute_pressure_features"]
