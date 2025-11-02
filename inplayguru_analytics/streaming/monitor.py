"""Streaming monitor that combines the client, feature engineering and model layers."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable, List, Optional

from collections import deque

from inplayguru_analytics.clients.inplayguru_client import InplayGuruClient
from inplayguru_analytics.features.pressure_features import PressureFeatures, compute_pressure_features
from inplayguru_analytics.models.pressure_model import PressureGoalModel


logger = logging.getLogger(__name__)


PredictionCallback = Callable[[int, PressureFeatures, float], None]


@dataclass
class LivePressureMonitor:
    """Continuously poll metrics, compute features and emit goal probabilities."""

    client: InplayGuruClient
    model: PressureGoalModel
    history_window: int = 12
    poll_interval: float = 10.0
    on_prediction: Optional[PredictionCallback] = None
    _history: Deque[dict] = field(default_factory=lambda: deque(maxlen=240))

    def run(self, match_id: str, *, stop_after: Optional[int] = None) -> None:
        """Start the polling loop.

        Parameters
        ----------
        match_id:
            Identifier of the match to monitor.
        stop_after:
            Optional amount of seconds after which the loop will stop. Useful for demos
            and tests.
        """

        logger.info("Starting live monitor for match %s", match_id)
        start_time = time.monotonic()
        iterations = 0

        while True:
            if stop_after is not None and time.monotonic() - start_time > stop_after:
                logger.info("Stopping live monitor after %.1f seconds", stop_after)
                break

            snapshot = self.client.fetch_match_metrics(match_id)
            normalized = self.client.normalize_metric_payload(snapshot)
            self._history.append(normalized)
            iterations += 1

            features = compute_pressure_features(list(self._history)[-self.history_window :])
            if not features:
                time.sleep(self.poll_interval)
                continue

            latest_feature = features[-1]
            probability = self.model.predict_probability(latest_feature)

            if self.on_prediction:
                try:
                    self.on_prediction(iterations, latest_feature, probability)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("Prediction callback raised an exception: %s", exc)

            logger.debug(
                "minute=%s pressure=%.3f momentum=%.3f velocity=%.3f prob=%.3f",
                latest_feature.minute,
                latest_feature.pressure_score,
                latest_feature.momentum_index,
                latest_feature.attack_velocity,
                probability,
            )

            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Utilities
    def export_history(self) -> List[dict]:
        return list(self._history)

    def export_history_json(self) -> str:
        return json.dumps(self.export_history(), indent=2)


__all__ = ["LivePressureMonitor", "PredictionCallback"]
