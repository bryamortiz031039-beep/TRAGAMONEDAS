"""Simple logistic regression model to estimate the probability of an upcoming goal."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import math

from inplayguru_analytics.features.pressure_features import PressureFeatures


@dataclass
class PressureGoalModel:
    """Lightweight logistic regression trained on pressure features.

    The model is intentionally simple so that it can be trained with a modest amount of
    historical data and updated online as new examples become available. It mirrors the
    workflow of ``sklearn.linear_model.SGDClassifier`` but keeps the dependency surface
    tiny.
    """

    learning_rate: float = 0.05
    l2: float = 0.001
    weights: List[float] = field(default_factory=lambda: [0.0] * 5)

    def _feature_vector(self, feat: PressureFeatures) -> List[float]:
        return [
            1.0,  # bias term
            float(feat.pressure_score),
            float(feat.momentum_index),
            float(feat.attack_velocity),
            float(feat.shot_conversion_gap),
        ]

    def fit(self, features: Sequence[PressureFeatures], labels: Sequence[int], epochs: int = 50) -> None:
        """Batch-train the model using gradient descent."""

        if not features:
            return

        x = [self._feature_vector(f) for f in features]
        y = [float(label) for label in labels]

        for _ in range(epochs):
            gradients = [0.0] * len(self.weights)
            for xi, yi in zip(x, y):
                prediction = self._sigmoid(self._dot(self.weights, xi))
                error = prediction - yi
                for j in range(len(self.weights)):
                    gradients[j] += error * xi[j]
            for j in range(len(self.weights)):
                gradients[j] /= len(x)
                if j > 0:
                    gradients[j] += self.l2 * self.weights[j]
                self.weights[j] -= self.learning_rate * gradients[j]

    def partial_fit(self, feature: PressureFeatures, label: int) -> None:
        """Perform a single stochastic gradient descent update."""

        x = self._feature_vector(feature)
        prediction = self._sigmoid(self._dot(self.weights, x))
        error = prediction - label
        for j in range(len(self.weights)):
            gradient = error * x[j]
            if j > 0:
                gradient += self.l2 * self.weights[j]
            self.weights[j] -= self.learning_rate * gradient

    def predict_probability(self, feature: PressureFeatures) -> float:
        """Return the probability of a goal within the next minute."""

        x = self._feature_vector(feature)
        return float(self._sigmoid(self._dot(self.weights, x)))

    def predict_proba_bulk(self, features: Iterable[PressureFeatures]) -> List[float]:
        return [self.predict_probability(f) for f in features]

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(weights, features))


__all__ = ["PressureGoalModel"]
