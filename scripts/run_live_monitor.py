"""Command line helper to demo the live pressure monitoring workflow."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inplayguru_analytics.clients.inplayguru_client import InplayGuruClient
from inplayguru_analytics.features.pressure_features import compute_pressure_features
from inplayguru_analytics.models.pressure_model import PressureGoalModel
from inplayguru_analytics.streaming.monitor import LivePressureMonitor


def _load_offline_history(path: Path):
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return [InplayGuruClient.normalize_metric_payload(entry) for entry in payload]


def _train_stub_model(history) -> PressureGoalModel:
    features = compute_pressure_features(history)
    if not features:
        raise ValueError("Not enough snapshots to compute features")

    labels = [0] * len(features)
    # Flag spikes in pressure as imminent goals (puremente con fines demostrativos).
    labels[-1] = 1

    model = PressureGoalModel(learning_rate=0.1)
    model.fit(features, labels, epochs=200)
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match_id", help="Identifier of the match to monitor")
    parser.add_argument(
        "--history",
        type=Path,
        help="Offline JSON file containing metric snapshots (for demos/testing)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between API requests",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=60,
        help="Stop after the given amount of seconds (0 to run indefinitely)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = InplayGuruClient()

    if args.history:
        history = _load_offline_history(args.history)
        model = _train_stub_model(history)

        def on_prediction(iteration, feature, probability):
            logging.info(
                "#%02d minute=%s pressure=%.3f momentum=%.3f velocity=%.3f goal_prob=%.2f%%",
                iteration,
                feature.minute,
                feature.pressure_score,
                feature.momentum_index,
                feature.attack_velocity,
                probability * 100,
            )

        offline_iterator = iter(history)

        def _offline_fetch(_match_id):
            try:
                return next(offline_iterator)
            except StopIteration:
                return history[-1]

        client.fetch_match_metrics = _offline_fetch  # type: ignore[assignment]

        monitor = LivePressureMonitor(
            client=client,
            model=model,
            poll_interval=args.poll_interval,
            on_prediction=on_prediction,
        )

        for snapshot in history:
            monitor._history.append(snapshot)  # preload history for demo

        monitor.run(match_id=args.match_id, stop_after=args.stop_after or None)
        return 0

    # Real online mode requires valid credentials. Users are expected to provide their
    # own model via the SDK-like API exposed by this repository.
    logging.error(
        "Online mode requires training data and credentials. Please supply --history"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
