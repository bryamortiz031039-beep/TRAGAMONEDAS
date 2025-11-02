"""HTTP client for interacting with the unofficial InplayGuru live metrics API.

The public site (https://inplayguru.com) exposes a JSON API that can be queried with
standard HTTP requests. The exact endpoints and authentication flow may change over
time, so the client in this module focuses on resilience and debuggability while
remaining lightweight enough to run in real-time streaming jobs.

Example
-------
>>> from inplayguru_analytics.clients.inplayguru_client import InplayGuruClient
>>> client = InplayGuruClient()
>>> live_matches = client.list_live_matches()
>>> metrics = client.fetch_match_metrics(match_id=live_matches[0]["id"])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import requests


@dataclass
class InplayGuruClientConfig:
    """Configuration required to talk to the InplayGuru backend."""

    base_url: str = "https://inplayguru.com/api"
    timeout_seconds: int = 10
    headers: Optional[Mapping[str, str]] = None
    cookies: Optional[Mapping[str, str]] = None


class InplayGuruClient:
    """Small helper around ``requests`` for polling InplayGuru metrics.

    The real production API is a mix of HTTP endpoints and a websocket stream. The
    websocket is usually a thin wrapper around a polling API, which means that polling
    every 5-10 seconds is typically more than enough for modelling real-time pressure.

    Parameters
    ----------
    config:
        Settings that control how the client connects to the API. The defaults work for
        the public site, but credentials (session cookies or API tokens) can be injected
        using the ``headers`` or ``cookies`` fields.
    session:
        Optional ``requests.Session``. Re-using a session is recommended so that TLS
        handshakes and cookies are cached.
    """

    def __init__(
        self,
        config: Optional[InplayGuruClientConfig] = None,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config or InplayGuruClientConfig()
        self._session = session or requests.Session()

        if self._config.headers:
            self._session.headers.update(dict(self._config.headers))
        if self._config.cookies:
            self._session.cookies.update(dict(self._config.cookies))

    # ------------------------------------------------------------------
    # HTTP helpers
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self._session.request(
            method,
            url,
            timeout=self._config.timeout_seconds,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Public API
    def list_live_matches(self) -> List[Dict[str, Any]]:
        """Return live matches currently tracked by InplayGuru.

        The endpoint usually responds with a list of matches where each match exposes at
        least the following fields: ``id``, ``competition``, ``homeTeam``, ``awayTeam`` and
        ``status``.
        """

        return self._request("GET", "/matches/live")

    def fetch_match_metrics(self, match_id: str) -> Dict[str, Any]:
        """Return the latest snapshot of metrics for ``match_id``.

        The payload includes the raw values shown on the live dashboard (dangerous
        attacks, shots, xG, win probability, etc.). The exact schema depends on the
        competition, so callers should prefer the feature engineering helpers from
        :mod:`inplayguru_analytics.features` which smooth out the differences.
        """

        return self._request("GET", f"/matches/{match_id}/metrics")

    def fetch_metrics_history(self, match_id: str) -> List[Dict[str, Any]]:
        """Return a chronological list of metrics for ``match_id``.

        Some use cases—like training a predictive model—require time-series data. The
        history endpoint returns the same metrics as :meth:`fetch_match_metrics`, but
        bucketed by minute. When the API does not offer a dedicated history endpoint we
        fall back to the live ``updates`` endpoint that powers the frontend widget.
        """

        try:
            return self._request("GET", f"/matches/{match_id}/metrics/history")
        except requests.HTTPError as exc:  # pragma: no cover - defensive guard
            if exc.response is not None and exc.response.status_code == 404:
                return self._request("GET", f"/matches/{match_id}/updates")
            raise

    # ------------------------------------------------------------------
    # Utility methods
    @staticmethod
    def normalize_metric_payload(payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Convert raw API payloads to a consistent dictionary.

        ``InplayGuru`` occasionally changes its key naming conventions (for example,
        ``attacks.total`` vs ``totalAttacks``). The normalisation step performs a best
        effort mapping so downstream code can remain stable.
        """

        aliases = {
            "dangerous_attacks": [
                "dangerousAttacks",
                "dangerous_attacks",
                "attacks.dangerous",
            ],
            "total_attacks": ["attacks.total", "totalAttacks", "attacks"],
            "possession_home": ["possession.home", "possessionHome"],
            "possession_away": ["possession.away", "possessionAway"],
            "shots_on_target": ["shots.onTarget", "shotsOnTarget"],
            "shots_total": ["shots.total", "shots"],
            "xg": ["xg", "expectedGoals"],
            "minute": ["minute", "clock"],
        }

        normalized: MutableMapping[str, Any] = {}
        for canonical, options in aliases.items():
            for key in options:
                if key in payload:
                    normalized[canonical] = payload[key]
                    break

        # Preserve original keys in case feature engineering needs something custom.
        for key, value in payload.items():
            normalized.setdefault(key, value)

        return normalized

    @staticmethod
    def iter_history_snapshots(payload: Iterable[Mapping[str, Any]]) -> Iterable[Dict[str, Any]]:
        """Yield normalized snapshots from the history payload."""

        for entry in payload:
            yield dict(InplayGuruClient.normalize_metric_payload(entry))


__all__ = ["InplayGuruClient", "InplayGuruClientConfig"]
