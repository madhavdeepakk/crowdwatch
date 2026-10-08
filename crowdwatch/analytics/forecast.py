"""Short-horizon occupancy forecast.

A threshold alarm tells you a zone is full. By then stewards are already
behind. What an operator needs is "the atrium will be full in about two
minutes", while there is still time to close a door or redirect a queue.

The method is deliberately simple and explainable: fit a straight line to the
last ``window_s`` seconds of the count and extend it. Two things keep it from
crying wolf:

* a rise only counts if it is larger than chance. People come and go all the
  time, so the count wanders like a random walk even when nothing is
  happening. For a walk with ``churn`` arrivals-plus-departures per second,
  the slope fitted over a window of W seconds has a standard deviation of
  sqrt(6/5 * churn / W). The zone's own turnover is measured from the
  tracks, and a trend must exceed that noise by ``min_z`` standard
  deviations before it is believed;
* the trend is damped. A crowd that gained 20 people in the last minute rarely
  gains 40 in the next two, because rises level off. So the slope is assumed
  to fade with time constant ``damping_s`` (the damped-trend idea of Gardner
  and McKenzie, 1985), which stops a short burst being extrapolated into a
  disaster.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trend:
    level: float            # fitted count right now
    slope: float            # people per second
    slope_stderr: float
    span_s: float           # how much history the fit used
    samples: int
    damping_s: float = 90.0  # how quickly the trend is assumed to fade

    def _reach(self, horizon_s: float) -> float:
        """Effective seconds of full-strength trend within the horizon."""
        return self.damping_s * (1.0 - math.exp(-horizon_s / self.damping_s))

    @property
    def per_minute(self) -> float:
        return self.slope * 60.0

    @property
    def t_stat(self) -> float:
        if self.slope_stderr <= 1e-12:
            return float("inf") if abs(self.slope) > 1e-9 else 0.0
        return self.slope / self.slope_stderr

    def predict(self, horizon_s: float) -> float:
        return max(0.0, self.level + self.slope * self._reach(horizon_s))

    def eta(self, threshold: float) -> Optional[float]:
        """Seconds until the damped trend reaches ``threshold`` (None if it never does)."""
        if self.level >= threshold:
            return 0.0
        if self.slope <= 1e-9:
            return None
        needed = (threshold - self.level) / (self.slope * self.damping_s)
        if needed >= 1.0:
            return None          # the trend fades out before getting there
        return -self.damping_s * math.log(1.0 - needed)


class TrendForecaster:
    """Sliding-window linear trend over a regularly resampled series."""

    KEEP_S = 120.0      # history kept, so slower trends can be fitted too

    def __init__(self, window_s: float = 45.0, sample_s: float = 1.0, damping_s: float = 90.0):
        self.window_s = window_s
        self.damping_s = damping_s
        self.sample_s = sample_s
        self.keep_s = max(self.KEEP_S, window_s)
        self._points: deque[tuple[float, float]] = deque()
        self._bucket_start: Optional[float] = None
        self._bucket_sum = 0.0
        self._bucket_n = 0

    def add(self, t: float, value: float) -> None:
        if self._bucket_start is None:
            self._bucket_start = t
        if t - self._bucket_start >= self.sample_s and self._bucket_n:
            mid = self._bucket_start + self.sample_s / 2
            self._points.append((mid, self._bucket_sum / self._bucket_n))
            self._bucket_start, self._bucket_sum, self._bucket_n = t, 0.0, 0
            while self._points and mid - self._points[0][0] > self.keep_s:
                self._points.popleft()
        self._bucket_sum += value
        self._bucket_n += 1

    def trend(self, window_s: Optional[float] = None) -> Optional[Trend]:
        """Fit the last ``window_s`` seconds (the configured window by default)."""
        if len(self._points) < 5:
            return None
        data = np.asarray(self._points)
        data = data[data[:, 0] >= data[-1, 0] - (window_s or self.window_s) - 1e-9]
        n = len(data)
        if n < 5:
            return None
        t, y = data[:, 0], data[:, 1]
        t_last = t[-1]
        x = t - t_last
        x_mean, y_mean = x.mean(), y.mean()
        sxx = float(((x - x_mean) ** 2).sum())
        if sxx <= 1e-9:
            return None
        slope = float(((x - x_mean) * (y - y_mean)).sum() / sxx)
        intercept = y_mean - slope * x_mean          # fitted value at x = 0, i.e. now
        residual = y - (intercept + slope * x)
        dof = max(n - 2, 1)
        sigma2 = float((residual**2).sum() / dof)
        # Counts are whole numbers, so even a perfectly steady zone carries
        # rounding noise. Floor the variance so tiny slopes never look certain.
        sigma2 = max(sigma2, 0.25)
        stderr = (sigma2 / sxx) ** 0.5
        return Trend(level=float(intercept), slope=slope, slope_stderr=stderr,
                     span_s=float(t[-1] - t[0]), samples=n, damping_s=self.damping_s)

    def rise_z(self, trend: Optional[Trend], churn_per_s: float = 0.0) -> float:
        """How many standard deviations the rise stands above ordinary coming and going."""
        if trend is None or trend.slope <= 0 or trend.span_s < 0.5 * self.window_s:
            return 0.0
        walk_sd = (1.2 * max(churn_per_s, 0.0) / trend.span_s) ** 0.5
        return trend.slope / max(walk_sd, trend.slope_stderr, 1e-9)

    def is_rising_clearly(self, trend: Optional[Trend], churn_per_s: float = 0.0,
                          min_z: float = 1.5) -> bool:
        return self.rise_z(trend, churn_per_s) >= min_z
