import math
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


EPS = 1e-6


def clip_probabilities(probabilities):
    probs = np.asarray(probabilities, dtype=float)
    return np.clip(probs, EPS, 1.0 - EPS)


def probs_to_logits(probabilities):
    probs = clip_probabilities(probabilities)
    return np.log(probs / (1.0 - probs))


def logits_to_probs(logits):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def binary_log_loss(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    probs = clip_probabilities(probabilities)
    return -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))


class BaseCalibrator:
    method_name = "base"

    def fit(self, probabilities, labels):
        raise NotImplementedError

    def transform(self, probabilities):
        raise NotImplementedError

    def to_metadata(self):
        return {"method": self.method_name}


class IdentityCalibrator(BaseCalibrator):
    method_name = "identity"

    def fit(self, probabilities, labels):
        return self

    def transform(self, probabilities):
        return np.asarray(probabilities, dtype=float)


class IsotonicCalibrator(BaseCalibrator):
    method_name = "isotonic_regression"

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probabilities, labels):
        self.model.fit(np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=float))
        return self

    def transform(self, probabilities):
        return np.asarray(self.model.transform(np.asarray(probabilities, dtype=float)), dtype=float)

    def to_metadata(self):
        return {
            "method": self.method_name,
            "X_thresholds": self.model.X_thresholds_.tolist(),
            "y_thresholds": self.model.y_thresholds_.tolist(),
        }


class PlattScalingCalibrator(BaseCalibrator):
    method_name = "platt"

    def __init__(self):
        self.model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)

    def fit(self, probabilities, labels):
        logits = probs_to_logits(probabilities).reshape(-1, 1)
        self.model.fit(logits, np.asarray(labels, dtype=int))
        return self

    def transform(self, probabilities):
        logits = probs_to_logits(probabilities).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]

    def to_metadata(self):
        return {
            "method": self.method_name,
            "coef": self.model.coef_.ravel().tolist(),
            "intercept": self.model.intercept_.ravel().tolist(),
        }


@dataclass
class TemperatureSearchResult:
    log_temperature: float
    loss: float


class TemperatureScalingCalibrator(BaseCalibrator):
    method_name = "temperature"

    def __init__(self):
        self.temperature = 1.0

    @staticmethod
    def _loss_for_temperature(temperature, logits, labels):
        scaled_probs = logits_to_probs(logits / temperature)
        return binary_log_loss(labels, scaled_probs)

    def _search(self, logits, labels, low=-4.0, high=4.0, steps=161, rounds=3):
        best = TemperatureSearchResult(log_temperature=0.0, loss=math.inf)
        current_low = low
        current_high = high
        for _ in range(rounds):
            candidates = np.linspace(current_low, current_high, steps)
            for log_temperature in candidates:
                temperature = float(np.exp(log_temperature))
                loss = self._loss_for_temperature(temperature, logits, labels)
                if loss < best.loss:
                    best = TemperatureSearchResult(log_temperature=float(log_temperature), loss=float(loss))
            step = (current_high - current_low) / (steps - 1)
            current_low = best.log_temperature - step
            current_high = best.log_temperature + step
        return best

    def fit(self, probabilities, labels):
        logits = probs_to_logits(probabilities)
        best = self._search(logits, np.asarray(labels, dtype=float))
        self.temperature = float(np.exp(best.log_temperature))
        return self

    def transform(self, probabilities):
        logits = probs_to_logits(probabilities)
        return logits_to_probs(logits / self.temperature)

    def to_metadata(self):
        return {"method": self.method_name, "temperature": self.temperature}


class BetaCalibrationCalibrator(BaseCalibrator):
    method_name = "beta_calibration"

    def __init__(self):
        self.model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)

    def fit(self, probabilities, labels):
        probs = clip_probabilities(probabilities)
        features = np.stack([np.log(probs), np.log(1.0 - probs)], axis=1)
        self.model.fit(features, np.asarray(labels, dtype=int))
        return self

    def transform(self, probabilities):
        probs = clip_probabilities(probabilities)
        features = np.stack([np.log(probs), np.log(1.0 - probs)], axis=1)
        return self.model.predict_proba(features)[:, 1]

    def to_metadata(self):
        return {
            "method": self.method_name,
            "coef": self.model.coef_.ravel().tolist(),
            "intercept": self.model.intercept_.ravel().tolist(),
        }


class HistogramBinningCalibrator(BaseCalibrator):
    method_name = "histogram_binning"

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_values = None

    def fit(self, probabilities, labels):
        probs = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels, dtype=float)
        self.bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_values = np.zeros(self.n_bins, dtype=float)
        for i in range(self.n_bins):
            left = self.bin_edges[i]
            right = self.bin_edges[i + 1]
            if i == self.n_bins - 1:
                mask = (probs >= left) & (probs <= right)
            else:
                mask = (probs >= left) & (probs < right)
            if np.any(mask):
                self.bin_values[i] = float(np.mean(labels[mask]))
            else:
                center = (left + right) / 2.0
                self.bin_values[i] = center
        return self

    def transform(self, probabilities):
        probs = np.asarray(probabilities, dtype=float)
        bin_indices = np.digitize(probs, self.bin_edges[1:-1], right=False)
        return self.bin_values[bin_indices]

    def to_metadata(self):
        return {
            "method": self.method_name,
            "n_bins": self.n_bins,
            "bin_edges": self.bin_edges.tolist(),
            "bin_values": self.bin_values.tolist(),
        }


def build_calibrator(method_name):
    normalized = method_name.strip().lower()
    if normalized == "identity":
        return IdentityCalibrator()
    if normalized in {"isotonic", "isotonic_regression"}:
        return IsotonicCalibrator()
    if normalized in {"platt", "platt_scaling", "platt_scaling_logistic"}:
        return PlattScalingCalibrator()
    if normalized in {"temperature", "temperature_scaling", "temperature_scaling_logistic", "ts"}:
        return TemperatureScalingCalibrator()
    if normalized in {"beta", "beta_calibration"}:
        return BetaCalibrationCalibrator()
    if normalized in {"histogram", "histogram_binning"}:
        return HistogramBinningCalibrator()
    raise ValueError(f"Unknown calibration method: {method_name}")
