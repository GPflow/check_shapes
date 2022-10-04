from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import numpy as np

N_FEATURES = 20
N_TRAINING_ROWS = 5_000
N_TEST_ROWS = 100

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


class Model(ABC):

    # training_features: [n_rows, n_features]
    # training_targets: [n_rows, 1]
    @abstractmethod
    def train(self, training_features: np.ndarray, training_targets: np.ndarray) -> None:
        ...

    # test_features: [n_rows, n_features]
    # return: [n_rows, 1]
    @abstractmethod
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[np.ndarray] = None  # [n_features]

    # weights: [n_features]
    def set_weights(self, weights: np.ndarray) -> None:
        self._weights = weights

    # Inherit shapes.
    def train(self, training_features: np.ndarray, training_targets: np.ndarray) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        # weights: [n_features]
        # return: [n_features]
        def loss_grads(weights: np.ndarray) -> np.ndarray:
            pred = self._predict(weights, training_features)  # [n_rows, 1]
            grads = 2 * (pred - training_targets) * training_features  # [n_rows, n_features]
            return np.mean(grads, axis=0)  # [n_features]

        # weights: [n_features]
        # return: [n_features]
        def step(weights: np.ndarray) -> np.ndarray:
            return weights - LEARNING_RATE * loss_grads(weights)

        n_features = training_features.shape[-1]
        weights = np.zeros((n_features,))
        for _ in range(N_ITERATIONS):
            weights = step(weights)

        self._weights = weights

    # Inherit shapes.
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    # weights: [n_features]
    # test_fetaures: [n_rows, n_features]
    # return: [n_rows, 1]
    def _predict(weights: np.ndarray, test_features: np.ndarray) -> np.ndarray:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: np.ndarray  # [n_training_rows, n_features]
    training_targets: np.ndarray  # [n_training_rows, 1]
    test_features: np.ndarray  # [n_test_rows, n_features]
    test_targets: np.ndarray  # [n_test_rows, 1]


def create_data() -> TestData:
    rng = np.random.default_rng(42)
    model = LinearModel()
    model.set_weights(np.arange(N_FEATURES))
    training_features = rng.random(size=[N_TRAINING_ROWS, N_FEATURES])
    training_targets = model.predict(training_features) + rng.normal(
        0.0, 1.0, size=[N_TRAINING_ROWS, 1]
    )
    test_features = rng.random(size=[N_TEST_ROWS, N_FEATURES])
    test_targets = model.predict(test_features)
    return TestData(training_features, training_targets, test_features, test_targets)


def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets - test_predictions
    result = {}
    result["rmse"] = np.sqrt(np.mean(err ** 2))
    return result


def main() -> None:
    before = perf_counter()
    data = create_data()
    model = LinearModel()
    model.train(data.training_features, data.training_targets)
    metrics = dict(evaluate(model, data))
    after = perf_counter()
    metrics["time (s)"] = after - before
    for key, value in metrics.items():
        print(key, value)


if __name__ == "__main__":
    main()
