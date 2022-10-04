from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random

from shape_checking_study.check_shapes import check_shapes, inherit_check_shapes

N_FEATURES = 100
N_TRAINING_ROWS = 20_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


class Model(ABC):
    @abstractmethod
    @check_shapes(
        ("training_features", ["n_rows", "n_features"]),
        ("training_targets", ["n_rows", 1]),
    )
    def train(self, training_features: np.ndarray, training_targets: np.ndarray) -> None:
        ...

    @abstractmethod
    @check_shapes(
        ("test_features", ["n_rows", "n_features"]),
        ("return", ["n_rows", 1]),
    )
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[np.ndarray] = None

    @check_shapes(
        ("weights", ["n_features"]),
    )
    def set_weights(self, weights: np.ndarray) -> None:
        self._weights = weights

    @inherit_check_shapes
    def train(self, training_features: np.ndarray, training_targets: np.ndarray) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        @check_shapes(
            ("weights", ["n_features"]),
            ("return", []),
        )
        def loss(weights: np.ndarray) -> np.ndarray:
            pred = self._predict(weights, training_features)  # [n_rows, 1]
            err = pred - training_targets  # [n_rows, 1]
            return jnp.mean(err ** 2)

        loss_grads = grad(loss)

        @jit
        @check_shapes(
            ("weights", ["n_features"]),
            ("return", ["n_features"]),
        )
        def step(weights: np.ndarray) -> np.ndarray:
            return weights - LEARNING_RATE * loss_grads(weights)

        n_features = training_features.shape[-1]
        weights = np.zeros((n_features,))

        for _ in range(N_ITERATIONS):
            weights = step(weights)

        self._weights = weights

    @inherit_check_shapes
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    @check_shapes(
        ("weights", ["n_features"]),
        ("test_features", ["n_rows", "n_features"]),
        ("return", ["n_rows", 1]),
    )
    def _predict(weights: np.ndarray, test_features: np.ndarray) -> np.ndarray:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: np.ndarray
    training_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray

    @check_shapes(
        ("self.training_features", ["n_training_rows", "n_features"]),
        ("self.training_targets", ["n_training_rows", 1]),
        ("self.test_features", ["n_test_rows", "n_features"]),
        ("self.test_targets", ["n_test_rows", 1]),
    )
    def __post_init__(self) -> None:
        pass


def create_data() -> TestData:
    key = random.PRNGKey(42)
    key1, key2, key3 = random.split(key, 3)
    model = LinearModel()
    model.set_weights(jnp.arange(N_FEATURES))
    training_features = random.uniform(key1, [N_TRAINING_ROWS, N_FEATURES])
    training_targets = model.predict(training_features) + random.normal(
        key2, [N_TRAINING_ROWS, 1]
    )
    test_features = random.uniform(key3, [N_TEST_ROWS, N_FEATURES])
    test_targets = model.predict(test_features)
    return TestData(training_features, training_targets, test_features, test_targets)


def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets - test_predictions
    result = {}
    result["rmse"] = jnp.sqrt(jnp.mean(err ** 2))
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
