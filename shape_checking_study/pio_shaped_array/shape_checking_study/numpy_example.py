from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import numpy as np
from pio_learning_utilities.arrays.numpy import ShapedNdArray
from pio_learning_utilities.arrays.shaped_array import fixed_dimension, variable_dimension

N_FEATURES = 20
N_TRAINING_ROWS = 5_000
N_TEST_ROWS = 100

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


class Features(ShapedNdArray):

    n_rows = variable_dimension()
    n_features = variable_dimension()


class Targets(ShapedNdArray):

    n_rows = variable_dimension()
    n_targets = fixed_dimension(1)


class Weights(ShapedNdArray):

    n_features = variable_dimension()


class Model(ABC):
    @abstractmethod
    def train(self, training_features: Features, training_targets: Targets) -> None:
        ...

    @abstractmethod
    def predict(self, test_features: Features) -> Targets:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[Weights] = None

    def set_weights(self, weights: Weights) -> None:
        self._weights = weights

    def train(self, training_features: Features, training_targets: Targets) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        def loss_grads(weights: Weights) -> Weights:
            pred: Targets = self._predict(weights, training_features)
            grads = Features(
                2 * (pred.array - training_targets.array) * training_features.array,
                n_rows=pred.n_rows,
                n_features=weights.n_features,
            )
            return Weights(np.mean(grads.array, axis=0), n_features=weights.n_features)

        def step(weights: Weights) -> Weights:
            return Weights(
                weights.array - LEARNING_RATE * loss_grads(weights).array,
                n_features=weights.n_features,
            )

        n_features = training_features.n_features
        weights = Weights(np.zeros((n_features,)), n_features=n_features)
        for _ in range(N_ITERATIONS):
            weights = step(weights)

        self._weights = weights

    def predict(self, test_features: Features) -> Targets:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    def _predict(weights: Weights, test_features: Features) -> Targets:
        return Targets(
            test_features.array @ weights.array[:, None], n_rows=test_features.n_rows
        )


@dataclass
class TestData:
    training_features: Features
    training_targets: Targets
    test_features: Features
    test_targets: Targets


def create_data() -> TestData:
    rng = np.random.default_rng(42)
    model = LinearModel()
    model.set_weights(Weights(np.arange(N_FEATURES), n_features=N_FEATURES))
    training_features = Features(
        rng.random(size=[N_TRAINING_ROWS, N_FEATURES]),
        n_rows=N_TRAINING_ROWS,
        n_features=N_FEATURES,
    )
    training_targets = Targets(
        model.predict(training_features).array
        + rng.normal(0.0, 1.0, size=[N_TRAINING_ROWS, 1]),
        n_rows=N_TRAINING_ROWS,
    )
    test_features = Features(
        rng.random(size=[N_TEST_ROWS, N_FEATURES]), n_rows=N_TEST_ROWS, n_features=N_FEATURES
    )
    test_targets = model.predict(test_features)
    return TestData(
        training_features,
        training_targets,
        test_features,
        test_targets,
    )


def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets.array - test_predictions.array
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
