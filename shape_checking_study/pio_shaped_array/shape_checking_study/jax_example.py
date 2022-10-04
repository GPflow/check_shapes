from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random
from pio_learning_utilities.arrays.numpy import ShapedNdArray
from pio_learning_utilities.arrays.shaped_array import fixed_dimension, variable_dimension

N_FEATURES = 100
N_TRAINING_ROWS = 20_000
N_TEST_ROWS = 10_000

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


class Loss(ShapedNdArray):

    pass


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
        n_features = training_features.n_features

        # ShapedNdArray doesn't seem to work with jax.grad.
        def loss(weights: np.ndarray) -> np.ndarray:
            pred = self._predict(Weights(weights, n_features=n_features), training_features)
            err = pred.array - training_targets.array
            return jnp.mean(err ** 2)

        loss_grads = grad(loss)

        @jit
        # ShapedNdArray doesn't seem to work with jax.jit.
        def step(weights: np.ndarray) -> np.ndarray:
            return weights - LEARNING_RATE * loss_grads(weights)

        weights = Weights(np.zeros((n_features,)), n_features=n_features)

        for _ in range(N_ITERATIONS):
            weights = Weights(step(weights.array), n_features=n_features)

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
    key = random.PRNGKey(42)
    key1, key2, key3 = random.split(key, 3)
    model = LinearModel()
    model.set_weights(Weights(jnp.arange(N_FEATURES), n_features=N_FEATURES))
    training_features = Features(
        random.uniform(key1, [N_TRAINING_ROWS, N_FEATURES]),
        n_rows=N_TRAINING_ROWS,
        n_features=N_FEATURES,
    )
    training_targets = Targets(
        model.predict(training_features).array + random.normal(key2, [N_TRAINING_ROWS, 1]),
        n_rows=N_TRAINING_ROWS,
    )
    test_features = Features(
        random.uniform(key3, [N_TEST_ROWS, N_FEATURES]),
        n_rows=N_TEST_ROWS,
        n_features=N_FEATURES,
    )
    test_targets = model.predict(test_features)
    return TestData(training_features, training_targets, test_features, test_targets)


def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets.array - test_predictions.array
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
