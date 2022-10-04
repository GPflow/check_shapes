from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, NewType, Optional, cast

import jax.numpy as jnp
import numpy as np
import tensor_annotations.jax as tjax
from jax import grad, jit, random  # type: ignore
from tensor_annotations import axes

N_FEATURES = 100
N_TRAINING_ROWS = 20_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


NRows = NewType("NRows", axes.Axis)
NFeatures = NewType("NFeatures", axes.Axis)
N1 = NewType("N1", axes.Axis)


class Model(ABC):
    @abstractmethod
    def train(
        self,
        training_features: tjax.Array2[NRows, NFeatures],
        training_targets: tjax.Array2[NRows, N1],
    ) -> None:
        ...

    @abstractmethod
    def predict(self, test_features: tjax.Array2[NRows, NFeatures]) -> tjax.Array2[NRows, N1]:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[tjax.Array1[NFeatures]] = None

    def set_weights(self, weights: tjax.Array1[NFeatures]) -> None:
        self._weights = weights

    def train(
        self,
        training_features: tjax.Array2[NRows, NFeatures],
        training_targets: tjax.Array2[NRows, N1],
    ) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        def loss(weights: tjax.Array1[NFeatures]) -> tjax.Array0:
            pred: tjax.Array2[NRows, N1] = self._predict(weights, training_features)
            err: tjax.Array2[NRows, N1] = pred - training_targets
            return jnp.mean(err ** 2)

        loss_grads = grad(loss)

        @jit  # type: ignore
        def step(weights: tjax.Array1[NFeatures]) -> tjax.Array1[NFeatures]:
            return cast(tjax.Array1[NFeatures], weights - LEARNING_RATE * loss_grads(weights))

        n_features = training_features.shape[-1]
        weights = np.zeros((n_features,))

        for _ in range(N_ITERATIONS):
            weights = step(weights)

        self._weights = weights

    def predict(self, test_features: tjax.Array2[NRows, NFeatures]) -> tjax.Array2[NRows, N1]:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    def _predict(
        weights: tjax.Array1[NFeatures], test_features: tjax.Array2[NRows, NFeatures]
    ) -> tjax.Array2[NRows, N1]:
        return cast(tjax.Array2[NRows, N1], test_features @ weights[:, None])


@dataclass
class TestData:
    training_features: tjax.Array2[NRows, NFeatures]
    training_targets: tjax.Array2[NRows, N1]
    test_features: tjax.Array2[NRows, NFeatures]
    test_targets: tjax.Array2[NRows, N1]


def create_data() -> TestData:
    key = random.PRNGKey(42)
    key1, key2, key3 = random.split(key, 3)
    model = LinearModel()
    model.set_weights(jnp.arange(N_FEATURES))
    training_features = cast(
        tjax.Array2[NRows, NFeatures], random.uniform(key1, [N_TRAINING_ROWS, N_FEATURES])
    )
    training_targets = model.predict(training_features) + cast(
        tjax.Array2[NRows, N1], random.normal(key2, [N_TRAINING_ROWS, 1])
    )
    test_features = cast(
        tjax.Array2[NRows, NFeatures], random.uniform(key3, [N_TEST_ROWS, N_FEATURES])
    )
    test_targets = model.predict(test_features)
    return TestData(training_features, training_targets, test_features, test_targets)


def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets - test_predictions
    result = {}
    result["rmse"] = float(jnp.sqrt(jnp.mean(err ** 2)))  # type: ignore
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
