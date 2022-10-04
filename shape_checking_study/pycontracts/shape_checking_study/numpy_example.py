from abc import abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import numpy as np
from contracts import ContractsMeta, contract

N_FEATURES = 20
N_TRAINING_ROWS = 5_000
N_TEST_ROWS = 100

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


class Model(metaclass=ContractsMeta):
    @abstractmethod
    @contract(training_features="array[RxF]", training_targets="array[Rx1]")
    def train(self, training_features: np.ndarray, training_targets: np.ndarray) -> None:
        ...

    @abstractmethod
    @contract(test_features="array[RxF]", returns="array[Rx1]")
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[np.ndarray] = None

    @contract(weights="array[F]")
    def set_weights(self, weights: np.ndarray) -> None:
        self._weights = weights

    @contract(training_features="array[RxF]", training_targets="array[Rx1]")
    def train(self, training_features: np.ndarray, training_targets: np.ndarray) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        @contract(weights="array[F]", returns="array[F]")
        def loss_grads(weights: np.ndarray) -> np.ndarray:
            pred = self._predict(weights, training_features)
            grads = 2 * (pred - training_targets) * training_features
            return np.mean(grads, axis=0)

        @contract(weights="array[F]", returns="array[F]")
        def step(weights: np.ndarray) -> np.ndarray:
            return weights - LEARNING_RATE * loss_grads(weights)

        n_features = training_features.shape[-1]
        weights = np.zeros((n_features,))
        for _ in range(N_ITERATIONS):
            weights = step(weights)

        self._weights = weights

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    @contract(weights="array[F]", test_features="array[RxF]", returns="array[Rx1]")
    def _predict(weights: np.ndarray, test_features: np.ndarray) -> np.ndarray:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    trainingfeatures: np.ndarray
    trainingtargets: np.ndarray
    testfeatures: np.ndarray
    testtargets: np.ndarray

    @contract(
        self="attr(trainingfeatures:array[RxF])"
        ",attr(trainingtargets:array[Rx1])"
        ",attr(testfeatures:array[TxF])"
        ",attr(testtargets:array[Tx1])"
    )
    def __post_init__(self) -> None:
        pass


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
    test_predictions = model.predict(data.testfeatures)
    err = data.testtargets - test_predictions
    result = {}
    result["rmse"] = np.sqrt(np.mean(err ** 2))
    return result


def main() -> None:
    before = perf_counter()
    data = create_data()
    model = LinearModel()
    model.train(data.trainingfeatures, data.trainingtargets)
    metrics = dict(evaluate(model, data))
    after = perf_counter()
    metrics["time (s)"] = after - before
    for key, value in metrics.items():
        print(key, value)


if __name__ == "__main__":
    main()
