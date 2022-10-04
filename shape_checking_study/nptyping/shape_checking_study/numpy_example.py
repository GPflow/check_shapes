from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Optional

import numpy as np

from nptyping import NDArray

N_FEATURES = 20
N_TRAINING_ROWS = 5_000
N_TEST_ROWS = 100

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


FeatureTensor = NDArray[Any, Any]
TargetTensor = NDArray[Any, 1]
WeightsTensor = NDArray[Any]


class Model(ABC):
    @abstractmethod
    def train(self, training_features: FeatureTensor, training_targets: TargetTensor) -> None:
        ...

    @abstractmethod
    def predict(self, test_features: FeatureTensor) -> TargetTensor:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[WeightsTensor] = None

    def set_weights(self, weights: NDArray[Any]) -> None:
        assert isinstance(weights, WeightsTensor)
        self._weights = weights

    def train(self, training_features: FeatureTensor, training_targets: TargetTensor) -> None:
        assert isinstance(training_features, FeatureTensor)
        assert isinstance(training_targets, TargetTensor)
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        def loss_grads(weights: WeightsTensor) -> WeightsTensor:
            assert isinstance(weights, WeightsTensor)
            pred: TargetTensor = self._predict(weights, training_features)
            assert isinstance(pred, TargetTensor)
            grads: FeatureTensor = 2 * (pred - training_targets) * training_features
            assert isinstance(grads, FeatureTensor)
            result = np.mean(grads, axis=0)
            assert isinstance(result, WeightsTensor)
            return result

        def step(weights: NDArray[Any]) -> NDArray[Any]:
            return weights - LEARNING_RATE * loss_grads(weights)

        n_features = training_features.shape[-1]
        weights = np.zeros((n_features,))
        for _ in range(N_ITERATIONS):
            weights = step(weights)

        self._weights = weights

    def predict(self, test_features: NDArray[Any, Any]) -> NDArray[Any, 1]:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    def _predict(weights: NDArray[Any], test_features: NDArray[Any, Any]) -> NDArray[Any, 1]:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: NDArray[Any, Any]
    training_targets: NDArray[Any, Any]
    test_features: NDArray[Any, Any]
    test_targets: NDArray[Any, Any]


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
    a: NDArray[3] = np.zeros((3,))
    assert isinstance(a, NDArray[3])
    b: NDArray[7] = a
    assert isinstance(b, NDArray[7])

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
