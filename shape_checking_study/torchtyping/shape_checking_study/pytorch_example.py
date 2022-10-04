# For some reason mypy is *super* slow with pytorch.
# type: ignore
# pylint: disable=no-member
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import torch
from typeguard import typechecked

from torchtyping import TensorType, patch_typeguard

patch_typeguard()

N_FEATURES = 100
N_TRAINING_ROWS = 20_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


class Model(ABC):
    @abstractmethod
    def train(
        self,
        training_features: TensorType["n_rows", "n_features"],
        training_targets: TensorType["n_rows", 1],
    ) -> None:
        ...

    @abstractmethod
    def predict(
        self, test_features: TensorType["n_rows", "n_features"]
    ) -> TensorType["n_rows", 1]:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[TensorType["n_features"]] = None

    def set_weights(self, weights: TensorType["n_features"]) -> None:
        self._weights = weights

    @typechecked
    def train(
        self,
        training_features: TensorType["n_rows", "n_features"],
        training_targets: TensorType["n_rows", 1],
    ) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        @typechecked
        def loss(weights: TensorType["n_features"]) -> TensorType[()]:
            pred: TensorType["n_rows", 1] = self._predict(weights, training_features)
            err: TensorType["n_rows", 1] = pred - training_targets
            return torch.mean(err ** 2)

        @typechecked
        def step(weights: TensorType["n_features"]) -> None:
            loss(weights).backward()
            with torch.no_grad():
                weights -= LEARNING_RATE * weights.grad
                weights.grad = None

        n_features = training_features.shape[-1]
        weights = torch.zeros((n_features,), requires_grad=True)

        for _ in range(N_ITERATIONS):
            step(weights)

        self._weights = weights.detach().clone()

    @typechecked
    def predict(
        self, test_features: TensorType["n_rows", "n_features"]
    ) -> TensorType["n_rows", 1]:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    def _predict(
        weights: TensorType["n_features"], test_features: TensorType["n_rows", "n_features"]
    ) -> TensorType["n_rows", 1]:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: TensorType["n_training_rows", "n_features"]
    training_targets: TensorType["n_training_rows", 1]
    test_features: TensorType["n_test_rows", "n_features"]
    test_targets: TensorType["n_test_rows", 1]


@typechecked
def create_data() -> TestData:
    rng = torch.Generator()
    rng.manual_seed(42)
    model = LinearModel()
    model.set_weights(torch.arange(0, N_FEATURES, dtype=torch.float32))
    training_features = torch.rand([N_TRAINING_ROWS, N_FEATURES], generator=rng)
    training_targets = model.predict(training_features) + torch.randn(
        [N_TRAINING_ROWS, 1], generator=rng
    )
    test_features = torch.rand([N_TEST_ROWS, N_FEATURES], generator=rng)
    test_targets = model.predict(test_features)
    return TestData(training_features, training_targets, test_features, test_targets)


@typechecked
def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets - test_predictions
    result = {}
    result["rmse"] = torch.sqrt(torch.mean(err ** 2)).item()
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
