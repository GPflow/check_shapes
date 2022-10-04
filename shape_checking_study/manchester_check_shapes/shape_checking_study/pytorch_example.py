# For some reason mypy is *super* slow with pytorch.
# type: ignore
# pylint: disable=no-member
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import torch

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
    def train(self, training_features: torch.Tensor, training_targets: torch.Tensor) -> None:
        ...

    @abstractmethod
    @check_shapes(
        ("test_features", ["n_rows", "n_features"]),
        ("return", ["n_rows", 1]),
    )
    def predict(self, test_features: torch.Tensor) -> torch.Tensor:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[torch.Tensor] = None

    @check_shapes(
        ("weights", ["n_features"]),
    )
    def set_weights(self, weights: torch.Tensor) -> None:
        self._weights = weights

    @inherit_check_shapes
    def train(self, training_features: torch.Tensor, training_targets: torch.Tensor) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        @check_shapes(
            ("weights", ["n_features"]),
            ("return", []),
        )
        def loss(weights: torch.Tensor) -> torch.Tensor:
            pred = self._predict(weights, training_features)
            err = pred - training_targets
            return torch.mean(err ** 2)

        @check_shapes(
            ("weights", ["n_features"]),
        )
        def step(weights: torch.Tensor) -> None:
            loss(weights).backward()
            with torch.no_grad():
                weights -= LEARNING_RATE * weights.grad
                weights.grad = None

        n_features = training_features.shape[-1]
        weights = torch.zeros((n_features,), requires_grad=True)

        for _ in range(N_ITERATIONS):
            step(weights)

        self._weights = weights.detach().clone()

    @inherit_check_shapes
    def predict(self, test_features: torch.Tensor) -> torch.Tensor:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    @check_shapes(
        ("weights", ["n_features"]),
        ("test_features", ["n_rows", "n_features"]),
        ("return", ["n_rows", 1]),
    )
    def _predict(weights: torch.Tensor, test_features: torch.Tensor) -> torch.Tensor:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: torch.Tensor
    training_targets: torch.Tensor
    test_features: torch.Tensor
    test_targets: torch.Tensor

    @check_shapes(
        ("self.training_features", ["n_training_rows", "n_features"]),
        ("self.training_targets", ["n_training_rows", 1]),
        ("self.test_features", ["n_test_rows", "n_features"]),
        ("self.test_targets", ["n_test_rows", 1]),
    )
    def __post_init__(self) -> None:
        pass


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
