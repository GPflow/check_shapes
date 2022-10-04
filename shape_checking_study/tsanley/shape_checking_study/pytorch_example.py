# For some reason mypy is *super* slow with pytorch.
# type: ignore
# pylint: disable=no-member
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import torch
from tsalib import dim_vars

from tsanley.dynamic import init_analyzer

N_FEATURES = 100
N_TRAINING_ROWS = 20_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2

R, A, E, F, O = dim_vars(
    "Rows(r):0" " TrainingRows:0" " TestRows:0" " Features(f):0" " Outputs(O):1"
)


class Model(ABC):
    @abstractmethod
    def train(self, training_features: "r,f", training_targets: "r,o") -> None:
        ...

    @abstractmethod
    def predict(self, test_features: "r,f") -> "r,o":
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: "f" = None

    def set_weights(self, weights: "f") -> None:
        self._weights = weights

    def train(self, training_features: "r,f", training_targets: "r,o") -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        def loss(weights: "f") -> "":
            pred: "r,o" = self._predict(weights, training_features)
            err: "r,o" = pred - training_targets
            return torch.mean(err ** 2)

        def step(weights: "f") -> None:
            loss(weights).backward()
            with torch.no_grad():
                weights -= LEARNING_RATE * weights.grad
                weights.grad = None

        n_features = training_features.shape[-1]
        weights = torch.zeros((n_features,), requires_grad=True)

        for _ in range(N_ITERATIONS):
            step(weights)

        self._weights = weights.detach().clone()

    def predict(self, test_features: "r,f") -> "r,o":
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    def _predict(weights: "f", test_features: "r,f") -> "r,o":
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: "a,f"
    training_targets: "a,o"
    test_features: "e,f"
    test_targets: "e,o"


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
    init_analyzer(trace_func_names=["main"])
    main()
