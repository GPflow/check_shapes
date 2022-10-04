from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, NewType, Optional, cast

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from tensor_annotations import axes

N_FEATURES = 100
N_TRAINING_ROWS = 100_000
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
        training_features: ttf.Tensor2[NRows, NFeatures],
        training_targets: ttf.Tensor2[NRows, N1],
    ) -> None:
        ...

    @abstractmethod
    def predict(self, test_features: ttf.Tensor2[NRows, NFeatures]) -> ttf.Tensor2[NRows, N1]:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[ttf.Tensor1[NFeatures]] = None

    def set_weights(self, weights: ttf.Tensor1[NFeatures]) -> None:
        self._weights = weights

    def train(
        self,
        training_features: ttf.Tensor2[NRows, NFeatures],
        training_targets: ttf.Tensor2[NRows, N1],
    ) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        def loss(weights: ttf.Tensor1[NFeatures]) -> ttf.Tensor0:
            pred: ttf.Tensor2[NRows, N1] = self._predict(weights, training_features)
            err: ttf.Tensor2[NRows, N1] = pred - training_targets
            return cast(ttf.Tensor0, tf.reduce_mean(err ** 2))

        @tf.function
        def step(weights: ttf.Tensor1[NFeatures]) -> ttf.Tensor1[NFeatures]:
            with tf.GradientTape() as g:  # type: ignore
                g.watch(weights)
                l: ttf.Tensor0 = loss(weights)
            loss_grads: ttf.Tensor1[NFeatures] = g.gradient(l, weights)
            return weights - LEARNING_RATE * loss_grads

        n_features = training_features.shape[-1]  # type: ignore
        weights = tf.Variable(tf.zeros((n_features,)))
        for _ in range(N_ITERATIONS):
            weights.assign(step(weights))

        self._weights = tf.constant(weights)

    def predict(self, test_features: ttf.Tensor2[NRows, NFeatures]) -> ttf.Tensor2[NRows, N1]:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    def _predict(
        weights: ttf.Tensor1[NFeatures], test_features: ttf.Tensor2[NRows, NFeatures]
    ) -> ttf.Tensor2[NRows, N1]:
        return cast(ttf.Tensor2[NRows, N1], test_features @ weights[:, None])


@dataclass
class TestData:
    training_features: ttf.Tensor2[NRows, NFeatures]
    training_targets: ttf.Tensor2[NRows, N1]
    test_features: ttf.Tensor2[NRows, NFeatures]
    test_targets: ttf.Tensor2[NRows, N1]


def create_data() -> TestData:
    tf.random.set_seed(42)
    model = LinearModel()
    model.set_weights(tf.range(N_FEATURES, dtype=tf.float32))
    training_features = tf.random.uniform([N_TRAINING_ROWS, N_FEATURES], 0.0, 1.0)
    training_targets = model.predict(training_features) + tf.random.normal(
        [N_TRAINING_ROWS, 1], 0.0, 1.0
    )
    test_features = tf.random.uniform([N_TEST_ROWS, N_FEATURES], 0.0, 1.0)
    test_targets = model.predict(test_features)
    return TestData(training_features, training_targets, test_features, test_targets)


def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets - test_predictions
    result = {}
    result["rmse"] = tf.sqrt(tf.reduce_mean(err ** 2)).numpy()  # type: ignore
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
