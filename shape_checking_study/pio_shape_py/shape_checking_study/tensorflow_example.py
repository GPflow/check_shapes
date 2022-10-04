# pylint: disable=unnecessary-lambda
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import tensorflow as tf
from pio_core.util.assertions import ensures, requires
from pio_core.util.shape import IsShapeOrShapedCompatible, assert_shape_or_shaped_compatible

N_FEATURES = 100
N_TRAINING_ROWS = 100_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


class Model(ABC):
    @abstractmethod
    def train(self, training_features: tf.Tensor, training_targets: tf.Tensor) -> None:
        ...

    @abstractmethod
    def predict(self, test_features: tf.Tensor) -> tf.Tensor:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[tf.Tensor] = None

    @requires(lambda weights: IsShapeOrShapedCompatible([None])(weights))
    def set_weights(self, weights: tf.Tensor) -> None:
        self._weights = weights

    @requires(
        lambda training_features: IsShapeOrShapedCompatible([None, None])(training_features)
    )
    @requires(lambda training_targets: IsShapeOrShapedCompatible([None, 1])(training_targets))
    def train(self, training_features: tf.Tensor, training_targets: tf.Tensor) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        @requires(lambda weights: IsShapeOrShapedCompatible([None])(weights))
        @ensures(lambda result: IsShapeOrShapedCompatible([])(result))
        def loss(weights: tf.Tensor) -> tf.Tensor:
            pred = self._predict(weights, training_features)
            assert_shape_or_shaped_compatible(pred, [None, 1])
            err = pred - training_targets
            assert_shape_or_shaped_compatible(err, pred)
            return tf.reduce_mean(err ** 2)

        @tf.function
        @requires(lambda weights: IsShapeOrShapedCompatible([None])(weights))
        @ensures(lambda result, weights: IsShapeOrShapedCompatible(weights)(result))
        def step(weights: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as g:
                g.watch(weights)
                l = loss(weights)
                assert_shape_or_shaped_compatible(l, [])
            loss_grads = g.gradient(l, weights)
            assert_shape_or_shaped_compatible(loss_grads, weights)
            return weights - LEARNING_RATE * loss_grads

        n_features = training_features.shape[-1]
        weights = tf.Variable(tf.zeros((n_features,)))
        for _ in range(N_ITERATIONS):
            weights.assign(step(weights))

        self._weights = tf.constant(weights)

    @requires(lambda test_features: IsShapeOrShapedCompatible([None, None])(test_features))
    @ensures(
        lambda result, test_features: IsShapeOrShapedCompatible([test_features.shape[0], 1])(
            result
        )
    )
    def predict(self, test_features: tf.Tensor) -> tf.Tensor:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    @requires(lambda weights: IsShapeOrShapedCompatible([None])(weights))
    @requires(
        lambda test_features, weights: IsShapeOrShapedCompatible([None, weights.shape[0]])(
            test_features
        )
    )
    @ensures(
        lambda result, test_features: IsShapeOrShapedCompatible([test_features.shape[0], 1])(
            result
        )
    )
    def _predict(weights: tf.Tensor, test_features: tf.Tensor) -> tf.Tensor:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: tf.Tensor
    training_targets: tf.Tensor
    test_features: tf.Tensor
    test_targets: tf.Tensor

    @requires(lambda self: self.training_features.shape[0] == self.training_targets.shape[0])
    @requires(lambda self: self.test_features.shape[0] == self.test_targets.shape[0])
    @requires(lambda self: self.training_features.shape[1] == self.test_features.shape[1])
    @requires(lambda self: self.training_targets.shape[1] == self.test_targets.shape[1])
    def __post_init__(self) -> None:
        pass


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
    result["rmse"] = tf.sqrt(tf.reduce_mean(err ** 2)).numpy()
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
