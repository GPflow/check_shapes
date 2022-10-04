from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import tensorflow as tf

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

    def set_weights(self, weights: tf.Tensor) -> None:
        tf.debugging.assert_shapes(
            [
                (weights, ["n_features"]),
            ]
        )
        self._weights = weights

    def train(self, training_features: tf.Tensor, training_targets: tf.Tensor) -> None:
        n_rows, n_features = training_features.shape
        tf.debugging.assert_shapes(
            [
                (training_features, [n_rows, n_features]),
                (training_targets, [n_rows, 1]),
            ]
        )
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        def loss(weights: tf.Tensor) -> tf.Tensor:
            tf.debugging.assert_shapes(
                [
                    (weights, [n_features]),
                ]
            )
            pred = self._predict(weights, training_features)
            tf.debugging.assert_shapes(
                [
                    (pred, [n_rows, 1]),
                ]
            )
            err = pred - training_targets
            tf.debugging.assert_shapes(
                [
                    (err, [n_rows, 1]),
                ]
            )
            result = tf.reduce_mean(err ** 2)
            tf.debugging.assert_shapes(
                [
                    (result, []),
                ]
            )
            return result

        @tf.function  # type: ignore
        def step(weights: tf.Tensor) -> tf.Tensor:
            tf.debugging.assert_shapes(
                [
                    (weights, [n_features]),
                ]
            )
            with tf.GradientTape() as g:
                g.watch(weights)
                l = loss(weights)
                tf.debugging.assert_shapes(
                    [
                        (l, []),
                    ]
                )
            loss_grads = g.gradient(l, weights)
            tf.debugging.assert_shapes(
                [
                    (loss_grads, [n_features]),
                ]
            )
            result = weights - LEARNING_RATE * loss_grads
            tf.debugging.assert_shapes(
                [
                    (result, [n_features]),
                ]
            )
            return result

        n_features = training_features.shape[-1]
        weights = tf.Variable(tf.zeros((n_features,)))
        for _ in range(N_ITERATIONS):
            weights.assign(step(weights))

        self._weights = tf.constant(weights)

    def predict(self, test_features: tf.Tensor) -> tf.Tensor:
        n_rows, n_features = test_features.shape
        assert self._weights is not None
        tf.debugging.assert_shapes(
            [
                (self._weights, [n_features]),
            ]
        )
        result = self._predict(self._weights, test_features)
        tf.debugging.assert_shapes(
            [
                (result, [n_rows, 1]),
            ]
        )
        return result

    @staticmethod
    def _predict(weights: tf.Tensor, test_features: tf.Tensor) -> tf.Tensor:
        n_rows, n_features = test_features.shape
        tf.debugging.assert_shapes(
            [
                (weights, [n_features]),
                (test_features, [n_rows, n_features]),
            ]
        )
        result = test_features @ weights[:, None]
        tf.debugging.assert_shapes(
            [
                (result, [n_rows, 1]),
            ]
        )
        return result


@dataclass
class TestData:
    training_features: tf.Tensor
    training_targets: tf.Tensor
    test_features: tf.Tensor
    test_targets: tf.Tensor

    def __post_init__(self) -> None:
        tf.debugging.assert_shapes(
            [
                (self.training_features, ["n_training_rows", "n_features"]),
                (self.training_targets, ["n_training_rows", 1]),
                (self.test_features, ["n_test_rows", "n_features"]),
                (self.test_targets, ["n_test_rows", 1]),
            ]
        )


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
