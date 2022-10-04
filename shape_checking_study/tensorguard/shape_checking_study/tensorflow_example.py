from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import tensorflow as tf

import tensorguard

N_FEATURES = 100
N_TRAINING_ROWS = 100_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


def mktg() -> tensorguard.TensorGuard:
    return tensorguard.TensorGuard()


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
        tg = mktg()
        self._weights = tg.guard(weights, "n_features")

    def train(self, training_features: tf.Tensor, training_targets: tf.Tensor) -> None:
        tg = mktg()
        tg.guard(training_features, "n_rows, n_features")
        tg.guard(training_targets, "n_rows, 1")
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:

        def loss(weights: tf.Tensor) -> tf.Tensor:
            tg.guard(weights, "n_features")
            pred = tg.guard(self._predict(weights, training_features), "n_rows, 1")
            err = tg.guard(pred - training_targets, "n_rows, 1")
            return tf.reduce_mean(err ** 2)

        @tf.function  # type: ignore
        def step(weights: tf.Tensor) -> tf.Tensor:
            tg.guard(weights, "n_features")
            with tf.GradientTape() as g:
                g.watch(weights)
                l = loss(weights)
            loss_grads = tg.guard(g.gradient(l, weights), "n_features")
            return tg.guard(weights - LEARNING_RATE * loss_grads, "n_features")

        n_features = training_features.shape[-1]
        weights = tf.Variable(tf.zeros((n_features,)))
        for _ in range(N_ITERATIONS):
            weights.assign(step(weights))

        self._weights = tf.constant(weights)

    def predict(self, test_features: tf.Tensor) -> tf.Tensor:
        tg = mktg()
        tg.guard(test_features, "n_rows, n_features")
        assert self._weights is not None
        return tg.guard(self._predict(self._weights, test_features), "n_rows, 1")

    @staticmethod
    def _predict(weights: tf.Tensor, test_features: tf.Tensor) -> tf.Tensor:
        tg = mktg()
        tg.guard(weights, "n_features")
        tg.guard(test_features, "n_rows, n_features")
        return tg.guard(test_features @ weights[:, None], "n_rows, 1")


@dataclass
class TestData:
    training_features: tf.Tensor
    training_targets: tf.Tensor
    test_features: tf.Tensor
    test_targets: tf.Tensor

    def __post_init__(self) -> None:
        tg = mktg()
        tg.guard(self.training_features, "n_training_rows, n_features")
        tg.guard(self.training_targets, "n_training_rows, 1")
        tg.guard(self.test_features, "n_test_rows, n_features")
        tg.guard(self.test_targets, "n_test_rows, 1")


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
