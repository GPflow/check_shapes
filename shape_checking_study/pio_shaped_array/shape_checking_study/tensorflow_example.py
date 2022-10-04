from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional

import tensorflow as tf
from pio_learning_utilities.arrays.shaped_array import fixed_dimension, variable_dimension
from pio_learning_utilities.arrays.tensorflow import ShapedTfTensor

N_FEATURES = 100
N_TRAINING_ROWS = 100_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 10_000
LEARNING_RATE = 1e-2


class Features(ShapedTfTensor):

    n_rows = variable_dimension()
    n_features = variable_dimension()


class Targets(ShapedTfTensor):

    n_rows = variable_dimension()
    n_targets = fixed_dimension(1)


class Weights(ShapedTfTensor):

    n_features = variable_dimension()


class Loss(ShapedTfTensor):

    pass


class Model(ABC):
    @abstractmethod
    def train(self, training_features: Features, training_targets: Targets) -> None:
        ...

    @abstractmethod
    def predict(self, test_features: Features) -> Targets:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[Weights] = None

    def set_weights(self, weights: Weights) -> None:
        self._weights = weights

    def train(self, training_features: Features, training_targets: Targets) -> None:
        # We intentionally split this into a few more functions than might technically be needed, so
        # that we have something to annotate with type checks:
        n_features = training_features.n_features

        def loss(weights: Weights) -> Loss:
            pred = self._predict(weights, training_features)
            err = pred.array - training_targets.array
            return Loss(tf.reduce_mean(err ** 2))

        @tf.function  # type: ignore
        # ShapedTfTensor doesn't seem to work with tf.function.
        def step(weights: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as g:
                g.watch(weights)
                l = loss(Weights(weights, n_features=n_features))
            loss_grads = g.gradient(l.array, weights)
            return weights - LEARNING_RATE * loss_grads

        weights = Weights(tf.Variable(tf.zeros((n_features,))), n_features=n_features)
        for _ in range(N_ITERATIONS):
            weights.array.assign(step(weights.array))

        self._weights = Weights(tf.constant(weights.array), n_features=n_features)

    def predict(self, test_features: Features) -> Targets:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    def _predict(weights: Weights, test_features: Features) -> Targets:
        return Targets(
            test_features.array @ weights.array[:, None], n_rows=test_features.n_rows
        )


@dataclass
class TestData:
    training_features: Features
    training_targets: Targets
    test_features: Features
    test_targets: Targets


def create_data() -> TestData:
    tf.random.set_seed(42)
    model = LinearModel()
    model.set_weights(Weights(tf.range(N_FEATURES, dtype=tf.float32), n_features=N_FEATURES))
    training_features = Features(
        tf.random.uniform([N_TRAINING_ROWS, N_FEATURES], 0.0, 1.0),
        n_rows=N_TRAINING_ROWS,
        n_features=N_FEATURES,
    )
    training_targets = Targets(
        model.predict(training_features).array
        + tf.random.normal([N_TRAINING_ROWS, 1], 0.0, 1.0),
        n_rows=N_TRAINING_ROWS,
    )
    test_features = Features(
        tf.random.uniform([N_TEST_ROWS, N_FEATURES], 0.0, 1.0),
        n_rows=N_TEST_ROWS,
        n_features=N_FEATURES,
    )
    test_targets = model.predict(test_features)
    return TestData(training_features, training_targets, test_features, test_targets)


def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
    test_predictions = model.predict(data.test_features)
    err = data.test_targets.array - test_predictions.array
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
