# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
An example of using ``check_shapes`` with ``TensorFlow``.

This example fits a linear model to some data, using gradient descent.
"""

# pylint: disable=import-error  # Dependencies might not be installed.
# pylint: disable=no-member  # PyLint struggles with TensorFlow.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

from check_shapes import Shape
from check_shapes import check_shape as cs
from check_shapes import check_shapes, disable_check_shapes, inherit_check_shapes
from check_shapes.config import ShapeCheckingState
from check_shapes.exceptions import ShapeMismatchError

N_FEATURES = 100
N_TRAINING_ROWS = 100_000
N_TEST_ROWS = 10_000

N_ITERATIONS = 1_000
LEARNING_RATE = 1e-2


class Model(ABC):
    @abstractmethod
    @check_shapes(
        "training_features: [n_rows, n_features]",
        "training_targets: [n_rows, 1]",
    )
    def train(self, training_features: tf.Tensor, training_targets: tf.Tensor) -> None:
        ...

    @abstractmethod
    @check_shapes(
        "test_features: [n_rows, n_features]",
        "return: [n_rows, 1]",
    )
    def predict(self, test_features: tf.Tensor) -> tf.Tensor:
        ...


class LinearModel(Model):
    def __init__(self) -> None:
        self._weights: Optional[tf.Tensor] = None  # [n_features]

    @check_shapes(
        "weights: [n_features]",
    )
    def set_weights(self, weights: tf.Tensor) -> None:
        self._weights = weights

    @inherit_check_shapes
    def train(self, training_features: tf.Tensor, training_targets: tf.Tensor) -> None:
        # We intentionally split this into a few more functions than might technically be
        # needed, so that we have something to annotate with type checks:

        @check_shapes(
            "weights: [n_features]",
            "return: []",
        )
        def loss(weights: tf.Tensor) -> tf.Tensor:
            pred = cs(self._predict(weights, training_features), "[n_rows, 1]")
            err = cs(pred - training_targets, "[n_rows, 1]")
            return tf.reduce_mean(err ** 2)

        @tf.function  # type: ignore
        @check_shapes(
            "weights: [n_features]",
            "return: [n_features]",
        )
        def step(weights: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as g:
                g.watch(weights)
                l = cs(loss(weights), "[]")
            loss_grads = cs(g.gradient(l, weights), "[n_features]")
            return weights - LEARNING_RATE * loss_grads

        n_features = training_features.shape[-1]
        weights = tf.Variable(tf.zeros((n_features,)))
        for _ in range(N_ITERATIONS):
            weights.assign(step(weights))

        self._weights = tf.constant(weights)

    @inherit_check_shapes
    def predict(self, test_features: tf.Tensor) -> tf.Tensor:
        assert self._weights is not None
        return self._predict(self._weights, test_features)

    @staticmethod
    @check_shapes(
        "weights: [n_features]",
        "test_features: [n_rows, n_features]",
        "return: [n_rows, 1]",
    )
    def _predict(weights: tf.Tensor, test_features: tf.Tensor) -> tf.Tensor:
        return test_features @ weights[:, None]


@dataclass
class TestData:
    training_features: tf.Tensor
    training_targets: tf.Tensor
    test_features: tf.Tensor
    test_targets: tf.Tensor

    @check_shapes(
        "self.training_features: [n_training_rows, n_features]",
        "self.training_targets: [n_training_rows, 1]",
        "self.test_features: [n_test_rows, n_features]",
        "self.test_targets: [n_test_rows, 1]",
    )
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
