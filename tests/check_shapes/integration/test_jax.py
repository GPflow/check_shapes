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

# pylint: disable=import-error  # Dependencies might not be installed.
# pylint: disable=ungrouped-imports

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Optional

import pytest

from check_shapes import Shape
from check_shapes import check_shape as cs
from check_shapes import check_shapes, inherit_check_shapes

from ..utils import assert_has_shape

try:
    import jax.numpy as jnp
    from jax import grad, jit, random

    requires_jax = lambda f: f
except ImportError:
    from unittest.mock import MagicMock

    jnp = MagicMock()
    np = MagicMock()
    grad = MagicMock()
    jit = MagicMock()
    random = MagicMock()

    requires_jax = pytest.mark.skip("Jax not installed.")


# Some versions of NumPy has generic ndarrays while other don't:
ndarray = Any


@requires_jax
@pytest.mark.parametrize(
    "shaped,expected_shape",
    [
        (jnp.zeros(()), ()),
        (jnp.zeros((3, 4)), (3, 4)),
    ],
)
def test_get_shape(shaped: Any, expected_shape: Shape) -> None:
    assert_has_shape(shaped, expected_shape)


@requires_jax
def test_check_shapes() -> None:
    N_FEATURES = 10
    N_TRAINING_ROWS = 1_000
    N_TEST_ROWS = 100

    N_ITERATIONS = 1_000
    LEARNING_RATE = 1e-2

    class Model(ABC):
        @abstractmethod
        @check_shapes(
            "training_features: [n_rows, n_features]",
            "training_targets: [n_rows, 1]",
        )
        def train(self, training_features: ndarray, training_targets: ndarray) -> None:
            ...

        @abstractmethod
        @check_shapes(
            "test_features: [n_rows, n_features]",
            "return: [n_rows, 1]",
        )
        def predict(self, test_features: ndarray) -> ndarray:
            ...

    class LinearModel(Model):
        def __init__(self) -> None:
            self._weights: Optional[ndarray] = None  # [n_features]

        @check_shapes(
            "weights: [n_features]",
        )
        def set_weights(self, weights: ndarray) -> None:
            self._weights = weights

        @inherit_check_shapes
        def train(self, training_features: ndarray, training_targets: ndarray) -> None:
            # We intentionally split this into a few more functions than might technically be
            # needed, so that we have something to annotate with type checks:

            @check_shapes(
                "weights: [n_features]",
                "return: []",
            )
            def loss(weights: ndarray) -> ndarray:
                pred = cs(self._predict(weights, training_features), "[n_rows, 1]")
                err = cs(pred - training_targets, "[n_rows, 1]")
                return jnp.mean(err ** 2)

            loss_grads = grad(loss)

            @jit
            @check_shapes(
                "weights: [n_features]",
                "return: [n_features]",
            )
            def step(weights: ndarray) -> ndarray:
                return weights - LEARNING_RATE * loss_grads(weights)

            n_features = training_features.shape[-1]
            weights = jnp.zeros((n_features,))

            for _ in range(N_ITERATIONS):
                weights = step(weights)

            self._weights = weights

        @inherit_check_shapes
        def predict(self, test_features: ndarray) -> ndarray:
            assert self._weights is not None
            return self._predict(self._weights, test_features)

        @staticmethod
        @check_shapes(
            "weights: [n_features]",
            "test_features: [n_rows, n_features]",
            "return: [n_rows, 1]",
        )
        def _predict(weights: ndarray, test_features: ndarray) -> ndarray:
            return test_features @ weights[:, None]

    @dataclass
    class TestData:
        training_features: ndarray
        training_targets: ndarray
        test_features: ndarray
        test_targets: ndarray

        @check_shapes(
            "self.training_features: [n_training_rows, n_features]",
            "self.training_targets: [n_training_rows, 1]",
            "self.test_features: [n_test_rows, n_features]",
            "self.test_targets: [n_test_rows, 1]",
        )
        def __post_init__(self) -> None:
            pass

    def create_data() -> TestData:
        key = random.PRNGKey(42)
        key1, key2, key3 = random.split(key, 3)
        model = LinearModel()
        model.set_weights(jnp.arange(N_FEATURES))
        training_features = random.uniform(key1, [N_TRAINING_ROWS, N_FEATURES])
        training_targets = model.predict(training_features) + random.normal(
            key2, [N_TRAINING_ROWS, 1]
        )
        test_features = random.uniform(key3, [N_TEST_ROWS, N_FEATURES])
        test_targets = model.predict(test_features)
        return TestData(training_features, training_targets, test_features, test_targets)

    def evaluate(model: Model, data: TestData) -> Mapping[str, float]:
        test_predictions = model.predict(data.test_features)
        err = data.test_targets - test_predictions
        result = {}
        result["rmse"] = float(jnp.sqrt(jnp.mean(err ** 2)))
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

    main()
