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
# pylint: disable=unused-argument  # Bunch of fake functions below has unused arguments.
# pylint: disable=no-member  # PyLint struggles with TensorFlow.
# pylint: disable=ungrouped-imports

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Optional

import pytest
from packaging.version import Version

from check_shapes import Shape
from check_shapes import check_shape as cs
from check_shapes import check_shapes, disable_check_shapes, inherit_check_shapes
from check_shapes.config import ShapeCheckingState
from check_shapes.exceptions import ShapeMismatchError

from ..utils import assert_has_shape

try:
    import numpy as np
    import tensorflow as tf

    requires_tf = lambda f: f
except ImportError:
    from unittest.mock import MagicMock

    np = MagicMock()
    tf = MagicMock()

    requires_tf = pytest.mark.skip("TensorFlow not installed.")


@requires_tf
@pytest.mark.parametrize(
    "shaped,expected_shape",
    [
        (tf.Variable(np.zeros(())), ()),
        (tf.Variable(np.zeros((2, 4))), (2, 4)),
        # pylint: disable=unexpected-keyword-arg
        (tf.Variable(np.zeros((2, 4)), shape=[2, None]), (2, None)),
        (tf.Variable(np.zeros((2, 4)), shape=tf.TensorShape(None)), None),
    ],
)
def test_get_shape(shaped: Any, expected_shape: Shape) -> None:
    assert_has_shape(shaped, expected_shape)


@requires_tf
@pytest.mark.parametrize(
    "state,eager_expected,compiled_expected",
    [
        (ShapeCheckingState.ENABLED, True, True),
        (ShapeCheckingState.EAGER_MODE_ONLY, True, False),
        (ShapeCheckingState.DISABLED, False, False),
    ],
)
def test_shape_checking_state__bool(
    state: ShapeCheckingState, eager_expected: bool, compiled_expected: bool
) -> None:
    enabled = None

    def run() -> None:
        nonlocal enabled
        enabled = bool(state)

    run()
    assert eager_expected == enabled

    tf.function(run)()  # pylint: disable=no-member
    assert compiled_expected == enabled


@requires_tf
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

    main()


@requires_tf
def test_check_shapes__keras() -> None:
    # pylint: disable=arguments-differ,abstract-method,no-value-for-parameter,unexpected-keyword-arg

    @check_shapes(
        "x: [*]",
        "return: [*]",
    )
    def f(x: tf.Tensor) -> tf.Tensor:
        return x + 3

    class SuperLayer(tf.keras.layers.Layer):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._b = tf.Variable(0.0)

        @check_shapes(
            "x: [batch, input_dim]",
            "y: [batch, 1]",
            "return: [batch, input_dim]",
            tf_decorator=True,
        )
        def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return f(x) + y + self._b

    class SubLayer(SuperLayer):
        @inherit_check_shapes
        def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return x - y + self._b

    class MyModel(tf.keras.Model):  # type: ignore[misc]
        def __init__(self, join: SuperLayer) -> None:
            super().__init__()
            self._join = join

        @check_shapes(
            "xy: [batch, input_dim_plus_one]",
            "return: [batch, input_dim]",
            tf_decorator=True,
        )
        def call(self, xy: tf.Tensor) -> tf.Tensor:
            x = cs(xy[:, :-1], "[batch, input_dim]")
            y = cs(xy[:, -1:], "[batch, 1]")
            return self._join(x, y)

    x = tf.ones((32, 3))
    y = tf.zeros((32, 1))
    xy = tf.concat([x, y], axis=1)
    y_bad = tf.zeros((32, 2))
    targets = tf.ones((32, 3))

    def test_layer(join: SuperLayer) -> None:
        join(x, y)

        with pytest.raises(ShapeMismatchError):
            join(x, y_bad)

        model = MyModel(join)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.25),
            loss="mean_squared_error",
        )
        model.fit(x=xy, y=targets)

    test_layer(SuperLayer())
    test_layer(SubLayer())


_Err = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
_Loss = Callable[[], tf.Tensor]

_ID_WRAPPER = lambda x: x
_TF_FUNCTION = tf.function
_SHAPED_TF_FUNCTION_ERR = tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float64),
        tf.TensorSpec(shape=[], dtype=tf.float64),
    ]
)

_SHAPED_TF_FUNCTION_LOSS = tf.function(input_signature=[])
_UNSHAPED_TF_FUNCTION_ERR = tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ]
)
_UNSHAPED_TF_FUNCTION_LOSS = tf.function(input_signature=[])
_RELAXED_TF_FUNCTION = tf.function(experimental_relax_shapes=True)

_NONE_SHAPE = None
_TARGET_SHAPE = tf.TensorShape([])
_V_SHAPE = tf.TensorShape([50])
_UNKNOWN_SHAPE = tf.TensorShape(None)


@requires_tf
@pytest.mark.parametrize(
    "err_wrapper,loss_wrapper",
    [
        (_ID_WRAPPER, _ID_WRAPPER),
        (_TF_FUNCTION, _TF_FUNCTION),
        (_SHAPED_TF_FUNCTION_ERR, _SHAPED_TF_FUNCTION_LOSS),
        (_UNSHAPED_TF_FUNCTION_ERR, _UNSHAPED_TF_FUNCTION_LOSS),
        (_RELAXED_TF_FUNCTION, _RELAXED_TF_FUNCTION),
    ],
)
@pytest.mark.parametrize("target_shape", [_NONE_SHAPE, _TARGET_SHAPE, _UNKNOWN_SHAPE])
@pytest.mark.parametrize("v_shape", [_NONE_SHAPE, _V_SHAPE, _UNKNOWN_SHAPE])
def test_check_shapes_compilation(
    err_wrapper: Callable[[_Err], _Err],
    loss_wrapper: Callable[[_Loss], _Loss],
    target_shape: Optional[tf.TensorShape],
    v_shape: Optional[tf.TensorShape],
) -> None:
    # Yeah, this test seems to be pushing the limits of TensorFlow compilation (which is probably
    # good), but a bunch of this is fragile.
    tf_version = Version(tf.__version__)

    if (target_shape is _UNKNOWN_SHAPE) or (v_shape is _UNKNOWN_SHAPE):
        if (err_wrapper is _TF_FUNCTION) or (err_wrapper is _RELAXED_TF_FUNCTION):
            if Version("2.7.0") <= tf_version < Version("2.8.0"):
                pytest.skip("TensorFlow 2.7.* segfaults when trying to compile this.")
            if Version("2.8.0") <= tf_version < Version("2.9.0"):
                pytest.skip("TensorFlow 2.8.* is missing a TraceType(?) when trying compile this.")

    # See: https://github.com/tensorflow/tensorflow/issues/56414
    if err_wrapper is _RELAXED_TF_FUNCTION:
        if Version("2.9.0") <= tf_version < Version("2.11.0"):
            err_wrapper = _TF_FUNCTION

    if Version(tf.__version__) < Version("2.5.0"):
        # TensorFlow < 2.5.0 doesn't like the optional `z` argument:

        class SqErr:
            @check_shapes(
                "x: [broadcast n...]",
                "y: [broadcast n...]",
                "return: [n...]",
            )
            def __call__(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
                return (x - y) ** 2

    else:

        class SqErr:  # type: ignore[no-redef]
            @check_shapes(
                "x: [broadcast n...]",
                "y: [broadcast n...]",
                "z: [broadcast n...]",
                "return: [n...]",
            )
            def __call__(
                self, x: tf.Tensor, y: tf.Tensor, z: Optional[tf.Tensor] = None
            ) -> tf.Tensor:
                # z only declared to test the case of `None` arguments.
                return (x - y) ** 2

    sq_err = err_wrapper(SqErr())

    dtype = np.float64
    target = tf.Variable(0.5, dtype=dtype, shape=target_shape)
    v = tf.Variable(np.linspace(0.0, 1.0), dtype=dtype, shape=v_shape)

    @loss_wrapper
    @check_shapes(
        "return: [1]",
    )
    def loss() -> tf.Tensor:
        # keepdims is just to add an extra dimension to make the check more interesting.
        return tf.reduce_sum(sq_err(v, target), keepdims=True)

    optimiser = tf.keras.optimizers.SGD(learning_rate=0.25)
    for _ in range(10):
        optimiser.minimize(loss, var_list=[v])

    np.testing.assert_allclose(target, v.numpy(), atol=0.01)


@requires_tf
@pytest.mark.parametrize("func_wrapper", [lambda x: x, tf.function], ids=["none", "tf.function"])
def test_check_shapes__disable__speed(func_wrapper: Callable[[Any], Any]) -> None:
    if func_wrapper is tf.function:
        pytest.skip(
            "This test is super flaky with tf.function, because the overhead of compiling"
            " seems to dominate any difference caused by check_shapes. However we probably"
            " do want some kind of test of the speed with tf.function, so we keep this"
            " skipped test around to remind us."
        )

    x = tf.zeros((3, 4, 5))
    y = tf.ones((3, 4, 5))

    def time_no_checks() -> float:
        before = perf_counter()

        def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
            return a + b

        f = func_wrapper(f)
        for _ in range(10):
            f(x, y)

        after = perf_counter()
        return after - before

    def time_disabled_checks() -> float:
        with disable_check_shapes():
            before = perf_counter()

            @check_shapes(
                "a: [d...]",
                "b: [d...]",
                "return: [d...]",
            )
            def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
                return a + b

            f = func_wrapper(f)
            for _ in range(10):
                f(x, y)

            after = perf_counter()
            return after - before

    def time_with_checks() -> float:
        before = perf_counter()

        @check_shapes(
            "a: [d...]",
            "b: [d...]",
            "return: [d...]",
        )
        def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
            return a + b

        f = func_wrapper(f)
        for _ in range(10):
            f(x, y)

        after = perf_counter()
        return after - before

    time_no_checks()  # Warm-up.
    t_no_checks = time_no_checks()

    time_disabled_checks()  # Warm-up.
    t_disabled_checks = time_disabled_checks()

    time_with_checks()  # Warm-up.
    t_with_checks = time_with_checks()

    assert t_no_checks < t_with_checks
    assert t_disabled_checks < t_with_checks


@requires_tf
def test_issue_1864() -> None:
    @tf.function
    @check_shapes(
        "x: [*]",
        "return: [*]",
    )
    def f(x: tf.Tensor) -> tf.Tensor:
        for _ in tf.range(3):
            x = x + 1.0
        return x

    x = tf.constant(7.0)
    f(x)


@requires_tf
def test_issue_1936() -> None:
    @tf.function
    @check_shapes(
        "x: [*]",
        "return: [*]",
    )
    def f_if(x: tf.Tensor) -> tf.Tensor:
        if tf.size(x) == 0:
            return x
        else:
            return x + x

    @tf.function
    @check_shapes(
        "x: [*]",
        "return: [*]",
    )
    def f_tf_cond(x: tf.Tensor) -> tf.Tensor:
        return tf.cond(tf.size(x) == 0, lambda: x, lambda: x + x)

    x = tf.constant(7.0)
    f_tf_cond(x)
    f_if(x)


@requires_tf
@pytest.mark.parametrize("model_type", ["SuperModel", "SubModel"])
def test_tf_saved_model(model_type: str, tmp_path: Path) -> None:
    class SuperModel:
        @check_shapes(
            "x: [any...]",
            "return: [any...]",
        )
        def f(self, x: tf.Tensor) -> tf.Tensor:
            return x

    class SubModel(SuperModel):
        @inherit_check_shapes
        def f(self, x: tf.Tensor) -> tf.Tensor:
            return x + 1

    x = np.arange(5)
    model = {
        "SuperModel": SuperModel,
        "SubModel": SubModel,
    }[model_type]()
    out_module = tf.Module()
    out_module.f = tf.function(
        model.f,
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float64)],
    )
    tf.saved_model.save(out_module, str(tmp_path))

    in_module = tf.saved_model.load(str(tmp_path))

    np.testing.assert_allclose(
        model.f(x),
        in_module.f(x),
    )
