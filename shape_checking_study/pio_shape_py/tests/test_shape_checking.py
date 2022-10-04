# pylint: disable=unnecessary-lambda
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pytest
from pio_core.util.assertions import ensures, requires
from pio_core.util.shape import IsShapeOrShapedCompatible, assert_shape_or_shaped_compatible


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(AssertionError):
            f()

    return g


def test_constant_axis() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([2, 3])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([4, 5])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2, 3)))


@must_fail
def test_constant_axis__bad_arg() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([2, 3])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([4, 5])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((3, 2)))


@must_fail
def test_constant_axis__bad_return() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([2, 3])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([4, 5])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((5, 4))

    f(np.zeros((2, 3)))


def test_variable_axis() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([None, None])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2, 3)))


@must_fail
def test_variable_axis__bad_arg() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([None, None])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2,)))


@must_fail
def test_variable_axis__bad_return() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([None, None])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5, 6))

    f(np.zeros((2, 3)))


def test_correlated_variable_axes() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
    @ensures(lambda result, x: IsShapeOrShapedCompatible([x.shape[0], 1])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)[:, None]

    f(np.zeros((2, 3)))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
    @ensures(lambda result, x: IsShapeOrShapedCompatible([x.shape[0], 1])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)

    f(np.zeros((2, 3)))


def test_variable_rank() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([..., None])(x))
    @ensures(lambda result, x: IsShapeOrShapedCompatible([x.shape[:-1], 1])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)[..., None]

    f(np.zeros((2,)))
    f(np.zeros((2, 3)))
    f(np.zeros((2, 3, 4)))


@must_fail
def test_variable_rank__bad() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([..., None])(x))
    @ensures(lambda result, x: IsShapeOrShapedCompatible([x.shape[:-1], 1])(result))
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((1, 2, 1))

    f(np.zeros((2, 3)))


def test_broadcasting() -> None:
    @requires(lambda a: IsShapeOrShapedCompatible([None, None])(a))
    @requires(lambda a, b: IsShapeOrShapedCompatible(a)(b))
    @ensures(lambda result, a: IsShapeOrShapedCompatible(a)(result))
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    f(np.ones((2, 3)), np.ones((2, 3)))
    f(np.ones((2, 1)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_arg() -> None:
    @requires(lambda a: IsShapeOrShapedCompatible([None, None])(a))
    @requires(lambda a, b: IsShapeOrShapedCompatible(a)(b))
    @ensures(lambda result, a: IsShapeOrShapedCompatible(a)(result))
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    f(np.ones((2, 2)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_return() -> None:
    @requires(lambda a: IsShapeOrShapedCompatible([None, None])(a))
    @requires(lambda a, b: IsShapeOrShapedCompatible(a)(b))
    @ensures(lambda result, a: IsShapeOrShapedCompatible(a)(result))
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.ones((2, 1)), np.ones((1, 3)))


def test_tuples() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x[0]))
    @requires(lambda x: IsShapeOrShapedCompatible([x[0].shape[0], None])(x[1]))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x[1])(result[0]))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x[0])(result[1]))
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[0]

    f((np.zeros((2, 3)), np.zeros((2, 4))))


@must_fail
def test_tuples__bad_arg() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x[0]))
    @requires(lambda x: IsShapeOrShapedCompatible([x[0].shape[0], None])(x[1]))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x[1])(result[0]))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x[0])(result[1]))
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[0]

    f((np.zeros((2, 3)), np.zeros((3, 4))))


@must_fail
def test_tuples__bad_return() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x[0]))
    @requires(lambda x: IsShapeOrShapedCompatible([x[0].shape[0], None])(x[1]))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x[1])(result[0]))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x[0])(result[1]))
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[1]

    f((np.zeros((2, 3)), np.zeros((2, 4))))


def test_dataclass() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x.a))
    @requires(lambda x: IsShapeOrShapedCompatible([x.a.shape[0], None])(x.b))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x.b)(result.a))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x.a)(result.b))
    def f(x: Data) -> Data:
        return Data(a=x.b, b=x.a)

    f(Data(a=np.zeros((2, 3)), b=np.zeros((2, 4))))


@must_fail
def test_dataclass__bad_arg() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x.a))
    @requires(lambda x: IsShapeOrShapedCompatible([x.a.shape[0], None])(x.b))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x.b)(result.a))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x.a)(result.b))
    def f(x: Data) -> Data:
        return Data(a=x.b, b=x.a)

    f(Data(a=np.zeros((2, 3)), b=np.zeros((3, 4))))


@must_fail
def test_dataclass__bad_return() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x.a))
    @requires(lambda x: IsShapeOrShapedCompatible([x.a.shape[0], None])(x.b))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x.b)(result.a))
    @ensures(lambda result, x: IsShapeOrShapedCompatible(x.a)(result.b))
    def f(x: Data) -> Data:
        return Data(a=x.b, b=x.b)

    f(Data(a=np.zeros((2, 3)), b=np.zeros((2, 4))))


def test_lists() -> None:
    @requires(lambda xs: all(IsShapeOrShapedCompatible([None, None])(x) for x in xs))
    @ensures(lambda result: IsShapeOrShapedCompatible([None, None])(result))
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0)

    concat_rows([np.zeros((2, 3))])
    concat_rows([np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))])
    concat_rows([np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((3, 3))])


@must_fail
def test_lists__bad() -> None:
    @requires(lambda xs: all(IsShapeOrShapedCompatible([None, None])(x) for x in xs))
    @ensures(lambda result: IsShapeOrShapedCompatible([None, None])(result))
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0)

    concat_rows([np.zeros((1, 3)), np.zeros((2, 4)), np.zeros((3, 3))])


def test_class_inheritance() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
        @ensures(lambda result, x: IsShapeOrShapedCompatible([x.shape[0], 1])(result))
        def f(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=-1)[:, None]

    b = B()
    b.f(np.zeros((2, 3)))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
        @ensures(lambda result, x: IsShapeOrShapedCompatible([x.shape[0], 1])(result))
        def f(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=-1)

    b = B()
    b.f(np.zeros((2, 3)))


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        @requires(
            lambda train_features: IsShapeOrShapedCompatible([None, None])(train_features)
        )
        @requires(
            lambda train_features, train_labels: IsShapeOrShapedCompatible(
                [train_features.shape[0], None]
            )(train_labels)
        )
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @requires(lambda test_features: IsShapeOrShapedCompatible([None, None])(test_features))
        @ensures(
            lambda result, test_features: IsShapeOrShapedCompatible(
                [test_features.shape[0], None]
            )(result)
        )
        def predict(self, test_features: np.ndarray) -> np.ndarray:
            return np.sum(test_features, axis=-1, keepdims=True) + self._weights

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 3)))


@must_fail
def test_instance_shape__bad_arg() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        @requires(
            lambda train_features: IsShapeOrShapedCompatible([None, None])(train_features)
        )
        @requires(
            lambda train_features, train_labels: IsShapeOrShapedCompatible(
                [train_features.shape[0], None]
            )(train_labels)
        )
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @requires(lambda test_features: IsShapeOrShapedCompatible([None, None])(test_features))
        @ensures(
            lambda result, test_features: IsShapeOrShapedCompatible(
                [test_features.shape[0], None]
            )(result)
        )
        def predict(self, test_features: np.ndarray) -> np.ndarray:
            return np.sum(test_features, axis=-1, keepdims=True) + self._weights

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 4)))


@must_fail
def test_instance_shape__bad_return() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        @requires(
            lambda train_features: IsShapeOrShapedCompatible([None, None])(train_features)
        )
        @requires(
            lambda train_features, train_labels: IsShapeOrShapedCompatible(
                [train_features.shape[0], None]
            )(train_labels)
        )
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @requires(lambda test_features: IsShapeOrShapedCompatible([None, None])(test_features))
        @ensures(
            lambda result, test_features: IsShapeOrShapedCompatible(
                [test_features.shape[0], None]
            )(result)
        )
        def predict(self, test_features: np.ndarray) -> np.ndarray:
            return np.sum(test_features, axis=-1, keepdims=True)

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 3)))


def test_inner_functions() -> None:
    @requires(lambda target: IsShapeOrShapedCompatible([None])(target))
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        @requires(
            lambda prediction: IsShapeOrShapedCompatible([None, target.shape[0]])(prediction)
        )
        @ensures(lambda result: IsShapeOrShapedCompatible([])(result))
        def loss(prediction: np.ndarray) -> np.ndarray:
            return np.sum((target - prediction) ** 2)

        return loss  # type: ignore

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 3)))


@must_fail
def test_inner_functions__bad_arg() -> None:
    @requires(lambda target: IsShapeOrShapedCompatible([None])(target))
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        @requires(
            lambda prediction: IsShapeOrShapedCompatible([None, target.shape[0]])(prediction)
        )
        @ensures(lambda result: IsShapeOrShapedCompatible([])(result))
        def loss(prediction: np.ndarray) -> np.ndarray:
            return np.sum((target - prediction) ** 2)

        return loss  # type: ignore

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 2)))


def test_intermediate_results() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([])(result))
    def f(x: np.ndarray) -> np.ndarray:
        n_rows, n_cols = x.shape
        a = x
        assert_shape_or_shaped_compatible(a, [n_rows, n_cols])
        b = np.sum(a, axis=-1)
        assert_shape_or_shaped_compatible(b, [n_rows])
        c = np.sum(b, axis=-1)
        assert_shape_or_shaped_compatible(c, [])
        return c

    f(np.zeros((2, 3)))


@must_fail
def test_intermediate_results__bad() -> None:
    @requires(lambda x: IsShapeOrShapedCompatible([None, None])(x))
    @ensures(lambda result: IsShapeOrShapedCompatible([])(result))
    def f(x: np.ndarray) -> np.ndarray:
        n_rows, n_cols = x.shape
        a = x
        assert_shape_or_shaped_compatible(a, [n_rows, n_cols])
        b = np.sum(a, axis=-1)
        assert_shape_or_shaped_compatible(b, [n_cols])
        c = np.sum(b, axis=-1)
        assert_shape_or_shaped_compatible(c, [])
        return c

    f(np.zeros((2, 3)))


def test_dtypes() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return x

    f(np.zeros((2, 3), dtype=np.float32))
    f(np.zeros((2, 3), dtype=np.float64))


@must_fail
def test_dtypes__bad_type() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        return x

    f(np.zeros((2, 3), dtype=np.int32))


@must_fail
def test_dtypes__bad_return() -> None:
    # x: in {float32, float64}
    # return: same as x
    def f(x: np.ndarray) -> np.ndarray:
        return x.astype(np.float64)

    f(np.zeros((2, 3), dtype=np.float32))
