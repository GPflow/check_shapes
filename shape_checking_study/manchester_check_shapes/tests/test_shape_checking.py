from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pytest

from shape_checking_study.check_shapes import (
    ShapeMismatchError,
    check_shapes,
    inherit_check_shapes,
)


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(ShapeMismatchError):
            f()

    return g


def test_constant_axis() -> None:
    @check_shapes(
        ("x", [2, 3]),
        ("return", [4, 5]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2, 3)))


@must_fail
def test_constant_axis__bad_arg() -> None:
    @check_shapes(
        ("x", [2, 3]),
        ("return", [4, 5]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((3, 2)))


@must_fail
def test_constant_axis__bad_return() -> None:
    @check_shapes(
        ("x", [2, 3]),
        ("return", [4, 5]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((5, 4))

    f(np.zeros((2, 3)))


def test_variable_axis() -> None:
    @check_shapes(
        ("x", ["n_rows_in", "n_cols_in"]),
        ("return", ["n_rows_out", "n_cols_out"]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2, 3)))


@must_fail
def test_variable_axis__bad_arg() -> None:
    @check_shapes(
        ("x", ["n_rows_in", "n_cols_in"]),
        ("return", ["n_rows_out", "n_cols_out"]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2,)))


@must_fail
def test_variable_axis__bad_return() -> None:
    @check_shapes(
        ("x", ["n_rows_in", "n_cols_in"]),
        ("return", ["n_rows_out", "n_cols_out"]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5, 6))

    f(np.zeros((2, 3)))


def test_correlated_variable_axes() -> None:
    @check_shapes(
        ("x", ["n_rows", "n_cols_in"]),
        ("return", ["n_rows", 1]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)[:, None]

    f(np.zeros((2, 3)))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    @check_shapes(
        ("x", ["n_rows", "n_cols_in"]),
        ("return", ["n_rows", 1]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)

    f(np.zeros((2, 3)))


def test_variable_rank() -> None:
    @check_shapes(
        ("x", ["batch...", "n_cols_in"]),
        ("return", ["batch...", 1]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)[..., None]

    f(np.zeros((2,)))
    f(np.zeros((2, 3)))
    f(np.zeros((2, 3, 4)))


@must_fail
def test_variable_rank__bad() -> None:
    @check_shapes(
        ("x", ["batch...", "n_cols_in"]),
        ("return", ["batch...", 1]),
    )
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((1, 2, 1))

    f(np.zeros((2, 3)))


def test_broadcasting() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d2"]),
        ("return", ["d1", "d2"]),
    )
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    f(np.ones((2, 3)), np.ones((2, 3)))
    f(np.ones((2, 1)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_arg() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d2"]),
        ("return", ["d1", "d2"]),
    )
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    f(np.ones((2, 2)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_return() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d2"]),
        ("return", ["d1", "d2"]),
    )
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.ones((2, 1)), np.ones((1, 3)))


def test_tuples() -> None:
    @check_shapes(
        ("x[0]", ["n_rows", "n_cols0"]),
        ("x[1]", ["n_rows", "n_cols1"]),
        ("return[0]", ["n_rows", "n_cols1"]),
        ("return[1]", ["n_rows", "n_cols0"]),
    )
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[0]

    f((np.zeros((2, 3)), np.zeros((2, 4))))


@must_fail
def test_tuples__bad_arg() -> None:
    @check_shapes(
        ("x[0]", ["n_rows", "n_cols0"]),
        ("x[1]", ["n_rows", "n_cols1"]),
        ("return[0]", ["n_rows", "n_cols1"]),
        ("return[1]", ["n_rows", "n_cols0"]),
    )
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[0]

    f((np.zeros((2, 3)), np.zeros((3, 4))))


@must_fail
def test_tuples__bad_return() -> None:
    @check_shapes(
        ("x[0]", ["n_rows", "n_cols0"]),
        ("x[1]", ["n_rows", "n_cols1"]),
        ("return[0]", ["n_rows", "n_cols1"]),
        ("return[1]", ["n_rows", "n_cols0"]),
    )
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[1]

    f((np.zeros((2, 3)), np.zeros((2, 4))))


def test_dataclass() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    @check_shapes(
        ("x.a", ["n_rows", "n_cols0"]),
        ("x.b", ["n_rows", "n_cols1"]),
        ("return.a", ["n_rows", "n_cols1"]),
        ("return.b", ["n_rows", "n_cols0"]),
    )
    def f(x: Data) -> Data:
        return Data(a=x.b, b=x.a)

    f(Data(a=np.zeros((2, 3)), b=np.zeros((2, 4))))


@must_fail
def test_dataclass__bad_arg() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    @check_shapes(
        ("x.a", ["n_rows", "n_cols0"]),
        ("x.b", ["n_rows", "n_cols1"]),
        ("return.a", ["n_rows", "n_cols1"]),
        ("return.b", ["n_rows", "n_cols0"]),
    )
    def f(x: Data) -> Data:
        return Data(a=x.b, b=x.a)

    f(Data(a=np.zeros((2, 3)), b=np.zeros((3, 4))))


@must_fail
def test_dataclass__bad_return() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    @check_shapes(
        ("x.a", ["n_rows", "n_cols0"]),
        ("x.b", ["n_rows", "n_cols1"]),
        ("return.a", ["n_rows", "n_cols1"]),
        ("return.b", ["n_rows", "n_cols0"]),
    )
    def f(x: Data) -> Data:
        return Data(a=x.b, b=x.b)

    f(Data(a=np.zeros((2, 3)), b=np.zeros((2, 4))))


def test_lists() -> None:
    @check_shapes(
        # ("xs[*]", [?, n_cols]),
        ("return", ["n_total_rows", "n_cols"]),
    )
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0)

    concat_rows([np.zeros((2, 3))])
    concat_rows([np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))])
    concat_rows([np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((3, 3))])


@must_fail
def test_lists__bad() -> None:
    @check_shapes(
        # ("xs[*]", [?, n_cols]),
        ("return", ["n_total_rows", "n_cols"]),
    )
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0)

    concat_rows([np.zeros((1, 3)), np.zeros((2, 4)), np.zeros((3, 3))])


def test_class_inheritance() -> None:
    class A(ABC):
        @abstractmethod
        @check_shapes(
            ("x", ["n_rows", "n_cols_in"]),
            ("return", ["n_rows", 1]),
        )
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        @inherit_check_shapes
        def f(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=-1)[:, None]

    b = B()
    b.f(np.zeros((2, 3)))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class A(ABC):
        @abstractmethod
        @check_shapes(
            ("x", ["n_rows", "n_cols_in"]),
            ("return", ["n_rows", 1]),
        )
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        @inherit_check_shapes
        def f(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=-1)

    b = B()
    b.f(np.zeros((2, 3)))


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        @check_shapes(
            ("train_features", ["n_train_rows", "n_features"]),
            ("train_labels", ["n_train_rows", "n_labels"]),
        )
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @check_shapes(
            ("test_features", ["n_test_rows", "n_features"]),
            ("return", ["n_test_rows", "n_labels"]),
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

        @check_shapes(
            ("train_features", ["n_train_rows", "n_features"]),
            ("train_labels", ["n_train_rows", "n_labels"]),
        )
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @check_shapes(
            ("test_features", ["n_test_rows", "n_features"]),
            ("return", ["n_test_rows", "n_labels"]),
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

        @check_shapes(
            ("train_features", ["n_train_rows", "n_features"]),
            ("train_labels", ["n_train_rows", "n_labels"]),
        )
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @check_shapes(
            ("test_features", ["n_test_rows", "n_features"]),
            ("return", ["n_test_rows", "n_labels"]),
        )
        def predict(self, test_features: np.ndarray) -> np.ndarray:
            return np.sum(test_features, axis=-1, keepdims=True)

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 3)))


def test_inner_functions() -> None:
    @check_shapes(
        ("target", ["n_outputs"]),
    )
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        @check_shapes(
            ("prediction", ["n_rows", "n_outputs"]),
            ("return", []),
        )
        def loss(prediction: np.ndarray) -> np.ndarray:
            return np.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 3)))


@must_fail
def test_inner_functions__bad_arg() -> None:
    @check_shapes(
        ("target", ["n_outputs"]),
    )
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        @check_shapes(
            ("prediction", ["n_rows", "n_outputs"]),
            ("return", []),
        )
        def loss(prediction: np.ndarray) -> np.ndarray:
            return np.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 2)))


def test_intermediate_results() -> None:
    @check_shapes(
        ("x", ["n_rows", "n_cols"]),
        ("return", []),
    )
    def f(x: np.ndarray) -> np.ndarray:
        a = x
        b = np.sum(a, axis=-1)
        c = np.sum(b, axis=-1)
        return c

    f(np.zeros((2, 3)))


@must_fail
def test_intermediate_results__bad() -> None:
    @check_shapes(
        ("x", ["n_rows", "n_cols"]),
        ("return", []),
    )
    def f(x: np.ndarray) -> np.ndarray:
        a = x
        b = np.sum(a, axis=-1)
        c = np.sum(b, axis=-1)
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
    def f(x: np.ndarray) -> np.ndarray:
        return x.astype(np.float64)

    f(np.zeros((2, 3), dtype=np.float32))
