from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pytest

import tensorguard


def mktg() -> tensorguard.TensorGuard:
    return tensorguard.TensorGuard()


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(tensorguard.exception.ShapeError):
            f()

    return g


def test_constant_axis() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "2, 3")
        return tg.guard(np.zeros((4, 5)), "4, 5")

    f(np.zeros((2, 3)))


@must_fail
def test_constant_axis__bad_arg() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "2, 3")
        return tg.guard(np.zeros((4, 5)), "4, 5")

    f(np.zeros((3, 2)))


@must_fail
def test_constant_axis__bad_return() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "2, 3")
        return tg.guard(np.zeros((5, 4)), "4, 5")

    f(np.zeros((2, 3)))


def test_variable_axis() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "n_rows_in, n_cols_in")
        return tg.guard(np.zeros((4, 5)), "n_rows_out, n_cols_out")

    f(np.zeros((2, 3)))


@must_fail
def test_variable_axis__bad_arg() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "n_rows_in, n_cols_in")
        return tg.guard(np.zeros((4, 5)), "n_rows_out, n_cols_out")

    f(np.zeros((2,)))


@must_fail
def test_variable_axis__bad_return() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "n_rows_in, n_cols_in")
        return tg.guard(np.zeros((4, 5, 6)), "n_rows_ou, n_cols_out")

    f(np.zeros((2, 3)))


def test_correlated_variable_axes() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "n_rows, n_cols_in")
        return tg.guard(np.sum(x, axis=-1)[:, None], "n_rows, 1")

    f(np.zeros((2, 3)))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "n_rows, n_cols_in")
        return tg.guard(np.sum(x, axis=-1), "n_rows, 1")

    f(np.zeros((2, 3)))


def test_variable_rank() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "..., n_cols_in")
        return tg.guard(np.sum(x, axis=-1)[..., None], "..., 1")

    f(np.zeros((2,)))
    f(np.zeros((2, 3)))
    f(np.zeros((2, 3, 4)))


@must_fail
def test_variable_rank__bad() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "..., n_cols_in")
        return tg.guard(np.zeros((1, 2, 1)), "..., 1")

    f(np.zeros((2, 3)))


def test_broadcasting() -> None:
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(a, "d1, d2")
        tg.guard(b, "d1, d2")
        return tg.guard(a + b, "d1, d2")

    f(np.ones((2, 3)), np.ones((2, 3)))
    f(np.ones((2, 1)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_arg() -> None:
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(a, "d1, d2")
        tg.guard(b, "d1, d2")
        return tg.guard(a + b, "d1, d2")

    f(np.ones((2, 2)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_return() -> None:
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(a, "d1, d2")
        tg.guard(b, "d1, d2")
        return tg.guard(np.zeros((4, 5)), "d1, d2")

    f(np.ones((2, 1)), np.ones((1, 3)))


def test_tuples() -> None:
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        tg = mktg()
        tg.guard(x[0], "n_rows, n_cols0")
        tg.guard(x[1], "n_rows, n_cols1")
        result = x[1], x[0]
        tg.guard(result[0], "n_rows, n_cols1")
        tg.guard(result[1], "n_rows, n_cols0")
        return result

    f((np.zeros((2, 3)), np.zeros((2, 4))))


@must_fail
def test_tuples__bad_arg() -> None:
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        tg = mktg()
        tg.guard(x[0], "n_rows, n_cols0")
        tg.guard(x[1], "n_rows, n_cols1")
        result = x[1], x[0]
        tg.guard(result[0], "n_rows, n_cols1")
        tg.guard(result[1], "n_rows, n_cols0")
        return result

    f((np.zeros((2, 3)), np.zeros((3, 4))))


@must_fail
def test_tuples__bad_return() -> None:
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        tg = mktg()
        tg.guard(x[0], "n_rows, n_cols0")
        tg.guard(x[1], "n_rows, n_cols1")
        result = x[1], x[1]
        tg.guard(result[0], "n_rows, n_cols1")
        tg.guard(result[1], "n_rows, n_cols0")
        return result

    f((np.zeros((2, 3)), np.zeros((2, 4))))


def test_dataclass() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    def f(x: Data) -> Data:
        tg = mktg()
        tg.guard(x.a, "n_rows, n_cols0")
        tg.guard(x.b, "n_rows, n_cols1")
        result = Data(a=x.b, b=x.a)
        tg.guard(result.a, "n_rows, n_cols1")
        tg.guard(result.b, "n_rows, n_cols0")
        return result

    f(Data(a=np.zeros((2, 3)), b=np.zeros((2, 4))))


@must_fail
def test_dataclass__bad_arg() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    def f(x: Data) -> Data:
        tg = mktg()
        tg.guard(x.a, "n_rows, n_cols0")
        tg.guard(x.b, "n_rows, n_cols1")
        result = Data(a=x.b, b=x.a)
        tg.guard(result.a, "n_rows, n_cols1")
        tg.guard(result.b, "n_rows, n_cols0")
        return result

    f(Data(a=np.zeros((2, 3)), b=np.zeros((3, 4))))


@must_fail
def test_dataclass__bad_return() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    def f(x: Data) -> Data:
        tg = mktg()
        tg.guard(x.a, "n_rows, n_cols0")
        tg.guard(x.b, "n_rows, n_cols1")
        result = Data(a=x.b, b=x.b)
        tg.guard(result.a, "n_rows, n_cols1")
        tg.guard(result.b, "n_rows, n_cols0")
        return result

    f(Data(a=np.zeros((2, 3)), b=np.zeros((2, 4))))


def test_lists() -> None:
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        tg = mktg()
        for x in xs:
            tg.guard(x, "*, n_cols")
        return tg.guard(np.concatenate(xs, axis=0), "n_rows, n_cols")

    concat_rows([np.zeros((2, 3))])
    concat_rows([np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))])
    concat_rows([np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((3, 3))])


@must_fail
def test_lists__bad() -> None:
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        tg = mktg()
        for x in xs:
            tg.guard(x, "*, n_cols")
        return tg.guard(np.concatenate(xs, axis=0), "n_rows, n_cols")

    concat_rows([np.zeros((1, 3)), np.zeros((2, 4)), np.zeros((3, 3))])


def test_class_inheritance() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        def f(self, x: np.ndarray) -> np.ndarray:
            tg = mktg()
            tg.guard(x, "n_rows, n_cols_in")
            return tg.guard(np.sum(x, axis=-1)[:, None], "n_rows, 1")

    b = B()
    b.f(np.zeros((2, 3)))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        def f(self, x: np.ndarray) -> np.ndarray:
            tg = mktg()
            tg.guard(x, "n_rows, n_cols_in")
            return tg.guard(np.sum(x, axis=-1), "n_rows")

    b = B()
    b.f(np.zeros((2, 3)))


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            tg = mktg()
            tg.guard(train_features, "n_train_rows, n_features")
            tg.guard(train_labels, "n_train_rows, n_labels")
            self._weights = np.sum(train_labels, axis=0)

        def predict(self, test_features: np.ndarray) -> np.ndarray:
            tg = mktg()
            tg.guard(test_features, "n_test_rows, n_features")
            return tg.guard(
                np.sum(test_features, axis=-1, keepdims=True) + self._weights,
                "n_test_rows, n_labels",
            )

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 3)))


@must_fail
def test_instance_shape__bad_arg() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            tg = mktg()
            tg.guard(train_features, "n_train_rows, n_features")
            tg.guard(train_labels, "n_train_rows, n_labels")
            self._weights = np.sum(train_labels, axis=0)

        def predict(self, test_features: np.ndarray) -> np.ndarray:
            tg = mktg()
            tg.guard(test_features, "n_test_rows, n_features")
            return tg.guard(
                np.sum(test_features, axis=-1, keepdims=True) + self._weights,
                "n_test_rows, n_labels",
            )

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 4)))


@must_fail
def test_instance_shape__bad_return() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            tg = mktg()
            tg.guard(train_features, "n_train_rows, n_features")
            tg.guard(train_labels, "n_train_rows, n_labels")
            self._weights = np.sum(train_labels, axis=0)

        def predict(self, test_features: np.ndarray) -> np.ndarray:
            tg = mktg()
            tg.guard(test_features, "n_test_rows, n_features")
            return tg.guard(
                np.sum(test_features, axis=-1, keepdims=True), "n_test_rows, n_labels"
            )

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 3)))


def test_inner_functions() -> None:
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        tg = mktg()
        tg.guard(target, "n_outputs")

        def loss(prediction: np.ndarray) -> np.ndarray:
            tg.guard(prediction, "n_rows, n_outputs")
            return np.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 3)))


@must_fail
def test_inner_functions__bad_arg() -> None:
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        tg = mktg()
        tg.guard(target, "n_outputs")

        def loss(prediction: np.ndarray) -> np.ndarray:
            tg.guard(prediction, "n_rows, n_outputs")
            return np.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 2)))


def test_intermediate_results() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "n_rows, n_cols")
        a = tg.guard(x, "n_rows, n_cols")
        b = tg.guard(np.sum(a, axis=-1), "n_rows")
        c = np.sum(b, axis=-1)
        return c

    f(np.zeros((2, 3)))


@must_fail
def test_intermediate_results__bad() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        tg = mktg()
        tg.guard(x, "n_rows, n_cols")
        a = tg.guard(x, "n_rows, n_cols")
        b = tg.guard(np.sum(a, axis=-1), "n_cols")
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
