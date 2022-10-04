from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pytest
from contracts import ContractsMeta, contract
from contracts.interface import ContractNotRespected


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(ContractNotRespected):
            f()

    return g


def test_constant_axis() -> None:
    @contract(x="array[2x3]", returns="array[4x5]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2, 3)))


@must_fail
def test_constant_axis__bad_arg() -> None:
    @contract(x="array[2x3]", returns="array[4x5]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((3, 2)))


@must_fail
def test_constant_axis__bad_return() -> None:
    @contract(x="array[2x3]", returns="array[4x5]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((5, 4))

    f(np.zeros((2, 3)))


def test_variable_axis() -> None:
    @contract(x="array[RxC]", returns="array[SxD]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2, 3)))


@must_fail
def test_variable_axis__bad_arg() -> None:
    @contract(x="array[RxC]", returns="array[SxD]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.zeros((2,)))


@must_fail
def test_variable_axis__bad_return() -> None:
    @contract(x="array[RxC]", returns="array[SxD]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5, 6))

    f(np.zeros((2, 3)))


def test_correlated_variable_axes() -> None:
    @contract(x="array[RxC]", returns="array[Rx1]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)[:, None]

    f(np.zeros((2, 3)))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    @contract(x="array[RxC]", returns="array[Rx1]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)

    f(np.zeros((2, 3)))


def test_variable_rank() -> None:
    @contract(x="array[Cx...]", returns="array[1x...]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=-1)[..., None]

    f(np.zeros((2,)))
    f(np.zeros((2, 3)))
    f(np.zeros((2, 3, 4)))


@must_fail
def test_variable_rank__bad() -> None:
    @contract(x="array[Cx...]", returns="array[1x...]")
    def f(x: np.ndarray) -> np.ndarray:
        return np.zeros((1, 2, 1))

    f(np.zeros((2, 3)))


def test_broadcasting() -> None:
    @contract(a="array[AxB]", b="array[AxB]", returns="array[AxB]")
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    f(np.ones((2, 3)), np.ones((2, 3)))
    f(np.ones((2, 1)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_arg() -> None:
    @contract(a="array[AxB]", b="array[AxB]", returns="array[AxB]")
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    f(np.ones((2, 2)), np.ones((1, 3)))


@must_fail
def test_broadcasting__bad_return() -> None:
    @contract(a="array[AxB]", b="array[AxB]", returns="array[AxB]")
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.zeros((4, 5))

    f(np.ones((2, 1)), np.ones((1, 3)))


def test_tuples() -> None:
    @contract(x="tuple(array[RxC],array[RxD])", returns="tuple(array[RxD], array[RxC])")
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[0]

    f((np.zeros((2, 3)), np.zeros((2, 4))))


@must_fail
def test_tuples__bad_arg() -> None:
    @contract(x="tuple(array[RxC],array[RxD])", returns="tuple(array[RxD], array[RxC])")
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[0]

    f((np.zeros((2, 3)), np.zeros((3, 4))))


@must_fail
def test_tuples__bad_return() -> None:
    @contract(x="tuple(array[RxC],array[RxD])", returns="tuple(array[RxD], array[RxC])")
    def f(x: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return x[1], x[1]

    f((np.zeros((2, 3)), np.zeros((2, 4))))


def test_dataclass() -> None:
    @dataclass
    class Data:
        a: np.ndarray
        b: np.ndarray

    @contract(
        x="attr(a:array[RxC]),attr(b:array[RxD])",
        returns="attr(a:array[RxD]),attr(b:array[RxC])",
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

    @contract(
        x="attr(a:array[RxC]),attr(b:array[RxD])",
        returns="attr(a:array[RxD]),attr(b:array[RxC])",
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

    @contract(
        x="attr(a:array[RxC]),attr(b:array[RxD])",
        returns="attr(a:array[RxD]),attr(b:array[RxC])",
    )
    def f(x: Data) -> Data:
        return Data(a=x.b, b=x.b)

    f(Data(a=np.zeros((2, 3)), b=np.zeros((2, 4))))


def test_lists() -> None:
    @contract(xs="seq(array[*xC])", returns="array[RxC]")
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0)

    concat_rows([np.zeros((2, 3))])
    concat_rows([np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))])
    concat_rows([np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((3, 3))])


@must_fail
def test_lists__bad() -> None:
    @contract(xs="seq(array[*xC])", returns="array[RxC]")
    def concat_rows(xs: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0)

    concat_rows([np.zeros((1, 3)), np.zeros((2, 4)), np.zeros((3, 3))])


def test_class_inheritance() -> None:
    class A(metaclass=ContractsMeta):
        @abstractmethod
        @contract(x="array[RxC]", returns="array[Rx1]")
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        def f(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=-1)[:, None]

    b = B()
    b.f(np.zeros((2, 3)))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class A(ABC):
        @abstractmethod
        @contract(x="array[RxC]", returns="array[Rx1]")
        def f(self, x: np.ndarray) -> np.ndarray:
            ...

    class B(A):
        def f(self, x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=-1)

    b = B()
    b.f(np.zeros((2, 3)))


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[np.ndarray] = None

        @contract(train_features="array[RxF]", train_labels="array[RxL]")
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @contract(test_features="array[RxF]", returns="array[RxL]")
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

        @contract(train_features="array[RxF]", train_labels="array[RxL]")
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @contract(test_features="array[RxF]", returns="array[RxL]")
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

        @contract(train_features="array[RxF]", train_labels="array[RxL]")
        def train(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
            self._weights = np.sum(train_labels, axis=0)

        @contract(test_features="array[RxF]", returns="array[RxL]")
        def predict(self, test_features: np.ndarray) -> np.ndarray:
            return np.sum(test_features, axis=-1, keepdims=True)

    a = A()
    a.train(np.ones((4, 3)), np.ones((4, 2)))
    a.predict(np.ones((2, 3)))


def test_inner_functions() -> None:
    @contract(x="array[O]")
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        @contract(x="array[RxO]", returns="array[]")
        def loss(prediction: np.ndarray) -> np.ndarray:
            return np.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 3)))


@must_fail
def test_inner_functions__bad_arg() -> None:
    @contract(x="array[O]")
    def create_loss(
        target: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        @contract(x="array[RxO]", returns="array[]")
        def loss(prediction: np.ndarray) -> np.ndarray:
            return np.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(np.zeros((3,)))
    loss(np.zeros((4, 2)))


def test_intermediate_results() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        a = x
        b = np.sum(a, axis=-1)
        c = np.sum(b, axis=-1)
        return c

    f(np.zeros((2, 3)))


@must_fail
def test_intermediate_results__bad() -> None:
    def f(x: np.ndarray) -> np.ndarray:
        a = x
        b = np.sum(a, axis=-1)
        c = np.sum(b, axis=-1)
        return c

    f(np.zeros((2, 3)))


def test_dtypes() -> None:
    @contract(x="array(float32|float64)", returns="array(float32|float64)")
    def f(x: np.ndarray) -> np.ndarray:
        return x

    f(np.zeros((2, 3), dtype=np.float32))
    f(np.zeros((2, 3), dtype=np.float64))


@must_fail
def test_dtypes__bad_type() -> None:
    @contract(x="array(float32|float64)", returns="array(float32|float64)")
    def f(x: np.ndarray) -> np.ndarray:
        return x

    f(np.zeros((2, 3), dtype=np.int32))


@must_fail
def test_dtypes__bad_return() -> None:
    @contract(x="array(float32|float64)", returns="array(float32|float64)")
    def f(x: np.ndarray) -> np.ndarray:
        return x.astype(np.float64)

    f(np.zeros((2, 3), dtype=np.float32))
