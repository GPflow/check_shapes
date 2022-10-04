# For some reason mypy is *super* slow with pytorch.
# type: ignore
# pylint: disable=no-member
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Tuple

import pytest
import torch
from typeguard import typechecked

from torchtyping import TensorType, patch_typeguard

patch_typeguard()


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(TypeError):
            f()

    return g


def test_constant_axis() -> None:
    @typechecked
    def f(x: TensorType[2, 3]) -> TensorType[4, 5]:
        return torch.zeros((4, 5))

    f(torch.zeros((2, 3)))


@must_fail
def test_constant_axis__bad_arg() -> None:
    @typechecked
    def f(x: TensorType[2, 3]) -> TensorType[4, 5]:
        return torch.zeros((4, 5))

    f(torch.zeros((3, 2)))


@must_fail
def test_constant_axis__bad_return() -> None:
    @typechecked
    def f(x: TensorType[2, 3]) -> TensorType[4, 5]:
        return torch.zeros((5, 4))

    f(torch.zeros((2, 3)))


def test_variable_axis() -> None:
    @typechecked
    def f(x: TensorType["n_rows_in", "n_cols_in"]) -> TensorType["n_rows_out", "n_cols_out"]:
        return torch.zeros((4, 5))

    f(torch.zeros((2, 3)))


@must_fail
def test_variable_axis__bad_arg() -> None:
    @typechecked
    def f(x: TensorType["n_rows_in", "n_cols_in"]) -> TensorType["n_rows_out", "n_cols_out"]:
        return torch.zeros((4, 5))

    f(torch.zeros((2,)))


@must_fail
def test_variable_axis__bad_return() -> None:
    @typechecked
    def f(x: TensorType["n_rows_in", "n_cols_in"]) -> TensorType["n_rows_out", "n_cols_out"]:
        return torch.zeros((4, 5, 6))

    f(torch.zeros((2, 3)))


def test_correlated_variable_axes() -> None:
    @typechecked
    def f(x: TensorType["n_rows", "n_cols_in"]) -> TensorType["n_rows", 1]:
        return torch.sum(x, axis=-1)[:, None]

    f(torch.zeros((2, 3)))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    @typechecked
    def f(x: TensorType["n_rows", "n_cols_in"]) -> TensorType["n_rows", 1]:
        return torch.sum(x, axis=-1)

    f(torch.zeros((2, 3)))


def test_variable_rank() -> None:
    @typechecked
    def f(x: TensorType["batch":..., "n_cols_in"]) -> TensorType["batch":..., 1]:
        return torch.sum(x, axis=-1)[..., None]

    f(torch.zeros((2,)))
    f(torch.zeros((2, 3)))
    f(torch.zeros((2, 3, 4)))


@must_fail
def test_variable_rank__bad() -> None:
    @typechecked
    def f(x: TensorType["batch":..., "n_cols_in"]) -> TensorType["batch":..., 1]:
        return torch.zeros((1, 2, 1))

    f(torch.zeros((2, 3)))


def test_broadcasting() -> None:
    @typechecked
    def f(a: TensorType["d1", "d2"], b: TensorType["d1", "d2"]) -> TensorType["d1", "d2"]:
        return a + b

    f(torch.ones((2, 3)), torch.ones((2, 3)))
    f(torch.ones((2, 1)), torch.ones((1, 3)))


@must_fail
def test_broadcasting__bad_arg() -> None:
    @typechecked
    def f(a: TensorType["d1", "d2"], b: TensorType["d1", "d2"]) -> TensorType["d1", "d2"]:
        return a + b

    f(torch.ones((2, 2)), torch.ones((1, 3)))


@must_fail
def test_broadcasting__bad_return() -> None:
    @typechecked
    def f(a: TensorType["d1", "d2"], b: TensorType["d1", "d2"]) -> TensorType["d1", "d2"]:
        return torch.zeros((4, 5))

    f(torch.ones((2, 1)), torch.ones((1, 3)))


def test_tuples() -> None:
    @typechecked
    def f(
        x: Tuple[TensorType["n_rows", "n_cols0"], TensorType["n_rows", "n_cols1"]]
    ) -> Tuple[TensorType["n_rows", "n_cols1"], TensorType["n_rows", "n_cols0"]]:
        return x[1], x[0]

    f((torch.zeros((2, 3)), torch.zeros((2, 4))))


@must_fail
def test_tuples__bad_arg() -> None:
    @typechecked
    def f(
        x: Tuple[TensorType["n_rows", "n_cols0"], TensorType["n_rows", "n_cols1"]]
    ) -> Tuple[TensorType["n_rows", "n_cols1"], TensorType["n_rows", "n_cols0"]]:
        return x[1], x[0]

    f((torch.zeros((2, 3)), torch.zeros((3, 4))))


@must_fail
def test_tuples__bad_return() -> None:
    @typechecked
    def f(
        x: Tuple[TensorType["n_rows", "n_cols0"], TensorType["n_rows", "n_cols1"]]
    ) -> Tuple[TensorType["n_rows", "n_cols1"], TensorType["n_rows", "n_cols0"]]:
        return x[1], x[1]

    f((torch.zeros((2, 3)), torch.zeros((2, 4))))


def test_dataclass() -> None:
    @dataclass
    class DataIn:
        a: TensorType["n_rows", "n_cols0"]
        b: TensorType["n_rows", "n_cols1"]

    @dataclass
    class DataOut:
        a: TensorType["n_rows", "n_cols1"]
        b: TensorType["n_rows", "n_cols0"]

    @typechecked
    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.a)

    f(DataIn(a=torch.zeros((2, 3)), b=torch.zeros((2, 4))))


@must_fail
def test_dataclass__bad_arg() -> None:
    @dataclass
    class DataIn:
        a: TensorType["n_rows", "n_cols0"]
        b: TensorType["n_rows", "n_cols1"]

    @dataclass
    class DataOut:
        a: TensorType["n_rows", "n_cols1"]
        b: TensorType["n_rows", "n_cols0"]

    @typechecked
    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.a)

    f(DataIn(a=torch.zeros((2, 3)), b=torch.zeros((3, 4))))


@must_fail
def test_dataclass__bad_return() -> None:
    @dataclass
    class DataIn:
        a: TensorType["n_rows", "n_cols0"]
        b: TensorType["n_rows", "n_cols1"]

    @dataclass
    class DataOut:
        a: TensorType["n_rows", "n_cols1"]
        b: TensorType["n_rows", "n_cols0"]

    @typechecked
    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.b)

    f(DataIn(a=torch.zeros((2, 3)), b=torch.zeros((2, 4))))


def test_lists() -> None:
    @typechecked
    def concat_rows(
        xs: Iterable[TensorType["n_rows":-1, "n_cols"]]
    ) -> TensorType["n_total_rows", "n_cols"]:
        return torch.concat(xs, axis=0)

    concat_rows([torch.zeros((2, 3))])
    concat_rows([torch.zeros((2, 3)), torch.zeros((2, 3)), torch.zeros((2, 3))])
    concat_rows([torch.zeros((1, 3)), torch.zeros((2, 3)), torch.zeros((3, 3))])


@must_fail
def test_lists__bad() -> None:
    @typechecked
    def concat_rows(
        xs: Iterable[TensorType["n_rows":-1, "n_cols"]]
    ) -> TensorType["n_total_rows", "n_cols"]:
        return torch.concat(xs, axis=0)

    concat_rows([torch.zeros((1, 3)), torch.zeros((2, 4)), torch.zeros((3, 3))])


def test_class_inheritance() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: TensorType["n_rows", "n_cols_in"]) -> TensorType["n_rows", 1]:
            ...

    class B(A):
        @typechecked
        def f(self, x: TensorType["n_rows", "n_cols_in"]) -> TensorType["n_rows", 1]:
            return torch.sum(x, axis=-1)[:, None]

    b = B()
    b.f(torch.zeros((2, 3)))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: TensorType["n_rows", "n_cols_in"]) -> TensorType["n_rows", 1]:
            ...

    class B(A):
        @typechecked
        def f(self, x: TensorType["n_rows", "n_cols_in"]) -> TensorType["n_rows"]:
            return torch.sum(x, axis=-1)

    b = B()
    b.f(torch.zeros((2, 3)))


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[TensorType["n_labels"]] = None

        @typechecked
        def train(
            self,
            train_features: TensorType["n_train_rows", "n_features"],
            train_labels: TensorType["n_train_rows", "n_labels"],
        ) -> None:
            self._weights = torch.sum(train_labels, axis=0)

        @typechecked
        def predict(
            self, test_features: TensorType["n_test_rows", "n_features"]
        ) -> TensorType["n_test_rows", "n_labels"]:
            return torch.sum(test_features, axis=-1, keepdims=True) + self._weights

    a = A()
    a.train(torch.ones((4, 3)), torch.ones((4, 2)))
    a.predict(torch.ones((2, 3)))


@must_fail
def test_instance_shape__bad_arg() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[TensorType["n_labels"]] = None

        @typechecked
        def train(
            self,
            train_features: TensorType["n_train_rows", "n_features"],
            train_labels: TensorType["n_train_rows", "n_labels"],
        ) -> None:
            self._weights = torch.sum(train_labels, axis=0)

        @typechecked
        def predict(
            self, test_features: TensorType["n_test_rows", "n_features"]
        ) -> TensorType["n_test_rows", "n_labels"]:
            return torch.sum(test_features, axis=-1, keepdims=True) + self._weights

    a = A()
    a.train(torch.ones((4, 3)), torch.ones((4, 2)))
    a.predict(torch.ones((2, 4)))


@must_fail
def test_instance_shape__bad_return() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[TensorType["n_labels"]] = None

        @typechecked
        def train(
            self,
            train_features: TensorType["n_train_rows", "n_features"],
            train_labels: TensorType["n_train_rows", "n_labels"],
        ) -> None:
            self._weights = torch.sum(train_labels, axis=0)

        @typechecked
        def predict(
            self, test_features: TensorType["n_test_rows", "n_features"]
        ) -> TensorType["n_test_rows", "n_labels"]:
            return torch.sum(test_features, axis=-1, keepdims=True)

    a = A()
    a.train(torch.ones((4, 3)), torch.ones((4, 2)))
    a.predict(torch.ones((2, 3)))


def test_inner_functions() -> None:
    @typechecked
    def create_loss(
        target: TensorType["n_outputs"],
    ) -> Callable[[TensorType["n_rows", "n_outputs"]], TensorType[()]]:
        @typechecked
        def loss(prediction: TensorType["n_rows", "n_outputs"]) -> TensorType[()]:
            return torch.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(torch.zeros((3,)))
    loss(torch.zeros((4, 3)))


@must_fail
def test_inner_functions__bad_arg() -> None:
    @typechecked
    def create_loss(
        target: TensorType["n_outputs"],
    ) -> Callable[[TensorType["n_rows", "n_outputs"]], TensorType[()]]:
        @typechecked
        def loss(prediction: TensorType["n_rows", "n_outputs"]) -> TensorType[()]:
            return torch.sum((target - prediction) ** 2)

        return loss

    loss = create_loss(torch.zeros((3,)))
    loss(torch.zeros((4, 2)))


def test_intermediate_results() -> None:
    @typechecked
    def f(x: TensorType["n_rows", "n_cols"]) -> TensorType[()]:
        a: TensorType["n_rows", "n_cols"] = x
        b: TensorType["n_rows"] = torch.sum(a, axis=-1)
        c: TensorType[()] = torch.sum(b, axis=-1)
        return c

    f(torch.zeros((2, 3)))


@must_fail
def test_intermediate_results__bad() -> None:
    @typechecked
    def f(x: TensorType["n_rows", "n_cols"]) -> TensorType[()]:
        a: TensorType["n_rows", "n_cols"] = x
        b: TensorType["n_rows"] = torch.sum(a, axis=-1)
        c: TensorType[()] = torch.sum(b, axis=-1)
        return c

    f(torch.zeros((2, 3)))


def test_dtypes() -> None:
    @typechecked
    def f(x: TensorType[Any, float]) -> TensorType[Any, float]:
        return x

    f(torch.zeros((2, 3), dtype=torch.float32))
    f(torch.zeros((2, 3), dtype=torch.float64))


@must_fail
def test_dtypes__bad_type() -> None:
    @typechecked
    def f(x: TensorType[Any, float]) -> TensorType[Any, float]:
        return x

    f(torch.zeros((2, 3), dtype=torch.int32))


@must_fail
def test_dtypes__bad_return() -> None:
    @typechecked
    def f(x: TensorType[Any, float]) -> TensorType[Any, float]:
        return x.astype(torch.float64)

    f(torch.zeros((2, 3), dtype=torch.float32))
