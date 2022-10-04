# For some reason mypy is *super* slow with pytorch.
# type: ignore
# pylint: disable=no-member
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Optional, Tuple

import pytest
import torch
from shapeguard import ShapeGuard


@pytest.fixture(autouse=True)
def reset_shapes() -> Iterable[None]:
    yield
    ShapeGuard.reset()


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(AssertionError):
            f()

    return g


def test_constant_axis() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg([2, 3])
        return torch.zeros((4, 5)).sg([4, 5])

    f(torch.zeros((2, 3)))


@must_fail
def test_constant_axis__bad_arg() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg([2, 3])
        return torch.zeros((4, 5)).sg([4, 5])

    f(torch.zeros((3, 2)))


@must_fail
def test_constant_axis__bad_return() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg([2, 3])
        return torch.zeros((5, 4)).sg([4, 5])

    f(torch.zeros((2, 3)))


def test_variable_axis() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["n_rows_in", "n_cols_in"])
        return torch.zeros((4, 5)).sg(["n_rows_out", "n_cols_out"])

    f(torch.zeros((2, 3)))


@must_fail
def test_variable_axis__bad_arg() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["n_rows_in", "n_cols_in"])
        return torch.zeros((4, 5)).sg(["n_rows_out", "n_cols_out"])

    f(torch.zeros((2,)))


@must_fail
def test_variable_axis__bad_return() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["n_rows_in", "n_cols_in"])
        return torch.zeros((4, 5, 6)).sg(["n_rows_out", "n_cols_out"])

    f(torch.zeros((2, 3)))


def test_correlated_variable_axes() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["n_rows", "n_cols_in"])
        return torch.sum(x, axis=-1)[:, None].sg(["n_rows", 1])

    f(torch.zeros((2, 3)))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["n_rows", "n_cols_in"])
        return torch.sum(x, axis=-1).sg(["n_rows", 1])

    f(torch.zeros((2, 3)))


def test_variable_rank() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["batch...", "n_cols_in"])
        return torch.sum(x, axis=-1)[..., None].sg(["batch...", 1])

    f(torch.zeros((2,)))
    f(torch.zeros((2, 3)))
    f(torch.zeros((2, 3, 4)))


@must_fail
def test_variable_rank__bad() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["batch...", "n_cols_in"])
        return torch.zeros((1, 2, 1)).sg(["batch...", 1])

    f(torch.zeros((2, 3)))


def test_broadcasting() -> None:
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a.sg(["d1", "d2"])
        b.sg(["d1", "d2"])
        return (a + b).sg(["d1", "d2"])

    f(torch.ones((2, 3)), torch.ones((2, 3)))
    f(torch.ones((2, 1)), torch.ones((1, 3)))


@must_fail
def test_broadcasting__bad_arg() -> None:
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a.sg(["d1", "d2"])
        b.sg(["d1", "d2"])
        return (a + b).sg(["d1", "d2"])

    f(torch.ones((2, 2)), torch.ones((1, 3)))


@must_fail
def test_broadcasting__bad_return() -> None:
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a.sg(["d1", "d2"])
        b.sg(["d1", "d2"])
        return torch.zeros((4, 5)).sg(["d1", "d2"])

    f(torch.ones((2, 1)), torch.ones((1, 3)))


def test_tuples() -> None:
    def f(x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x[0].sg(["n_rows", "n_cols0"])
        x[1].sg(["n_rows", "n_cols1"])
        return (
            x[1].sg(["n_rows", "n_cols1"]),
            x[0].sg(["n_rows", "n_cols0"]),
        )

    f((torch.zeros((2, 3)), torch.zeros((2, 4))))


@must_fail
def test_tuples__bad_arg() -> None:
    def f(x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x[0].sg(["n_rows", "n_cols0"])
        x[1].sg(["n_rows", "n_cols1"])
        return (
            x[1].sg(["n_rows", "n_cols1"]),
            x[0].sg(["n_rows", "n_cols0"]),
        )

    f((torch.zeros((2, 3)), torch.zeros((3, 4))))


@must_fail
def test_tuples__bad_return() -> None:
    def f(x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x[0].sg(["n_rows", "n_cols0"])
        x[1].sg(["n_rows", "n_cols1"])
        return (
            x[1].sg(["n_rows", "n_cols1"]),
            x[1].sg(["n_rows", "n_cols0"]),
        )

    f((torch.zeros((2, 3)), torch.zeros((2, 4))))


def test_dataclass() -> None:
    @dataclass
    class Data:
        a: torch.Tensor
        b: torch.Tensor

    def f(x: Data) -> Data:
        x.a.sg(["n_rows", "n_cols0"])
        x.b.sg(["n_rows", "n_cols1"])
        return Data(
            a=x.b.sg(["n_rows", "n_cols1"]),
            b=x.a.sg(["n_rows", "n_cols0"]),
        )

    f(Data(a=torch.zeros((2, 3)), b=torch.zeros((2, 4))))


@must_fail
def test_dataclass__bad_arg() -> None:
    @dataclass
    class Data:
        a: torch.Tensor
        b: torch.Tensor

    def f(x: Data) -> Data:
        x.a.sg(["n_rows", "n_cols0"])
        x.b.sg(["n_rows", "n_cols1"])
        return Data(
            a=x.b.sg(["n_rows", "n_cols1"]),
            b=x.a.sg(["n_rows", "n_cols0"]),
        )

    f(Data(a=torch.zeros((2, 3)), b=torch.zeros((3, 4))))


@must_fail
def test_dataclass__bad_return() -> None:
    @dataclass
    class Data:
        a: torch.Tensor
        b: torch.Tensor

    def f(x: Data) -> Data:
        x.a.sg(["n_rows", "n_cols0"])
        x.b.sg(["n_rows", "n_cols1"])
        return Data(
            a=x.b.sg(["n_rows", "n_cols1"]),
            b=x.b.sg(["n_rows", "n_cols0"]),
        )

    f(Data(a=torch.zeros((2, 3)), b=torch.zeros((2, 4))))


def test_lists() -> None:
    def concat_rows(xs: Iterable[torch.Tensor]) -> torch.Tensor:
        for x in xs:
            x.sg(["*", "n_cols"])
        return torch.concat(xs, axis=0).sg(["n_total_rows", "n_cols"])

    concat_rows([torch.zeros((2, 3))])
    concat_rows([torch.zeros((2, 3)), torch.zeros((2, 3)), torch.zeros((2, 3))])
    concat_rows([torch.zeros((1, 3)), torch.zeros((2, 3)), torch.zeros((3, 3))])


@must_fail
def test_lists__bad() -> None:
    def concat_rows(xs: Iterable[torch.Tensor]) -> torch.Tensor:
        for x in xs:
            x.sg(["*", "n_cols"])
        return torch.concat(xs, axis=0).sg(["n_total_rows", "n_cols"])

    concat_rows([torch.zeros((1, 3)), torch.zeros((2, 4)), torch.zeros((3, 3))])


def test_class_inheritance() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: torch.Tensor) -> torch.Tensor:
            ...

    class B(A):
        def f(self, x: torch.Tensor) -> torch.Tensor:
            x.sg(["n_rows", "n_cols_in"])
            return torch.sum(x, axis=-1)[:, None].sg(["n_rows", 1])

    b = B()
    b.f(torch.zeros((2, 3)))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: torch.Tensor) -> torch.Tensor:
            ...

    class B(A):
        def f(self, x: torch.Tensor) -> torch.Tensor:
            x.sg(["n_rows", "n_cols_in"])
            return torch.sum(x, axis=-1).sg(["n_rows"])

    b = B()
    b.f(torch.zeros((2, 3)))


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[torch.Tensor] = None

        def train(self, train_features: torch.Tensor, train_labels: torch.Tensor) -> None:
            train_features.sg(["n_train_rows", "n_features"])
            train_labels.sg(["n_train_rows", "n_labels"])
            self._weights = torch.sum(train_labels, axis=0).sg(["n_labels"])

        def predict(self, test_features: torch.Tensor) -> torch.Tensor:
            test_features.sg(["n_test_rows", "n_features"])
            return (torch.sum(test_features, axis=-1, keepdims=True) + self._weights).sg(
                ["n_test_rows", "n_labels"]
            )

    a = A()
    a.train(torch.ones((4, 3)), torch.ones((4, 2)))
    a.predict(torch.ones((2, 3)))


@must_fail
def test_instance_shape__bad_arg() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[torch.Tensor] = None

        def train(self, train_features: torch.Tensor, train_labels: torch.Tensor) -> None:
            train_features.sg(["n_train_rows", "n_features"])
            train_labels.sg(["n_train_rows", "n_labels"])
            self._weights = torch.sum(train_labels, axis=0).sg(["n_labels"])

        def predict(self, test_features: torch.Tensor) -> torch.Tensor:
            test_features.sg(["n_test_rows", "n_features"])
            return (torch.sum(test_features, axis=-1, keepdims=True) + self._weights).sg(
                ["n_test_rows", "n_labels"]
            )

    a = A()
    a.train(torch.ones((4, 3)), torch.ones((4, 2)))
    a.predict(torch.ones((2, 4)))


@must_fail
def test_instance_shape__bad_return() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[torch.Tensor] = None

        def train(self, train_features: torch.Tensor, train_labels: torch.Tensor) -> None:
            train_features.sg(["n_train_rows", "n_features"])
            train_labels.sg(["n_train_rows", "n_labels"])
            self._weights = torch.sum(train_labels, axis=0).sg(["n_labels"])

        def predict(self, test_features: torch.Tensor) -> torch.Tensor:
            test_features.sg(["n_test_rows", "n_features"])
            return torch.sum(test_features, axis=-1, keepdims=True).sg(
                ["n_test_rows", "n_labels"]
            )

    a = A()
    a.train(torch.ones((4, 3)), torch.ones((4, 2)))
    a.predict(torch.ones((2, 3)))


def test_inner_functions() -> None:
    def create_loss(
        target: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        target.sg(["n_outputs"])

        def loss(prediction: torch.Tensor) -> torch.Tensor:
            prediction.sg(["n_rows", "n_outputs"])
            return torch.sum((target - prediction) ** 2).sg([])

        return loss

    loss = create_loss(torch.zeros((3,)))
    loss(torch.zeros((4, 3)))


@must_fail
def test_inner_functions__bad_arg() -> None:
    def create_loss(
        target: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        target.sg(["n_outputs"])

        def loss(prediction: torch.Tensor) -> torch.Tensor:
            prediction.sg(["n_rows", "n_outputs"])
            return torch.sum((target - prediction) ** 2).sg([])

        return loss

    loss = create_loss(torch.zeros((3,)))
    loss(torch.zeros((4, 2)))


def test_intermediate_results() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["n_rows", "n_cols"])
        a = x.sg(["n_rows", "n_cols"])
        b = torch.sum(a, axis=-1).sg(["n_rows"])
        c = torch.sum(b, axis=-1).sg([])
        return c.sg([])

    f(torch.zeros((2, 3)))


@must_fail
def test_intermediate_results__bad() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        x.sg(["n_rows", "n_cols"])
        a = x.sg(["n_rows", "n_cols"])
        b = torch.sum(a, axis=-1).sg(["n_cols"])
        c = torch.sum(b, axis=-1).sg([])
        return c.sg([])

    f(torch.zeros((2, 3)))


def test_dtypes() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        return x

    f(torch.zeros((2, 3), dtype=torch.float32))
    f(torch.zeros((2, 3), dtype=torch.float64))


@must_fail
def test_dtypes__bad_type() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        return x

    f(torch.zeros((2, 3), dtype=torch.int32))


@must_fail
def test_dtypes__bad_return() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        return x.astype(torch.float64)

    f(torch.zeros((2, 3), dtype=torch.float32))
