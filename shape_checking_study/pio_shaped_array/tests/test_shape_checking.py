from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pytest
from pio_learning_utilities.arrays.numpy import ShapedNdArray
from pio_learning_utilities.arrays.shaped_array import (
    UnexpectedShapeError,
    fixed_dimension,
    variable_dimension,
)


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(UnexpectedShapeError):
            f()

    return g


class ConstantAxisIn(ShapedNdArray):

    n_rows = fixed_dimension(2)
    n_cols = fixed_dimension(3)


class ConstantAxisOut(ShapedNdArray):

    n_rows = fixed_dimension(4)
    n_cols = fixed_dimension(5)


def test_constant_axis() -> None:
    def f(x: ConstantAxisIn) -> ConstantAxisOut:
        return ConstantAxisOut(np.zeros((4, 5)))

    f(ConstantAxisIn(np.zeros((2, 3))))


@must_fail
def test_constant_axis__bad_arg() -> None:
    def f(x: ConstantAxisIn) -> ConstantAxisOut:
        return ConstantAxisOut(np.zeros((4, 5)))

    f(ConstantAxisIn(np.zeros((3, 2))))


@must_fail
def test_constant_axis__bad_return() -> None:
    def f(x: ConstantAxisIn) -> ConstantAxisOut:
        return ConstantAxisOut(np.zeros((5, 4)))

    f(ConstantAxisIn(np.zeros((2, 3))))


class VariableAxisIn(ShapedNdArray):

    n_rows = variable_dimension()
    n_cols = variable_dimension()


class VariableAxisOut(ShapedNdArray):

    n_rows = variable_dimension()
    n_cols = variable_dimension()


def test_variable_axis() -> None:
    def f(x: VariableAxisIn) -> VariableAxisOut:
        return VariableAxisOut(np.zeros((4, 5)), n_rows=4, n_cols=5)

    f(VariableAxisIn(np.zeros((2, 3)), n_rows=2, n_cols=3))


@must_fail
def test_variable_axis__bad_arg() -> None:
    def f(x: VariableAxisIn) -> VariableAxisOut:
        return VariableAxisOut(np.zeros((4, 5)), n_rows=4, n_cols=5)

    f(VariableAxisIn(np.zeros((2,)), n_rows=2, n_cols=3))


@must_fail
def test_variable_axis__bad_return() -> None:
    def f(x: VariableAxisIn) -> VariableAxisOut:
        return VariableAxisOut(np.zeros((4, 5, 6)), n_rows=4, n_cols=5)

    f(VariableAxisIn(np.zeros((2, 3)), n_rows=2, n_cols=3))


class CorrelatedAxesIn(ShapedNdArray):

    n_rows = variable_dimension()
    n_cols = variable_dimension()


class CorrelatedAxesOut(ShapedNdArray):

    n_rows = variable_dimension()
    n_cols = fixed_dimension(1)


def test_correlated_variable_axes() -> None:
    def f(x: CorrelatedAxesIn) -> CorrelatedAxesOut:
        return CorrelatedAxesOut(np.sum(x.array, axis=-1)[:, None], n_rows=x.n_rows)

    f(CorrelatedAxesIn(np.zeros((2, 3)), n_rows=2, n_cols=3))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    def f(x: CorrelatedAxesIn) -> CorrelatedAxesOut:
        return CorrelatedAxesOut(np.sum(x.array, axis=-1), n_rows=x.n_rows)

    f(CorrelatedAxesIn(np.zeros((2, 3)), n_rows=2, n_cols=3))


class VariableRankIn(ShapedNdArray):

    batch = variable_dimension()  # ???
    n_cols = variable_dimension()


class VariableRankOut(ShapedNdArray):

    batch = variable_dimension()  # ???
    n_cols = fixed_dimension(1)


def test_variable_rank() -> None:
    def f(x: VariableRankIn) -> VariableRankOut:
        return VariableRankOut(np.sum(x.array, axis=-1)[..., None], batch=x.batch)

    f(VariableRankIn(np.zeros((2,)), batch=[], n_cols=2))
    f(VariableRankIn(np.zeros((2, 3)), batch=[2], n_cols=3))
    f(VariableRankIn(np.zeros((2, 3, 4)), batch=[2, 3], n_cols=4))


@must_fail
def test_variable_rank__bad() -> None:
    def f(x: VariableRankIn) -> VariableRankOut:
        return VariableRankOut(np.zeros((1, 2, 1)), batch=x.batch)

    f(VariableRankIn(np.zeros((2, 3)), batch=[2], n_cols=3))


class Broadcasting(ShapedNdArray):

    d1 = variable_dimension()
    d2 = variable_dimension()


def test_broadcasting() -> None:
    def f(a: Broadcasting, b: Broadcasting) -> Broadcasting:
        return Broadcasting(a.array + b.array, d1=max(a.d1, b.d1), d2=max(a.d2, b.d2))

    f(Broadcasting(np.ones((2, 3)), d1=2, d2=3), Broadcasting(np.ones((2, 3)), d1=2, d2=3))
    f(Broadcasting(np.ones((2, 1)), d1=2, d2=1), Broadcasting(np.ones((1, 3)), d1=1, d2=3))


@must_fail
def test_broadcasting__bad_arg() -> None:
    def f(a: Broadcasting, b: Broadcasting) -> Broadcasting:
        return Broadcasting(a.array + b.array, d1=max(a.d1, b.d1), d2=max(a.d2, b.d2))

    f(Broadcasting(np.ones((2, 2)), d1=2, d2=2), Broadcasting(np.ones((2, 3)), d1=2, d2=3))


@must_fail
def test_broadcasting__bad_return() -> None:
    def f(a: Broadcasting, b: Broadcasting) -> Broadcasting:
        return Broadcasting(np.zeros((4, 5)), d1=max(a.d1, b.d1), d2=max(a.d2, b.d2))

    f(Broadcasting(np.ones((2, 1)), d1=2, d2=1), Broadcasting(np.ones((1, 3)), d1=1, d2=3))


class Collection1(ShapedNdArray):

    n_rows = variable_dimension()
    n_cols = variable_dimension()


class Collection2(ShapedNdArray):

    n_rows = variable_dimension()
    n_cols = variable_dimension()


def test_tuples() -> None:
    def f(x: Tuple[Collection1, Collection2]) -> Tuple[Collection2, Collection1]:
        return x[1], x[0]

    f(
        (
            Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            Collection2(np.zeros((2, 4)), n_rows=2, n_cols=4),
        )
    )


@must_fail
def test_tuples__bad_arg() -> None:
    def f(x: Tuple[Collection1, Collection2]) -> Tuple[Collection2, Collection1]:
        return x[1], x[0]

    f(
        (
            Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            Collection2(np.zeros((3, 4)), n_rows=3, n_cols=4),
        )
    )


@must_fail
def test_tuples__bad_return() -> None:
    def f(x: Tuple[Collection1, Collection2]) -> Tuple[Collection2, Collection1]:
        return x[1], x[1]

    f(
        (
            Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            Collection2(np.zeros((2, 4)), n_rows=2, n_cols=4),
        )
    )


@dataclass
class DataIn:
    a: Collection1
    b: Collection2


@dataclass
class DataOut:
    a: Collection2
    b: Collection1


def test_dataclass() -> None:
    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.a)

    f(
        DataIn(
            a=Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            b=Collection2(np.zeros((2, 4)), n_rows=2, n_cols=4),
        )
    )


@must_fail
def test_dataclass__bad_arg() -> None:
    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.a)

    f(
        DataIn(
            a=Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            b=Collection2(np.zeros((3, 4)), n_rows=3, n_cols=4),
        )
    )


@must_fail
def test_dataclass__bad_return() -> None:
    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.b)

    f(
        DataIn(
            a=Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            b=Collection2(np.zeros((2, 4)), n_rows=2, n_cols=4),
        )
    )


def test_lists() -> None:
    def concat_rows(xs: Iterable[Collection1]) -> Collection1:
        return Collection1(
            np.concatenate([x.array for x in xs], axis=0),
            n_rows=sum(x.n_rows for x in xs),
            n_cols=next(iter(xs)).n_cols,
        )

    concat_rows([Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3)])
    concat_rows(
        [
            Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
        ]
    )
    concat_rows(
        [
            Collection1(np.zeros((1, 3)), n_rows=1, n_cols=3),
            Collection1(np.zeros((2, 3)), n_rows=2, n_cols=3),
            Collection1(np.zeros((3, 3)), n_rows=3, n_cols=3),
        ]
    )


@must_fail
def test_lists__bad() -> None:
    def concat_rows(xs: Iterable[Collection1]) -> Collection1:
        return Collection1(
            np.concatenate([x.array for x in xs], axis=0),
            n_rows=sum(x.n_rows for x in xs),
            n_cols=next(iter(xs)).n_cols,
        )

    concat_rows(
        [
            Collection1(np.zeros((1, 3)), n_rows=1, n_cols=3),
            Collection1(np.zeros((2, 4)), n_rows=2, n_cols=4),
            Collection1(np.zeros((3, 3)), n_rows=3, n_cols=3),
        ]
    )


def test_class_inheritance() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: CorrelatedAxesIn) -> CorrelatedAxesOut:
            ...

    class B(A):
        def f(self, x: CorrelatedAxesIn) -> CorrelatedAxesOut:
            return CorrelatedAxesOut(np.sum(x.array, axis=-1)[:, None], n_rows=x.n_rows)

    b = B()
    b.f(CorrelatedAxesIn(np.zeros((2, 3)), n_rows=2, n_cols=3))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class MyAxesOut(ShapedNdArray):

        n_rows = variable_dimension()

    class A(ABC):
        @abstractmethod
        def f(self, x: CorrelatedAxesIn) -> CorrelatedAxesOut:
            ...

    class B(A):
        def f(self, x: CorrelatedAxesIn) -> MyAxesOut:
            return MyAxesOut(np.sum(x.array, axis=-1), n_rows=x.n_rows)

    b = B()
    b.f(CorrelatedAxesIn(np.zeros((2, 3)), n_rows=2, n_cols=3))


class Weights(ShapedNdArray):

    n_labels = variable_dimension()


class Features(ShapedNdArray):

    n_rows = variable_dimension()
    n_features = variable_dimension()


class Labels(ShapedNdArray):

    n_rows = variable_dimension()
    n_labels = variable_dimension()


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[Weights] = None

        def train(self, train_features: Features, train_labels: Labels) -> None:
            self._weights = Weights(
                np.sum(train_labels.array, axis=0), n_labels=train_labels.n_labels
            )

        def predict(self, test_features: Features) -> Labels:
            return Labels(
                np.sum(test_features.array, axis=-1, keepdims=True) + self._weights.array,
                n_rows=test_features.n_rows,
                n_labels=self._weights.n_labels,
            )

    a = A()
    a.train(
        Features(np.ones((4, 3)), n_rows=4, n_features=3),
        Labels(np.ones((4, 2)), n_rows=4, n_labels=2),
    )
    a.predict(Features(np.ones((2, 3)), n_rows=2, n_features=3))


@must_fail
def test_instance_shape__bad_arg() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[Weights] = None

        def train(self, train_features: Features, train_labels: Labels) -> None:
            self._weights = Weights(
                np.sum(train_labels.array, axis=0), n_labels=train_labels.n_labels
            )

        def predict(self, test_features: Features) -> Labels:
            return Labels(
                np.sum(test_features.array, axis=-1, keepdims=True) + self._weights.array,
                n_rows=test_features.n_rows,
                n_labels=self._weights.n_labels,
            )

    a = A()
    a.train(
        Features(np.ones((4, 3)), n_rows=4, n_features=3),
        Labels(np.ones((4, 2)), n_rows=4, n_labels=2),
    )
    a.predict(Features(np.ones((2, 4)), n_rows=2, n_features=4))


@must_fail
def test_instance_shape__bad_return() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[Weights] = None

        def train(self, train_features: Features, train_labels: Labels) -> None:
            self._weights = Weights(
                np.sum(train_labels.array, axis=0), n_labels=train_labels.n_labels
            )

        def predict(self, test_features: Features) -> Labels:
            return Labels(
                np.sum(test_features.array, axis=-1, keepdims=True),
                n_rows=test_features.n_rows,
                n_labels=self._weights.n_labels,
            )

    a = A()
    a.train(
        Features(np.ones((4, 3)), n_rows=4, n_features=3),
        Labels(np.ones((4, 2)), n_rows=4, n_labels=2),
    )
    a.predict(Features(np.ones((2, 3)), n_rows=2, n_features=3))


class Prediction(ShapedNdArray):

    n_rows = variable_dimension()
    n_outputs = variable_dimension()


class Target(ShapedNdArray):

    n_outputs = variable_dimension()


class Loss(ShapedNdArray):

    pass


def test_inner_functions() -> None:
    def create_loss(
        target: Target,
    ) -> Callable[[Prediction], Loss]:
        def loss(prediction: Prediction) -> Loss:
            return Loss(np.sum((target.array - prediction.array) ** 2))

        return loss

    loss = create_loss(Target(np.zeros((3,)), n_outputs=3))
    loss(Prediction(np.zeros((4, 3)), n_rows=4, n_outputs=3))


@must_fail
def test_inner_functions__bad_arg() -> None:
    def create_loss(
        target: Target,
    ) -> Callable[[Prediction], Loss]:
        def loss(prediction: Prediction) -> Loss:
            return Loss(np.sum((target - prediction) ** 2))

        return loss

    loss = create_loss(Target(np.zeros((3,)), n_outputs=3))
    loss(Target(np.zeros((4, 2)), n_rows=4, n_outputs=2))


class ArrayA(ShapedNdArray):

    n_rows = variable_dimension()
    n_cols = variable_dimension()


class ArrayB(ShapedNdArray):

    n_rows = variable_dimension()


class ArrayC(ShapedNdArray):

    pass


def test_intermediate_results() -> None:
    def f(x: ArrayA) -> ArrayC:
        a: ArrayA = x
        b = ArrayB(np.sum(a.array, axis=-1), n_rows=a.n_rows)
        c = ArrayC(np.sum(b.array, axis=-1))
        return c

    f(ArrayA(np.zeros((2, 3)), n_rows=2, n_cols=3))


@must_fail
def test_intermediate_results__bad() -> None:
    def f(x: ArrayA) -> ArrayC:
        a: ArrayA = x
        b = ArrayB(np.sum(a.array, axis=0), n_rows=a.n_rows)
        c = ArrayC(np.sum(b.array, axis=0))
        return c

    f(ArrayA(np.zeros((2, 3)), n_rows=2, n_cols=3))


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
