from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable, NewType, Optional, Tuple

import numpy as np
import pytest
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from tensor_annotations import axes

NRows = NewType("NRows", axes.Axis)
NRowsIn = NewType("NRowsIn", axes.Axis)
NRowsOut = NewType("NRowsOut", axes.Axis)
NCols = NewType("NCols", axes.Axis)
NColsIn = NewType("NColsIn", axes.Axis)
NColsOut = NewType("NColsOut", axes.Axis)
NCols0 = NewType("NCols0", axes.Axis)
NCols1 = NewType("NCols1", axes.Axis)
NFeatures = NewType("NFeatures", axes.Axis)
NLabels = NewType("NLabels", axes.Axis)
NOutputs = NewType("NOutputs", axes.Axis)
N1 = NewType("N1", axes.Axis)
N2 = NewType("N2", axes.Axis)
N3 = NewType("N3", axes.Axis)
N4 = NewType("N4", axes.Axis)
N5 = NewType("N5", axes.Axis)
D1 = NewType("D1", axes.Axis)
D2 = NewType("D2", axes.Axis)


class ShapeError(Exception):
    """ Place-holder for whatever error your shape checker raises. """


def must_fail(f: Callable[[], None]) -> Callable[[], None]:
    @wraps(f)
    def g() -> None:
        with pytest.raises(ShapeError):
            f()

    return g


def test_constant_axis() -> None:
    def f(x: ttf.Tensor2[N2, N3]) -> ttf.Tensor2[N4, N5]:
        return tf.zeros((4, 5))

    f(tf.zeros((2, 3)))


@must_fail
def test_constant_axis__bad_arg() -> None:
    def f(x: ttf.Tensor2[N2, N3]) -> ttf.Tensor2[N4, N5]:
        return tf.zeros((4, 5))

    f(tf.zeros((3, 2)))


@must_fail
def test_constant_axis__bad_return() -> None:
    def f(x: ttf.Tensor2[N2, N3]) -> ttf.Tensor2[N4, N5]:
        return tf.zeros((5, 4))

    f(tf.zeros((2, 3)))


def test_variable_axis() -> None:
    def f(x: ttf.Tensor2[NRowsIn, NColsIn]) -> ttf.Tensor2[NRowsOut, NColsOut]:
        return tf.zeros((4, 5))

    f(tf.zeros((2, 3)))


@must_fail
def test_variable_axis__bad_arg() -> None:
    def f(x: ttf.Tensor2[NRowsIn, NColsIn]) -> ttf.Tensor2[NRowsOut, NColsOut]:
        return tf.zeros((4, 5))

    f(tf.zeros((2,)))

    raise ShapeError(
        'Argument 1 to "f" has incompatible type "Tensor1[Any]"; expected "Tensor2[NRowsIn, NColsIn]"'
    )


@must_fail
def test_variable_axis__bad_return() -> None:
    def f(x: ttf.Tensor2[NRowsIn, NColsIn]) -> ttf.Tensor2[NRowsOut, NColsOut]:
        return tf.zeros((4, 5, 6))

    f(tf.zeros((2, 3)))

    raise ShapeError(
        'Incompatible return value type (got "Tensor3[Any, Any, Any]", expected "Tensor2[NRowsOut, NColsOut]")'
    )


def test_correlated_variable_axes() -> None:
    def f(x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, N1]:
        return tf.reduce_sum(x, axis=-1)[:, None]

    f(tf.zeros((2, 3)))


@must_fail
def test_correlated_variable_axes__bad() -> None:
    def f(x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, N1]:
        return tf.reduce_sum(x, axis=-1)

    f(tf.zeros((2, 3)))


def test_variable_rank() -> None:
    def f(x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(x, axis=-1)[..., None]

    f(tf.zeros((2,)))
    f(tf.zeros((2, 3)))
    f(tf.zeros((2, 3, 4)))


@must_fail
def test_variable_rank__bad() -> None:
    def f(x: tf.Tensor) -> tf.Tensor:
        return tf.zeros((1, 2, 1))

    f(tf.zeros((2, 3)))


def test_broadcasting() -> None:
    def f(a: ttf.Tensor2[D1, D2], b: ttf.Tensor2[D1, D2]) -> ttf.Tensor2[D1, D2]:
        return a + b

    f(tf.ones((2, 3)), tf.ones((2, 3)))
    f(tf.ones((2, 1)), tf.ones((1, 3)))


@must_fail
def test_broadcasting__bad_arg() -> None:
    def f(a: ttf.Tensor2[D1, D2], b: ttf.Tensor2[D1, D2]) -> ttf.Tensor2[D1, D2]:
        return a + b

    f(tf.ones((2, 2)), tf.ones((1, 3)))


@must_fail
def test_broadcasting__bad_return() -> None:
    def f(a: ttf.Tensor2[D1, D2], b: ttf.Tensor2[D1, D2]) -> ttf.Tensor2[D1, D2]:
        return tf.zeros((4, 5))

    f(tf.ones((2, 1)), tf.ones((1, 3)))


def test_tuples() -> None:
    def f(
        x: Tuple[ttf.Tensor2[NRows, NCols0], ttf.Tensor2[NRows, NCols1]]
    ) -> Tuple[ttf.Tensor2[NRows, NCols1], ttf.Tensor2[NRows, NCols0]]:
        return x[1], x[0]

    f((tf.zeros((2, 3)), tf.zeros((2, 4))))


@must_fail
def test_tuples__bad_arg() -> None:
    def f(
        x: Tuple[ttf.Tensor2[NRows, NCols0], ttf.Tensor2[NRows, NCols1]]
    ) -> Tuple[ttf.Tensor2[NRows, NCols1], ttf.Tensor2[NRows, NCols0]]:
        return x[1], x[0]

    f((tf.zeros((2, 3)), tf.zeros((3, 4))))


@must_fail
def test_tuples__bad_return() -> None:
    def f(
        x: Tuple[ttf.Tensor2[NRows, NCols0], ttf.Tensor2[NRows, NCols1]]
    ) -> Tuple[ttf.Tensor2[NRows, NCols1], ttf.Tensor2[NRows, NCols0]]:
        return x[1], x[1]

    f((tf.zeros((2, 3)), tf.zeros((2, 4))))


def test_dataclass() -> None:
    @dataclass
    class DataIn:
        a: ttf.Tensor2[NRows, NCols0]
        b: ttf.Tensor2[NRows, NCols1]

    @dataclass
    class DataOut:
        a: ttf.Tensor2[NRows, NCols1]
        b: ttf.Tensor2[NRows, NCols0]

    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.a)

    f(DataIn(a=tf.zeros((2, 3)), b=tf.zeros((2, 4))))


@must_fail
def test_dataclass__bad_arg() -> None:
    @dataclass
    class DataIn:
        a: ttf.Tensor2[NRows, NCols0]
        b: ttf.Tensor2[NRows, NCols1]

    @dataclass
    class DataOut:
        a: ttf.Tensor2[NRows, NCols1]
        b: ttf.Tensor2[NRows, NCols0]

    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.a)

    f(DataIn(a=tf.zeros((2, 3)), b=tf.zeros((3, 4))))


@must_fail
def test_dataclass__bad_return() -> None:
    @dataclass
    class DataIn:
        a: ttf.Tensor2[NRows, NCols0]
        b: ttf.Tensor2[NRows, NCols1]

    @dataclass
    class DataOut:
        a: ttf.Tensor2[NRows, NCols1]
        b: ttf.Tensor2[NRows, NCols0]

    def f(x: DataIn) -> DataOut:
        return DataOut(a=x.b, b=x.b)

    f(DataIn(a=tf.zeros((2, 3)), b=tf.zeros((2, 4))))


def test_lists() -> None:
    def concat_rows(xs: Iterable[ttf.Tensor2[NRows, NCols]]) -> ttf.Tensor2[NRows, NCols]:
        return tf.concat(xs, axis=0)

    concat_rows([tf.zeros((2, 3))])
    concat_rows([tf.zeros((2, 3)), tf.zeros((2, 3)), tf.zeros((2, 3))])
    concat_rows([tf.zeros((1, 3)), tf.zeros((2, 3)), tf.zeros((3, 3))])


@must_fail
def test_lists__bad() -> None:
    def concat_rows(xs: Iterable[ttf.Tensor2[NRows, NCols]]) -> ttf.Tensor2[NRows, NCols]:
        return tf.concat(xs, axis=0)

    concat_rows([tf.zeros((1, 3)), tf.zeros((2, 4)), tf.zeros((3, 3))])


def test_class_inheritance() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, N1]:
            ...

    class B(A):
        def f(self, x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, N1]:
            return tf.reduce_sum(x, axis=-1)[:, None]

    b = B()
    b.f(tf.zeros((2, 3)))


@must_fail
def test_class_inheritance__bad_def() -> None:
    class A(ABC):
        @abstractmethod
        def f(self, x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, N1]:
            ...

    class B(A):
        def f(self, x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor1[NRows]:
            return tf.reduce_sum(x, axis=-1)

    b = B()
    b.f(tf.zeros((2, 3)))


def test_instance_shape() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[ttf.Tensor1[NLabels]] = None

        def train(
            self,
            train_features: ttf.Tensor2[NRows, NFeatures],
            train_labels: ttf.Tensor2[NRows, NLabels],
        ) -> None:
            self._weights = tf.reduce_sum(train_labels, axis=0)

        def predict(
            self, test_features: ttf.Tensor2[NRows, NFeatures]
        ) -> ttf.Tensor2[NRows, NLabels]:
            return tf.reduce_sum(test_features, axis=-1, keepdims=True) + self._weights

    a = A()
    a.train(tf.ones((4, 3)), tf.ones((4, 2)))
    a.predict(tf.ones((2, 3)))


@must_fail
def test_instance_shape__bad_arg() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[ttf.Tensor1[NLabels]] = None

        def train(
            self,
            train_features: ttf.Tensor2[NRows, NFeatures],
            train_labels: ttf.Tensor2[NRows, NLabels],
        ) -> None:
            self._weights = tf.reduce_sum(train_labels, axis=0)

        def predict(
            self, test_features: ttf.Tensor2[NRows, NFeatures]
        ) -> ttf.Tensor2[NRows, NLabels]:
            return tf.reduce_sum(test_features, axis=-1, keepdims=True) + self._weights

    a = A()
    a.train(tf.ones((4, 3)), tf.ones((4, 2)))
    a.predict(tf.ones((2, 4)))


@must_fail
def test_instance_shape__bad_return() -> None:
    class A:
        def __init__(self) -> None:
            self._weights: Optional[ttf.Tensor1[NLabels]] = None

        def train(
            self,
            train_features: ttf.Tensor2[NRows, NFeatures],
            train_labels: ttf.Tensor2[NRows, NLabels],
        ) -> None:
            self._weights = tf.reduce_sum(train_labels, axis=0)

        def predict(
            self, test_features: ttf.Tensor2[NRows, NFeatures]
        ) -> ttf.Tensor2[NRows, NLabels]:
            return tf.reduce_sum(test_features, axis=-1, keepdims=True)

    a = A()
    a.train(tf.ones((4, 3)), tf.ones((4, 2)))
    a.predict(tf.ones((2, 3)))


def test_inner_functions() -> None:
    def create_loss(
        target: ttf.Tensor1[NOutputs],
    ) -> Callable[[ttf.Tensor2[NRows, NOutputs]], ttf.Tensor0]:
        def loss(prediction: ttf.Tensor2[NRows, NOutputs]) -> ttf.Tensor0:
            return tf.reduce_sum((target - prediction) ** 2)

        return loss

    loss = create_loss(tf.zeros((3,)))
    loss(tf.zeros((4, 3)))


@must_fail
def test_inner_functions__bad_arg() -> None:
    def create_loss(
        target: ttf.Tensor1[NOutputs],
    ) -> Callable[[ttf.Tensor2[NRows, NOutputs]], ttf.Tensor0]:
        def loss(prediction: ttf.Tensor2[NRows, NOutputs]) -> ttf.Tensor0:
            return tf.reduce_sum((target - prediction) ** 2)

        return loss

    loss = create_loss(tf.zeros((3,)))
    loss(tf.zeros((4, 2)))


def test_intermediate_results() -> None:
    def f(x: ttf.Tensor2[NRows, NCols]) -> ttf.Tensor0:
        a: ttf.Tensor2[NRows, NCols] = x
        b: ttf.Tensor1[NRows] = tf.reduce_sum(a, axis=-1)
        c: ttf.Tensor0 = tf.reduce_sum(b, axis=-1)
        return c

    f(tf.zeros((2, 3)))


@must_fail
def test_intermediate_results__bad() -> None:
    def f(x: ttf.Tensor2[NRows, NCols]) -> ttf.Tensor0:
        a: ttf.Tensor2[NRows, NCols] = x
        b: ttf.Tensor1[NCols] = tf.reduce_sum(a, axis=-1)
        c: ttf.Tensor0 = tf.reduce_sum(b, axis=-1)
        return c

    f(tf.zeros((2, 3)))


def test_dtypes() -> None:
    def f(x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, NColsIn]:
        return x

    f(tf.zeros((2, 3), dtype=tf.float32))
    f(tf.zeros((2, 3), dtype=tf.float64))


@must_fail
def test_dtypes__bad_type() -> None:
    def f(x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, NColsIn]:
        return x

    f(tf.zeros((2, 3), dtype=tf.int32))


@must_fail
def test_dtypes__bad_return() -> None:
    def f(x: ttf.Tensor2[NRows, NColsIn]) -> ttf.Tensor2[NRows, NColsIn]:
        return tf.cast(tf.float64, x)

    f(tf.zeros((2, 3), dtype=tf.float32))
