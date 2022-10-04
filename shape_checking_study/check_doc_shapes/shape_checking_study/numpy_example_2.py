"""============
check_shapes
============

A library for annotating and checking the shapes of tensors.

This library is compatible with both TensorFlow and NumPy.

The main entry point is :func:`shape_checking_study.numpy_example_2.check_shapes`.

For example::

    @tf.function
    @check_shapes(
        "features: [batch_shape..., n_features]",
        "weights: [n_features]",
        "return: [batch_shape...]",
    )
    def linear_model(
        features: tf.Tensor, weights: tf.Tensor
    ) -> tf.Tensor:
        ...


Check specification
+++++++++++++++++++

The shapes to check are specified by the arguments to :func:`check_shapes`. Each argument is a
string of the format::

    <argument specifier>: <shape specifier>


Argument specification
----------------------

The ``<argument specifier>`` must start with either the name of an argument to the decorated
function, or the special name ``return``. The value ``return`` refers to the value returned by the
function.

The ``<argument specifier>`` can then be modified to refer to elements of the object in two ways:

    * Use ``.<name>`` to refer to attributes of the object.
    * Use ``[<index>]`` to refer to elements of a sequence. This is particularly useful if your
      function returns a tuple of values.

We do not support looking up values in a  ``dict``.

For example::

    @check_shapes(
        "weights: ...",
        "data.training_data: ...",
        "return: ...",
        "return[0]: ...",
        "something[0].foo.bar[23]: ...",
    )
    def f(...):
        ...


Shape specification
-------------------

Shapes are specified by the syntax
``[<dimension specifier 1>, <dimension specifer 2>, ..., <dimension specifier n>]``, where
``<dimension specifier i>`` is
one of:

    * ``<integer>``, to require that dimension to have that exact size.
    * ``<name>``, to bind that dimension to a variable. Dimensions bound to the same variable must
      have the same size, though that size can be anything.
    * ``*<name>`` or ``<name>...``, to bind *any* number of leading dimensions to a variable. Again,
      multiple uses of the same variable name must match the same dimension sizes. Notice this only
      is valid for leading dimensions.

A scalar shape is specified by ``[]``.

For example::

    @check_shapes(
        "...: []",
        "...: [3, 4]",
        "...: [width, height]",
        "...: [*batch, n_features]",
        "...: [batch..., 2]",
    )
    def f(...):
        ...


Class inheritance
+++++++++++++++++

If you have a class hiererchy, you probably want to ensure that derived classes handle tensors with
the same shapes as the base classes. You can use the :func:`inherit_check_shapes` decorator to
inherit shapes from overridden methods.

Example::

    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            ("a", ["batch...", 4]),
            ("return", ["batch...", 1]),
        )
        def f(self, a: tf.Tensor) -> tf.Tensor:
            ...

    class SubClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: tf.Tensor) -> tf.Tensor:
            ...


Speed, and interactions with `tf.function`
++++++++++++++++++++++++++++++++++++++++++

If you want to wrap your function in both :func:`tf.function` and :func:`check_shapes` it is
recommended you put the :func:`tf.function` outermost so that the shape checks are inside
:func:`tf.function`.  Shape checks are performed while tracing graphs, but *not* compiled into the
actual graphs.  This is considered a feature as that means that :func:`check_shapes` doesn't impact
the execution speed of compiled functions. However, it also means that tensor dimensions of dynamic
size are not verified in compiled mode.

"""
from dataclasses import dataclass
from typing import Tuple, Callable
from functools import wraps

import numpy as np
from gpflow.experimental.check_shapes.base_types import C


def check_shapes(*spec_strs: str) -> Callable[[C], C]:
    """
    Decorator that checks the shapes of tensor arguments.

    See: `check_shapes`_.

    :param spec_strs: Specification of arguments to check. See: `Argument specification`_.

    """
    def wrap(func: C) -> C:
        return func
    return wrap


def inherit_check_shapes(func: C) -> C:
    """
    Decorator that inherits the :func:`check_shapes` decoration from any overridden method in a
    super-class.

    See: `Class inheritance`_.
    """
    return C


@dataclass
class Data:
    features: np.ndarray
    labels: np.ndarray


@check_shapes(
    "train_data.features: [*batch, n_train_rows, n_features]",
    "train_data.labels: [1, 3, n_train_rows, n_labels]",
    "test_features: [n_test_rows, n_features]",
    "return[0]: []",
    "return[1]: [n_test_rows, n_labels,]",
)
def train_and_predict(
    train_data: Data, test_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a model on `train_data`, then make predictions on `test_features`.

    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus quis venenatis neque. Proin
    eros nibh, malesuada sed vestibulum quis, bibendum sit amet dolor. Donec sit amet ultrices nisl,
    sed condimentum eros. Mauris venenatis libero at ex ultricies, sed laoreet sapien mollis. Nulla
    quis scelerisque lorem, id rhoncus massa. Aenean lobortis justo nisi, ac tempor metus tempor
    a. Aliquam venenatis ex in leo malesuada molestie. Maecenas sit amet leo et turpis interdum
    sollicitudin.

    Ut efficitur in risus vel sollicitudin. Interdum et malesuada fames ac ante ipsum primis in
    faucibus. Donec ac sollicitudin enim. Maecenas vitae consectetur felis, a facilisis nibh. Nullam
    sed molestie magna, et porttitor nunc. Donec scelerisque bibendum aliquam. Morbi ut justo
    dapibus, scelerisque ante non, laoreet sem. Interdum et malesuada fames ac ante ipsum primis in
    faucibus. Pellentesque aliquet massa eleifend arcu dictum, vel feugiat justo mollis. Morbi
    placerat leo urna, auctor egestas est dignissim non. Morbi felis ex, eleifend vitae efficitur
    at, vestibulum tempus quam. Praesent id lorem aliquam, semper dolor vel, placerat ex. Aenean
    porttitor et nisl non sagittis. Praesent a est erat. Morbi consectetur ipsum eros, et pharetra
    tellus fringilla facilisis. Cras vehicula urna id elit pharetra scelerisque.

    Quisque posuere justo sem, sed condimentum lacus molestie vitae. Nullam blandit dui nibh, eget
    elementum mauris varius ut. Sed non ligula a dolor pellentesque condimentum. Suspendisse
    consequat arcu est, et feugiat nisi fringilla sed. Cras lobortis molestie nulla eu
    egestas. Curabitur ac mauris iaculis, laoreet augue id, vehicula dolor. Fusce euismod placerat
    dui, eget porta ex.

    :param train_data:
        Data to train on.

        Maecenas ullamcorper nisi quis sem varius, et mollis nisl volutpat. In quis neque sit amet
        metus bibendum mollis. Aenean tempor in purus ac finibus. Sed ultrices tortor mauris, sit
        amet porta tortor dapibus quis. In gravida, ligula sed gravida auctor, nulla nunc fringilla
        orci, sit amet placerat leo lectus sed eros. Duis volutpat tincidunt efficitur. Donec mollis
        sed tortor eget tincidunt.

        Donec tristique auctor augue vel rhoncus. Vivamus et mattis dui. Praesent ultricies ut
        sapien a egestas. Morbi luctus sed ante eget vulputate. Nam ultrices feugiat urna, non
        facilisis tortor. Praesent sodales eu ante egestas accumsan. Mauris sed quam eu mi suscipit
        tincidunt.

    :param foo bar test_features: Features to make prediction from.
    :returns:
        Prediction mean, and variance.

    """
    mean, var = ..., ...
    return mean, var


if __name__ == "__main__":
    print(train_and_predict.__doc__)
