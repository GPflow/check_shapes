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
# pylint: disable=ungrouped-imports
# pylint: disable=protected-access  # To access the _TensorCoercible

from typing import Any

import pytest

from check_shapes import Shape

from ..utils import assert_has_shape

try:
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp

    requires_tfp = lambda f: f
except ImportError:
    from unittest.mock import MagicMock

    np = MagicMock()
    tf = MagicMock()
    tfp = MagicMock()

    requires_tfp = pytest.mark.skip("TensorFlow-Probability not installed.")


def make_tensor_coercible(
    shape: Shape, concrete: bool
) -> tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible:
    loc = tf.zeros(shape)
    scale = tf.ones(shape)
    dist = tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible(
        tfp.distributions.Normal(loc, scale), lambda self: loc
    )
    if concrete:
        tf.convert_to_tensor(dist)  # Triggers some caching within `dist`.
    return dist


@requires_tfp
@pytest.mark.parametrize(
    "shaped,expected_shape",
    [
        (tfp.util.TransformedVariable(3.0, tfp.bijectors.Exp()), ()),
        (tfp.util.TransformedVariable(np.zeros((4, 2)), tfp.bijectors.Exp()), (4, 2)),
        (make_tensor_coercible((), True), ()),
        (make_tensor_coercible((4, 5), True), (4, 5)),
        (make_tensor_coercible((), False), None),
        (make_tensor_coercible((4, 5), False), None),
    ],
)
def test_get_shape(shaped: Any, expected_shape: Shape) -> None:
    assert_has_shape(shaped, expected_shape)
