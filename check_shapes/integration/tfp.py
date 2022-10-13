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

# pylint: disable=import-outside-toplevel

from typing import Any

from ..base_types import Shape
from ..error_contexts import ErrorContext
from ..shapes import register_get_shape


def install_tfp_integration() -> bool:
    """
    Install various hooks to support TensorFlow-Probability.

    If TensorFlow-Probability is not installed in this environment this function does nothing.

    :return: Whether TensorFlow-Probability support hooks actually were installed.
    """
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp
    except ImportError:
        return False  # TensorFlow-Probability not installed - don't install integrations.

    @register_get_shape(tfp.util.DeferredTensor)
    def get_tensorflow_shape(shaped: Any, context: ErrorContext) -> Shape:
        shape = shaped.shape
        if not shape:
            return None
        return tuple(shape)

    # pylint: disable=protected-access  # To access the _TensorCoercible

    @register_get_shape(tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible)
    def get_tensor_coercible_shape(shaped: Any, context: ErrorContext) -> Shape:
        # This one is unpleasant. Sometimes the `shape` is a `TensorShape`, but sometimes it's a
        # function that returns a `TensorShape`. The version of TensorFlow probability seems to have
        # something to do with it, but it also seems to be more complicated...
        shape = shaped.shape
        if not shape:
            return None
        if not isinstance(shape, tf.TensorShape):
            shape = shape()
            if not shape:
                return None
        return tuple(shape)

    return True
