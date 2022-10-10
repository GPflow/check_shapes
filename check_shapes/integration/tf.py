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

import inspect
from typing import Any, Callable

from ..base_types import Shape
from ..config import add_is_compiled_mode
from ..decorator import WrapperPostProcessor, add_wrapper_post_processor
from ..error_contexts import ErrorContext
from ..shapes import register_get_shape


def install_tf_integration() -> None:
    """
    Install various hooks to support TensorFlow.

    If TensorFlow is not installed in this environment this function does nothing.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return  # TensorFlow not installed - don't install integrations.

    ########################################
    # Support for getting shapes of tf types
    ########################################

    @register_get_shape(tf.Tensor)
    @register_get_shape(tf.Variable)
    def get_tensorflow_shape(shaped: Any, context: ErrorContext) -> Shape:
        shape = shaped.shape
        if not shape:
            return None
        return tuple(shape)

    ################################################
    # Support for ShapeCheckingState.EAGER_MODE_ONLY
    ################################################

    add_is_compiled_mode(tf.inside_function)

    ########################################################
    # Work-around TensorFlow's many problems with decorators
    ########################################################

    class TfWrapperPostProcessor(WrapperPostProcessor):
        def on_wrap(
            self,
            func: Callable[..., Any],
            wrapped: Callable[..., Any],
            signature: inspect.Signature,
            tf_decorator: bool = False,
        ) -> Callable[..., Any]:
            # Work-around for TensorFlow saved_model expecting methods to have a `self` argument:
            if "self" in signature.parameters:

                wrapped_function = wrapped

                def wrapped_method(self: Any, *args: Any, **kwargs: Any) -> Any:
                    return wrapped_function(self, *args, **kwargs)

                wrapped = wrapped_method

            # Make TensorFlow understand our decoration:
            if tf_decorator:
                tf.compat.v1.flags.tf_decorator.make_decorator(func, wrapped)

            return wrapped

    add_wrapper_post_processor(TfWrapperPostProcessor())
