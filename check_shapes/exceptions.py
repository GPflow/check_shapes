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

# pylint: disable=broad-except

"""
Exceptions generated by `check_shapes`.

These rely heavily on the infrastructure in `error_contexts.py`.
"""
from .error_contexts import ErrorContext, MessageBuilder


class CheckShapesError(Exception):
    """
    Common super class for `check_shapes` errors.
    """

    error_message: str

    def __init__(self, context: ErrorContext) -> None:
        builder = MessageBuilder()
        builder.add_line("")
        builder.add_line(self.error_message)
        with builder.indent() as b:
            context.print(b)
        super().__init__(builder.build())

        self.context = context

        # Prevent Keras from rewriting our exception:
        self._keras_call_info_injected = True


class VariableTypeError(CheckShapesError):
    """
    Error raised if a variable is used both as a rank-1 and a variable-rank variable.
    """

    error_message = (
        "Cannot use the same variable to bind both a single dimension"
        " and a variable number of dimensions."
    )


class SpecificationParseError(CheckShapesError):
    """
    Error raised if there was an error parsing the shape specification.
    """

    error_message = "Unable to parse shape specification."


class DocstringParseError(CheckShapesError):
    """
    Error raised if there was an error parsing the shape specification.
    """

    error_message = "Unable to parse docstring."


class ArgumentReferenceError(CheckShapesError):
    """
    Error raised if the argument to check the shape of could not be resolved.
    """

    error_message = "Unable to resolve argument / missing argument."


class ShapeMismatchError(CheckShapesError):
    """
    Error raised if a function is called with tensors of the wrong shape.
    """

    error_message = "Tensor shape mismatch."


class NoShapeError(CheckShapesError):
    """
    Error raised if we are trying to get the shape of an object that does not have a shape.
    """

    error_message = "Unable to determine shape of object."
