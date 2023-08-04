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
"""
A library for annotating and checking the shapes of tensors.
"""
from .accessors import get_check_shapes
from .base_types import Dimension, Shape
from .checker import ShapeChecker
from .checker_context import check_shape, get_shape_checker
from .config import (
    DocstringFormat,
    ShapeCheckingState,
    disable_check_shapes,
    get_drop_frames,
    get_enable_check_shapes,
    get_enable_function_call_precompute,
    get_rewrite_docstrings,
    set_drop_frames,
    set_enable_check_shapes,
    set_enable_function_call_precompute,
    set_rewrite_docstrings,
)
from .decorator import check_shapes
from .error_contexts import ErrorContext
from .inheritance import inherit_check_shapes
from .integration.tf import install_tf_integration
from .integration.tfp import install_tfp_integration
from .integration.torch import install_torch_integration
from .shapes import get_shape, register_get_shape

install_tf_integration()
install_tfp_integration()
install_torch_integration()

__version__ = "1.1.1"

__all__ = [
    "Dimension",
    "DocstringFormat",
    "ErrorContext",
    "Shape",
    "ShapeChecker",
    "ShapeCheckingState",
    "accessors",
    "argument_ref",
    "base_types",
    "bool_specs",
    "check_shape",
    "check_shapes",
    "checker",
    "checker_context",
    "config",
    "decorator",
    "disable_check_shapes",
    "error_contexts",
    "exceptions",
    "get_check_shapes",
    "get_drop_frames",
    "get_enable_check_shapes",
    "get_enable_function_call_precompute",
    "get_rewrite_docstrings",
    "get_shape",
    "get_shape_checker",
    "inherit_check_shapes",
    "inheritance",
    "parser",
    "register_get_shape",
    "set_drop_frames",
    "set_enable_check_shapes",
    "set_enable_function_call_precompute",
    "set_rewrite_docstrings",
    "shapes",
    "specs",
]
