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
Definitions of commonly used types.
"""
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, TypeVar, Union

import numpy as np

from .type_flags import GENERIC_NP_ARRAYS, NP_TYPE_CHECKING

C = TypeVar("C", bound=Callable[..., Any])

Dimension = Optional[int]
"""
The size of a single observed dimension.

Use `None` if the size of that dimension is unknown.
"""

Shape = Optional[Tuple[Dimension, ...]]
"""
The complete shape of an observed object.

Use `None` if the object has a shape, but the shape is unknown.

Raise an exception if objects of that type can never have a shape.
"""

if TYPE_CHECKING and (not NP_TYPE_CHECKING):  # pragma: no cover
    AnyNDArray = Any
else:
    if GENERIC_NP_ARRAYS:
        # It would be nice to use something more interesting than `Any` here, but it looks like
        # the infrastructure in the rest of the ecosystem isn't really set up for this
        # yet. Maybe when we get Python 3.11?
        AnyNDArray = np.ndarray[Any, Any]  # type: ignore[misc]
    else:
        AnyNDArray = Union[np.ndarray]  # type: ignore[misc]
