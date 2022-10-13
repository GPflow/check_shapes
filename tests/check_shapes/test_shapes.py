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
from typing import Any

import pytest

from check_shapes import Shape, get_shape
from check_shapes.exceptions import NoShapeError

from .utils import TestContext, assert_has_shape


@pytest.mark.parametrize(
    "shaped,expected_shape",
    [
        (True, ()),
        (0, ()),
        (0.0, ()),
        ("foo", ()),
        ((), None),
        ((0,), (1,)),
        ([[0.1, 0.2]], (1, 2)),
        ([[[], []]], None),
    ],
)
def test_get_shape(shaped: Any, expected_shape: Shape) -> None:
    assert_has_shape(shaped, expected_shape)


@pytest.mark.parametrize(
    "shaped,expected_message",
    [
        (
            object(),
            """
Unable to determine shape of object.
  Fake test error context.
    Object type: builtins.object
""",
        ),
        (
            [[object()]],
            """
Unable to determine shape of object.
  Fake test error context.
    Index: [0]
      Index: [0]
        Object type: builtins.object
""",
        ),
    ],
)
def test_get_shape__error(shaped: Any, expected_message: str) -> None:
    with pytest.raises(NoShapeError) as e:
        get_shape(shaped, TestContext())

    (message,) = e.value.args
    assert expected_message == message
