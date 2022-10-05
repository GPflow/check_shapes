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

from ..base_types import Shape
from ..error_contexts import ErrorContext
from ..shapes import register_get_shape


def install_torch_integration() -> bool:
    """
    Install various hooks to support PyTorch.

    If PyTorch is not installed in this environment this function does nothing.

    :return: Whether PyTorch support hooks actually were installed.
    """
    try:
        import torch
    except ImportError:
        return False  # PyTorch not installed - don't install integrations.

    @register_get_shape(torch.Tensor)
    def get_torch_shape(shaped: torch.Tensor, context: ErrorContext) -> Shape:
        return tuple(shaped.shape)

    return True
