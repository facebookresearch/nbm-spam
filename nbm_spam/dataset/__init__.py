# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .build import build_dataset, DATASET_REGISTRY
from .tabular_california_housing import CaliforniaHousing
from .tabular_covtype import CovType
from .tabular_newsgroups import Newsgroups

__all__ = [
    "build_dataset",
    "DATASET_REGISTRY",
    "CaliforniaHousing",
    "CovType",
    "Newsgroups",
]
