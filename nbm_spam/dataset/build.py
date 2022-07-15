# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from nbm_spam.utils.registry import Registry
from torch.utils.data import Dataset


DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets.

Registered object must return instance of :class: `torch.utils.data.Dataset`.
"""


def build_dataset(dataset_name, *args, **kwargs):
    dataset = DATASET_REGISTRY.get(dataset_name)(*args, **kwargs)
    assert isinstance(dataset, Dataset)
    return dataset
