# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple

import torch
from nbm_spam.dataset.tabular_constants import (
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CaliforniaHousing(Dataset):
    def __init__(
        self,
        split: str,
        scale_range: Optional[Tuple[float, float]] = None,
        device: Any = None,
    ):

        self.device = device
        _data = fetch_california_housing()
        if scale_range:
            _scaler = MinMaxScaler(scale_range)
            _scaler.fit(_data.data)
            _scaled_data = _scaler.transform(_data.data)
        else:
            _scaled_data = _data.data.copy()

        self.split = split
        self.num_concepts = _scaled_data.shape[-1]
        self.num_classes = 1  # regression
        self.concept_names = _data.feature_names
        self.class_names = _data.target_names

        # split 'all' to 'train+val' vs 'test'
        _f_train, _f_test, _l_train, _l_test = train_test_split(
            _scaled_data,
            _data.target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=None,
        )
        # split 'train+val' to 'train' vs 'val'
        _f_train, _f_val, _l_train, _l_val = train_test_split(
            _f_train,
            _l_train,
            test_size=VAL_SIZE / (VAL_SIZE + TRAIN_SIZE),
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=None,
        )

        if self.split == "train":
            self.features = torch.Tensor(_f_train).to(device=self.device)
            self.labels = torch.Tensor(_l_train).to(device=self.device)
        elif self.split == "validation":
            self.features = torch.Tensor(_f_val).to(device=self.device)
            self.labels = torch.Tensor(_l_val).to(device=self.device)
        elif self.split == "test":
            self.features = torch.Tensor(_f_test).to(device=self.device)
            self.labels = torch.Tensor(_l_test).to(device=self.device)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
