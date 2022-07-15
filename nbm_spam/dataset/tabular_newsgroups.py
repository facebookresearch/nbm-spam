# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import torch
from nbm_spam.dataset.tabular_constants import RANDOM_STATE, TRAIN_SIZE, VAL_SIZE
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Newsgroups(Dataset):
    def __init__(
        self,
        split: str,
        categories: List[str] = None,
        scale_range: Optional[Tuple[float, float]] = None,
        device: Any = None,
    ):

        self.device = device
        _data = fetch_20newsgroups(
            subset="train", categories=categories, random_state=RANDOM_STATE
        )
        # only train / test split provided
        # split train into train and val
        _f_train, _f_val, _l_train, _l_val = train_test_split(
            _data.data,
            _data.target,
            test_size=VAL_SIZE / (VAL_SIZE + TRAIN_SIZE),
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=_data.target,
        )
        self._vectorizer = TfidfVectorizer(lowercase=False)
        self._vectorizer.fit(_f_train)
        _f_train = self._vectorizer.transform(_f_train).todense()
        if scale_range:
            self._scaler = MinMaxScaler(scale_range)
            self._scaler.fit(_f_train)
            _f_train = self._scaler.transform(_f_train)

        self.split = split
        self.num_concepts = _f_train.shape[-1]
        self.num_classes = max(_data.target) + 1
        self.concept_names = [
            item[0]
            for item in sorted(
                self._vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=False
            )
        ]
        self.class_names = _data.target_names

        if self.split == "train":
            self.features = torch.Tensor(_f_train).to(device=self.device)
            self.labels = torch.Tensor(_l_train).to(
                device=self.device, dtype=torch.int64
            )
        elif self.split == "validation":
            _f_val = self._vectorizer.transform(_f_val).todense()
            if scale_range:
                _f_val = self._scaler.transform(_f_val)
            self.features = torch.Tensor(_f_val).to(device=self.device)
            self.labels = torch.Tensor(_l_val).to(device=self.device, dtype=torch.int64)
        elif self.split == "test":
            _data = fetch_20newsgroups(
                subset="test", categories=categories, random_state=RANDOM_STATE
            )
            _f_test = self._vectorizer.transform(_data.data).todense()
            if scale_range:
                _f_test = self._scaler.transform(_f_test)
            self.features = torch.Tensor(_f_test).to(device=self.device)
            self.labels = torch.Tensor(_data.target).to(
                device=self.device, dtype=torch.int64
            )

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
