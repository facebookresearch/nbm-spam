# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .build import MODELS_REGISTRY


@MODELS_REGISTRY.register()
class ConceptMLP(nn.Module):
    def __init__(
        self,
        num_concepts,
        num_classes,
        mlp_dims,
        p_dropout,
        batchnorm=True,
    ):
        super().__init__()
        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._batchnorm = batchnorm
        self._mlp_dims = mlp_dims
        self._model = None
        self._layers = []
        self._relu = nn.ReLU()

        layers = []
        prev_dim = num_concepts

        for _, dim in enumerate(mlp_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if self._batchnorm is True:
                layers.append(nn.BatchNorm1d(dim))
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))

            layers.append(nn.ReLU())
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self._model = nn.Sequential(*layers)

    def forward(self, batch):
        return self._model(batch)
