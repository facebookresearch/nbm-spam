# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .build import MODELS_REGISTRY


@MODELS_REGISTRY.register()
class ConceptLinear(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super().__init__()
        self._model = nn.Linear(num_concepts, num_classes, bias=True)

    def forward(self, batch):
        return self._model(batch)
