# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import heapq
import math
from itertools import combinations
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .build import MODELS_REGISTRY


@MODELS_REGISTRY.register()
class ConceptSPAM(nn.Module):
    """
    ref:
        Scalable Interpretability via Polynomials.
        Abhimanyu Dubey, Filip Radenovic, Dhruv Mahajan.
        https://arxiv.org/pdf/2205.14108.pdf
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        ranks: List[int] = None,
        dropout: float = 0,
        ignore_unary: bool = False,
        reg_order: int = 2,
        lower_order_correction: bool = False,
        use_geometric_mean: bool = True,
        orthogonal: bool = False,
        proximal: bool = False,
        **_,
    ) -> None:
        super(ConceptSPAM, self).__init__()
        self._num_concepts = num_concepts
        self._num_features = self._num_concepts * int(not ignore_unary)
        self._num_classes = num_classes
        self._ranks = ranks
        self._degree = len(ranks) + 1
        self._dropout = dropout
        self._ignore_unary = ignore_unary
        self._reg_ord = reg_order
        self._lower_order_correction = lower_order_correction
        self._orthogonal = orthogonal
        self._use_geometric_mean = use_geometric_mean
        self._use_proximal = proximal

        self._final_repr = None

        print(f"Learning polynomial of degree {self._degree}")

        self.poly_weights = nn.ModuleList()
        self._degrees = []

        for i, _rank in enumerate(self._ranks):
            if _rank > 0:
                _deg_x = nn.Linear(self._num_concepts, _rank, bias=False)
                if self._orthogonal:
                    _deg_x = nn.utils.parametrizations.orthogonal(_deg_x)
                self.poly_weights.append(_deg_x)
                self._num_features += _rank
                self._degrees.append(i + 2)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.classifier = nn.Linear(self._num_features, self._num_classes)

    def _proximal_step(self):
        with torch.no_grad():
            self.classifier.weight[self.classifier.weight < 0] = 0
            for _poly in self.poly_weights:
                _poly.weight[_poly.weight < 0] = 0

    def forward(self, input):
        if self._use_proximal:
            self._proximal_step()

        x = None
        if not self._ignore_unary:
            x = input
        for i, _polynomial in enumerate(self.poly_weights):
            _input = input
            if self._use_geometric_mean:
                _input = torch.sign(input) * torch.pow(
                    torch.abs(input), 1 / self._degrees[i]
                )
            _xp = torch.pow(_polynomial(_input), self._degrees[i])
            if self._lower_order_correction:
                _correction = self._compute_correction(
                    _input, _polynomial.weight, self._degrees[i]
                )
                _xp = _xp - _correction
            if x is None:
                x = _xp
            else:
                x = torch.cat([x, _xp], dim=-1)
        x = self.dropout(x)
        return self.classifier(x)

    def tensor_regularization(self):
        loss = torch.sum(
            torch.linalg.norm(self.classifier.weight, dim=0, ord=self._reg_ord)
        ) + torch.sum(torch.linalg.norm(self.classifier.bias, ord=self._reg_ord))
        for _deg, _poly in zip(self._degrees, self.poly_weights):
            loss += torch.sum(
                torch.pow(
                    torch.linalg.norm(_poly.weight, ord=self._reg_ord, dim=0), _deg
                )
            )
        return loss

    def basis_l1_regularization(self):
        return torch.stack(
            [torch.abs(_poly.weight).sum() for _poly in self.poly_weights]
        ).sum()

    def _compute_correction(self, input: Tensor, weight: Tensor, degree: int) -> Tensor:
        """remove lower order interactions to prevent redundancy"""
        correction = None
        for i in range(2, degree + 1):
            dci = math.comb(degree, i) * (2 * (1 - (i % 2)) - 1)
            p1, p2 = i, degree - i
            _inp1, _inp2 = 1.0, 1.0
            if p1 > 0:
                _inp1 = F.linear(input**p1, weight**p1)
            if p2 > 0:
                _inp2 = F.linear(input**p2, weight**p2)
            _correction_i = _inp1 * _inp2
            if correction is None:
                correction = dci * _correction_i
            else:
                correction += dci * _correction_i
        return correction

    def get_importance(
        self,
        input: Tensor,
        target: Tensor = None,
        top_k: int = 1,
        **_,
    ) -> Tensor:
        """Returns importances for any input and target class"""
        if len(self._ranks) == 1:
            return self._get_importance_quadratic(input, target, top_k)

        _classifier_weights = self.classifier.weight[target, :]
        bias_weight_pos, bias_feature_pos = 0, 0
        top_importances = []

        nonzero_idxes = torch.nonzero(input).tolist()
        nonzero_idxes = [_x[0] for _x in nonzero_idxes]

        # process unary terms from linear
        _unary_importances = []
        if not self._ignore_unary:
            for _idx in nonzero_idxes:
                _unary_importances.append(input[_idx] * _classifier_weights[_idx])
            bias_weight_pos += self._num_concepts
            bias_feature_pos += self._num_concepts

        for _deg, _polynomial in enumerate(self.poly_weights):
            _input = input
            if self._use_geometric_mean:
                _input = torch.pow(input, 1 / self._degrees[_deg])
            # first process unary terms from polynomial
            for _loc_idx, _idx in enumerate(nonzero_idxes):
                _unary_importances[_loc_idx] += sum(
                    _classifier_weights[_rank + bias_weight_pos].item()
                    * (_polynomial.weight[_rank, _idx].item() * _input[_idx].item())
                    ** self._degrees[_deg]
                    for _rank in range(_polynomial.weight.shape[0])
                )

            # now processing higher order terms
            _out_ft = _polynomial.weight.shape[0]
            _raw_activations = _polynomial.weight * _input.repeat(_out_ft, 1)
            _combs = combinations(
                list(range(_raw_activations.shape[1])), self._degrees[_deg]
            )
            _combs_nonzero = [
                (_comb, _idx)
                for _idx, _comb in enumerate(_combs)
                if all(_x in nonzero_idxes for _x in _comb)
            ]
            _poly_importances = [
                (
                    sum(
                        _classifier_weights[_rank + bias_weight_pos].item()
                        * math.prod(
                            [_raw_activations[_rank][_idx].item() for _idx in _tuple]
                        )
                        for _rank in range(_raw_activations.shape[0])
                    ),
                    _tuple_idx + bias_feature_pos,
                    _tuple,
                )
                for _tuple, _tuple_idx in _combs_nonzero
            ]
            _poly_importances = [(abs(_x[0]), *_x) for _x in _poly_importances]
            for _imp in _poly_importances:
                heapq.heappush(top_importances, _imp)

            bias_feature_pos += len(list(_combs))
            bias_weight_pos += _out_ft

        # update unary terms in heap
        for _idx, _imp in zip(nonzero_idxes, _unary_importances):
            heapq.heappush(top_importances, (abs(_imp), _imp, _idx, (_idx)))

        _importances_top_k = heapq.nlargest(top_k, top_importances)
        print(_importances_top_k)
        return [(_x[2], _x[1]) for _x in _importances_top_k]

    def _get_importance_quadratic(
        self,
        input: Tensor,
        target: Tensor = None,
        top_k: int = 1,
        **_,
    ) -> Tensor:
        """Returns importances for any input and target class"""
        _classifier_weights = self.classifier.weight[target, :]
        bias_weight_pos, bias_feature_pos = 0, 0
        top_importances = []

        nonzero_idxes = torch.nonzero(input).tolist()
        nonzero_idxes = [_x[0] for _x in nonzero_idxes]

        # process unary terms from linear
        _unary_importances = []
        if not self._ignore_unary:
            for _idx in nonzero_idxes:
                _unary_importances.append(input[_idx] * _classifier_weights[_idx])
            bias_weight_pos += self._num_concepts
            bias_feature_pos += self._num_concepts

        _polynomial = self.poly_weights[0]

        _input = input
        if self._use_geometric_mean:
            _input = torch.pow(input, 1 / self._degrees[0])

        # first process unary terms from polynomial
        for _loc_idx, _idx in enumerate(nonzero_idxes):
            _unary_importances[_loc_idx] += sum(
                _classifier_weights[_rank + bias_weight_pos].item()
                * (_polynomial.weight[_rank, _idx].item() * _input[_idx].item())
                ** self._degrees[0]
                for _rank in range(_polynomial.weight.shape[0])
            )

        # now processing higher order terms
        _out_ft = _polynomial.weight.shape[0]

        _weight_matrix_unscaled = torch.bmm(
            _polynomial.weight.unsqueeze(-1), _polynomial.weight.unsqueeze(1)
        )
        _scaling_factor = (
            _classifier_weights[bias_weight_pos : bias_weight_pos + _out_ft]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, self._num_concepts, self._num_concepts)
        )
        _weight_matrix = torch.sum(_weight_matrix_unscaled * _scaling_factor, dim=0)
        _ft_matrix = torch.mm(_input.unsqueeze(0).t(), _input.unsqueeze(0))
        _activations = _weight_matrix * _ft_matrix
        _abs_activations = torch.abs(_activations)
        _abs_activations = torch.triu(_abs_activations, diagonal=1)

        _sorted_activations = torch.topk(
            (_abs_activations).view(-1), k=top_k, largest=True
        )
        _sorted_activation_values, _sorted_activation_idxes = (
            _sorted_activations.values.tolist(),
            _sorted_activations.indices.tolist(),
        )
        _sorted_activation_idxes_flat = [
            [k // self._num_concepts, k % self._num_concepts]
            for k in _sorted_activation_idxes
        ]
        assert all(k[0] < k[1] for k in _sorted_activation_idxes_flat), "error"

        _combs = list(
            combinations(list(range(_weight_matrix.shape[0])), self._degrees[0])
        )
        _sorted_activation_idxes = [
            _combs.index((_x[0], _x[1])) + bias_feature_pos
            for _x in _sorted_activation_idxes_flat
        ]

        for _idx, _val, _tuple in zip(
            _sorted_activation_idxes,
            _sorted_activation_values,
            _sorted_activation_idxes_flat,
        ):
            idx_x, idx_y = _tuple
            _raw_val = _activations[idx_x, idx_y]
            heapq.heappush(top_importances, (_val, _raw_val.item(), _idx, _tuple))

        # update unary terms in heap
        for _idx, _imp in zip(nonzero_idxes, _unary_importances):
            heapq.heappush(
                top_importances, (abs(_imp.item()), _imp.item(), _idx, (_idx))
            )

        _importances_top_k = heapq.nlargest(top_k, top_importances)
        return [(_x[2], _x[1]) for _x in _importances_top_k]
