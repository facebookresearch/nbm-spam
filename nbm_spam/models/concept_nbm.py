import math
from collections import OrderedDict
from itertools import combinations

import torch
import torch.nn as nn
from omegaconf.listconfig import ListConfig
from torch.nn.modules.activation import ReLU

from .build import MODELS_REGISTRY
from .concept_spam import ConceptSPAM


class ConceptNNBasesNary(nn.Module):
    """Neural Network learning bases."""

    def __init__(
        self, order, num_bases, hidden_dims, dropout=0.0, batchnorm=False
    ) -> None:
        """Initializes ConceptNNBases hyperparameters.
        Args:
            order: Order of N-ary concept interatctions.
            num_bases: Number of bases learned.
            hidden_dims: Number of units in hidden layers.
            dropout: Coefficient for dropout regularization.
            batchnorm (True): Whether to use batchnorm or not.
        """
        super(ConceptNNBasesNary, self).__init__()

        assert order > 0, "Order of N-ary interactions has to be larger than '0'."

        layers = []
        self._model_depth = len(hidden_dims) + 1
        self._batchnorm = batchnorm

        # First input_dim depends on the N-ary order
        input_dim = order
        for dim in hidden_dims:
            layers.append(nn.Linear(in_features=input_dim, out_features=dim))
            if self._batchnorm is True:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(ReLU())
            input_dim = dim

        # Last MLP layer
        layers.append(nn.Linear(in_features=input_dim, out_features=num_bases))
        # Add batchnorm and relu for bases
        if self._batchnorm is True:
            layers.append(nn.BatchNorm1d(num_bases))
        layers.append(ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@MODELS_REGISTRY.register()
class ConceptNBMNary(nn.Module):
    """
    Neural network (MLP) learns set of bases functions that are global,
    which are then used on each concept feature tuple individually.

    NBM model where higher order interactions of features are modeled in bases
    as f(xi, xj) for order 2 or f(xi, xj, xk) for arbitrary order d.

    ref:
        Neural Bases Model.
    """

    def __init__(
        self,
        num_concepts,
        num_classes,
        nary=None,
        num_bases=100,
        hidden_dims=(256, 128, 128),
        num_subnets=1,
        dropout=0.0,
        bases_dropout=0.0,
        batchnorm=True,
        output_penalty=0.0,
        polynomial=None,
    ):
        """Initializing NBM hyperparameters.

        Args:
            num_concepts: Number of concepts used as input to the model.
            num_classes: Number of output classes of the model.
            nary (None):
                None:: unary model with all features is initialized.
                List[int]:: list of n-ary orders to be initialized, eg,
                    [1] or [1, 2, 4] or [2, 3].
                Dict[str, List[Tuple]]:: for each order (key) a list of
                    index tuples (value) is given. Only those preselected indices
                    are used in the model. Eg,
                    {"1": [(0, ), (1, ), (2, )], "2": [(0, 1), (1, 2)]}
            num_bases (100): Number of bases learned.
            hidden_dims ([256, 128, 128]): Number of hidden units for neural
                MLP bases part of model.
            num_subnets (1): Number of neural networks used to learn bases.
            dropout (0.0): Coefficient for dropout within neural MLP bases.
            bases_dropout (0.0): Coefficient for dropping out entire basis.
            batchnorm (True): Whether to use batchnorm or not.
            polynomial (None): Supply SPAM initialization here to train NBM-SPAM.
                Note: if polynomial is not None, nary has to be of order 1.
        """
        super(ConceptNBMNary, self).__init__()

        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._num_bases = num_bases
        self._num_subnets = num_subnets
        if polynomial:
            self._num_subnets_per_polynomial = num_subnets
            self._num_subnets = (len(polynomial["ranks"]) + 1) * num_subnets
        self._batchnorm = batchnorm
        self._output_penalty = output_penalty

        if nary is None:
            # if no nary specified, unary model is initialized
            self._nary_indices = {"1": list(combinations(range(self._num_concepts), 1))}
        elif isinstance(nary, list) or isinstance(nary, ListConfig):
            self._nary_indices = {
                str(order): list(combinations(range(self._num_concepts), order))
                for order in nary
            }
        elif isinstance(nary, dict):
            self._nary_indices = nary
        else:
            raise TypeError("'nary': None or list or dict supported")

        self.bases_nary_models = nn.ModuleDict()
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                self.bases_nary_models[
                    self.get_key(order, subnet)
                ] = ConceptNNBasesNary(
                    order=int(order),
                    num_bases=self._num_bases,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    batchnorm=batchnorm,
                )

        self.bases_dropout = nn.Dropout(p=bases_dropout)

        num_out_features = (
            sum(len(self._nary_indices[order]) for order in self._nary_indices.keys())
            * self._num_subnets
        )
        self.featurizer = nn.Conv1d(
            in_channels=num_out_features * self._num_bases,
            out_channels=num_out_features,
            kernel_size=1,
            groups=num_out_features,
        )

        if polynomial:
            if list(self._nary_indices.keys()) != ["1"]:
                raise ValueError(
                    "'nary': if polynomial is not None, nary has to be of order '1'"
                )
            self._use_spam = True
            _poly_ranks = polynomial["ranks"]
            _other_keys = {k: v for k, v in polynomial.items() if k != "ranks"}
            self._spam = nn.ModuleList()
            self._spam_num_out_features = (
                sum(
                    len(self._nary_indices[order])
                    for order in self._nary_indices.keys()
                )
                * self._num_subnets_per_polynomial
            )
            # first order model in spam is linear
            self._spam.append(
                nn.Linear(
                    in_features=self._spam_num_out_features,
                    out_features=self._num_classes,
                    bias=True,
                )
            )
            # second and higher are polynomials
            for _id, _rank in enumerate(_poly_ranks):
                _temp_rank = [0] * _id + [_rank]
                self._spam.append(
                    ConceptSPAM(
                        num_concepts=self._spam_num_out_features,
                        num_classes=self._num_classes,
                        ignore_unary=True,
                        use_geometric_mean=False,
                        ranks=_temp_rank,
                        **_other_keys,
                    )
                )
        else:
            self._use_spam = False
            self.classifier = nn.Linear(
                in_features=num_out_features,
                out_features=self._num_classes,
                bias=True,
            )

    def get_key(self, order, subnet):
        return f"ord{order}_net{subnet}"

    def forward(self, input):
        bases = []
        for order in self._nary_indices.keys():
            for subnet in range(self._num_subnets):
                input_order = input[:, self._nary_indices[order]]

                bases.append(
                    self.bases_dropout(
                        self.bases_nary_models[self.get_key(order, subnet)](
                            input_order.reshape(-1, input_order.shape[-1])
                        ).reshape(input_order.shape[0], input_order.shape[1], -1)
                    )
                )

        bases = torch.cat(bases, dim=-2)

        out_feats = self.featurizer(bases.reshape(input_order.shape[0], -1, 1)).squeeze(
            -1
        )

        if self._use_spam:
            out = []
            for _poly_idx in range(len(self._spam)):
                _start_idx = _poly_idx * self._spam_num_out_features
                _end_idx = (_poly_idx + 1) * self._spam_num_out_features
                out.append(self._spam[_poly_idx](out_feats[:, _start_idx:_end_idx]))
            out = torch.sum(torch.stack(out, dim=-1), dim=-1)
        else:
            out = self.classifier(out_feats)

        if self.training:
            return out, out_feats
        else:
            return out


@MODELS_REGISTRY.register()
class ConceptNBMNarySparse(nn.Module):
    """
    Neural network (MLP) learns set of bases functions that are global,
    which are then used on each concept feature tuple individually.

    NBM model where higher order interactions of features are modeled in bases
    as f(xi, xj) for order 2 or f(xi, xj, xk) for arbitrary order d.

    NBM where not every n-tuple is fed into the model, but we use a threshold
    to sparsify input. Thus, we can support more n-ary terms, eg 100k or more.

    ref:
        Neural Bases Model.
    """

    def __init__(
        self,
        num_concepts,
        num_classes,
        nary=None,
        num_bases=100,
        hidden_dims=(256, 128, 128),
        dropout=0.0,
        bases_dropout=0.0,
        batchnorm=False,
        output_penalty=0.0,
        nary_ignore_input=0.0,
    ):
        """Initializing NBM hyperparameters.

        Args:
            num_concepts: Number of concepts used as input to the model.
            num_classes: Number of output classes of the model.
            nary (None):
                None:: unary model with all features is initialized.
                List[int]:: list of n-ary orders to be initialized, eg,
                    [1] or [1, 2, 4] or [2, 3].
                Dict[str, List[Tuple]]:: for each order (key) a list of
                    index tuples (value) is given. Only those preselected indices
                    are used in the model. Eg,
                    {"1": [(0, ), (1, ), (2, )], "2": [(0, 1), (1, 2)]}
            num_bases: Number of bases learned.
            hidden_dims: Number of hidden units for neural MLP bases part of model.
            dropout (0.0): Coefficient for dropout within neural MLP bases.
            bases_dropout (0.0): Coefficient for dropping out entire basis.
            batchnorm (True): Whether to use batchnorm or not.
            nary_ignore_input (0.0): Input value to be ignored in computation,
                so that sparse input is handled. Dictionary is also allowed, eg,
                nary_ignore_input={"1": -1.0, "2": 0.0}, to handle different
                ignore_input per nary order.
        """
        super(ConceptNBMNarySparse, self).__init__()
        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._num_bases = num_bases
        self._batchnorm = batchnorm
        self._output_penalty = output_penalty

        if nary is None:
            # if no nary specified, unary model is initialized
            self._nary_indices = {"1": list(combinations(range(self._num_concepts), 1))}
        elif isinstance(nary, list) or isinstance(nary, ListConfig):
            self._nary_indices = {
                str(order): list(combinations(range(self._num_concepts), order))
                for order in nary
            }
        elif isinstance(nary, dict):
            self._nary_indices = nary
        else:
            raise TypeError("'nary': None or list or dict supported")

        # set the input value that should be ignored to achieve sparse compute
        if isinstance(nary_ignore_input, float):
            self._nary_ignore_input = OrderedDict(
                {order: nary_ignore_input for order in self._nary_indices.keys()}
            )
        elif isinstance(nary_ignore_input, dict):
            self._nary_ignore_input = OrderedDict(sorted(nary_ignore_input.items()))
        else:
            raise TypeError("'nary_ignore_input': should be float or dict")

        self.bases_nary_models = nn.ModuleDict(
            {
                order: ConceptNNBasesNary(
                    order=int(order),
                    num_bases=self._num_bases,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    batchnorm=batchnorm,
                )
                for order in self._nary_indices.keys()
            }
        )

        self.bases_dropout = nn.Dropout(p=bases_dropout)

        self.featurizer_params = nn.ModuleDict(
            {
                order: nn.ParameterDict(
                    {
                        "weight": nn.Parameter(
                            torch.empty(
                                (len(self._nary_indices[order]), self._num_bases)
                            )
                        ),
                        "bias": nn.Parameter(
                            torch.empty(len(self._nary_indices[order]))
                        ),
                    }
                )
                for order in self._nary_indices.keys()
            }
        )

        num_out_features = sum(
            len(self._nary_indices[order]) for order in self._nary_indices.keys()
        )
        self.classifier = nn.Linear(
            in_features=num_out_features,
            out_features=self._num_classes,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for order in self._nary_indices.keys():
            nn.init.kaiming_uniform_(
                self.featurizer_params[order]["weight"], a=math.sqrt(5)
            )
            if self.featurizer_params[order]["bias"] is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.featurizer_params[order]["weight"]
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.featurizer_params[order]["bias"], -bound, bound)

    def forward(self, input):
        out_feats = []
        for order in self._nary_indices.keys():
            input_order = input[:, self._nary_indices[order]]

            # sparsify based on ignore_input value for respective order
            ignore_input = self._nary_ignore_input[order]
            sparse_indices = torch.any(input_order != ignore_input, dim=-1)

            bases_sparse = self.bases_dropout(
                self.bases_nary_models[order](input_order[sparse_indices, :])
            )

            weight = self.featurizer_params[order]["weight"][
                sparse_indices.nonzero()[:, -1], :
            ]
            bias = self.featurizer_params[order]["bias"][
                sparse_indices.nonzero()[:, -1]
            ]

            out_feats_sparse = torch.mul(weight, bases_sparse).sum(dim=-1) + bias

            out_feats_dense = torch.zeros(
                (input_order.shape[0], input_order.shape[1])
            ).to(input.device)
            out_feats_dense[sparse_indices] = out_feats_sparse

            out_feats.append(out_feats_dense)

        out_feats = torch.cat(out_feats, dim=-1)

        out = self.classifier(out_feats)

        if self.training:
            return out, out_feats
        else:
            return out
