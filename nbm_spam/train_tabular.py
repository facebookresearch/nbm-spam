# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import time
from math import ceil, floor
from typing import Any, List, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from hydra.utils import instantiate
from nbm_spam.dataset import build_dataset
from nbm_spam.models import build_model
from nbm_spam.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import Tensor


MODEL_MAP = {
    "linear": ("ConceptLinear", False),
    "mlp": ("ConceptMLP", True),
    "nam": ("ConceptNAMNary", True),
    "nbm": ("ConceptNBMNary", True),
    "nbm_sparse": ("ConceptNBMNarySparse", True),
    "spam": ("ConceptSPAM", True),
}


class TabularPredictionModule(pl.LightningModule):
    def __init__(
        self,
        num_concepts,
        num_classes,
        learning_rate,
        momentum,
        weight_decay,
        criterion,
        optimizer,
        model,
        model_params=None,
        trainer=None,
        datamodule=None,
        warmup=None,
        ckpt_dir=None,
    ):
        super(TabularPredictionModule, self).__init__()
        self.MODEL_ATTR = "model"
        self.MODEL_NAME_ATTR = "model"
        self.PARAM_ATTR = "model_params"

        self.save_hyperparameters(
            "num_concepts",
            "num_classes",
            "learning_rate",
            "momentum",
            "weight_decay",
            "criterion",
            "optimizer",
            "model",
            "model_params",
            "trainer",
            "datamodule",
            "warmup",
            "ckpt_dir",
        )

        self._init_criterion()
        self._init_metrics()
        self._init_model()
        self._init_logging_names()

        print(self.model)
        self._epoch_time = time.time()

    def _init_criterion(self) -> None:
        self._binary_classification = False
        self._regression = False

        assert self.hparams.criterion in [
            "BCEWithLogitsLoss",
            "MSELoss",
            "CrossEntropyLoss",
        ], "`criterion` must be in [`BCEWithLogitsLoss`, `MSELoss`, `CrossEntropyLoss`]"

        if self.hparams.criterion == "BCEWithLogitsLoss":
            print("`BCEWithLogitsLoss` criterion, learning logistic regression")
            assert (
                self.hparams.num_classes == 2
            ), "`num_classes` must be 2 for binary classification"
            self._binary_classification = True
        elif self.hparams.criterion == "MSELoss":
            print("`MSELoss` criterion, learning least-squares regression")
            assert (
                self.hparams.num_classes == 1
            ), "`num_classes` must be 1 for regression"
            self._regression = True
        else:
            print("`CrossEntropyLoss` criterion, learning multi-class classification")
            assert (
                self.hparams.num_classes > 2
            ), "`num_classes` must be greater than 2 for multi-class"
        self.criterion = getattr(nn, self.hparams.criterion)()

    def _init_metrics(self):
        for key in ["train", "val", "test"]:
            if self._binary_classification:
                setattr(self, f"{key}_m1", torchmetrics.AUROC())
                setattr(self, f"{key}_m2", torchmetrics.AveragePrecision(pos_label=1))
            elif self._regression:
                setattr(self, f"{key}_m1", torchmetrics.MeanSquaredError(squared=False))
                setattr(self, f"{key}_m2", torchmetrics.R2Score())
            else:
                setattr(self, f"{key}_m1", torchmetrics.Accuracy(top_k=1))
                setattr(self, f"{key}_m2", torchmetrics.Accuracy(top_k=5))

    def _init_model(self):
        assert self.hparams.model in MODEL_MAP, "Incorrect model name provided."
        model_name, _load_kwargs = MODEL_MAP[self.hparams.model]
        model_kwargs = self.hparams.model_params if _load_kwargs else {}
        self.model = build_model(
            model_name,
            self.hparams.num_concepts,
            self.hparams.num_classes - int(self._binary_classification),
            **model_kwargs,
        )

    def _init_logging_names(self) -> None:
        # basic logging names
        for key in ["train", "val", "test"]:
            setattr(self, f"_loggingname_{key}_loss", f"{key}/loss")
            if self._binary_classification:
                setattr(self, f"_loggingname_{key}_m1", f"{key}/auroc")
                setattr(self, f"_loggingname_{key}_m2", f"{key}/ap")
            elif self._regression:
                setattr(self, f"_loggingname_{key}_m1", f"{key}/rmse")
                setattr(self, f"_loggingname_{key}_m2", f"{key}/r2")
            else:
                setattr(self, f"_loggingname_{key}_m1", f"{key}/acc1")
                setattr(self, f"_loggingname_{key}_m2", f"{key}/acc5")

        # extra logging names
        _model_name = self.rgetattr(self.hparams, self.MODEL_NAME_ATTR)
        if _model_name in [
            "nam",
            "nbm",
            "nbm_sparse",
        ]:
            for lname in ["output_penalty"]:
                setattr(self, f"_loggingname_train_{lname}_loss", f"train/{lname}")

        if _model_name in ["spam"]:
            for lname in ["regularization", "basis_l1_loss"]:
                setattr(self, f"_loggingname_train_{lname}_loss", f"train/{lname}")

    def _compute_forward(self, inputs: Tensor) -> Tensor:
        _model = self.rgetattr(self, self.MODEL_ATTR)
        return _model(inputs)

    def _compute_criterion(self, preds: Tensor, targets: Tensor) -> Tensor:
        if self._binary_classification and preds.shape[-1] == 1:
            return self.criterion(preds.squeeze(-1), targets.float())
        elif self._regression:
            return self.criterion(preds.squeeze(-1), targets)
        return self.criterion(preds, targets)

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log(
            "time/epoch",
            time.time() - self._epoch_time,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )
        self._epoch_time = time.time()
        return super(TabularPredictionModule, self).training_epoch_end(outputs)

    def _get_outputs_and_losses(
        self,
        inputs: Tensor,
        targets: Tensor,
        key: str = "train",
    ) -> Tuple[Tensor, Tensor]:

        _model = self.rgetattr(self, self.MODEL_ATTR)
        _model_name = self.rgetattr(self.hparams, self.MODEL_NAME_ATTR)
        _model_hparams = self.rgetattr(self.hparams, self.PARAM_ATTR)

        if key in ["val", "test"]:
            outputs = self._compute_forward(inputs)
            loss = self._compute_criterion(outputs, targets)
            return loss, outputs

        if _model_name in [
            "nam",
            "nbm",
            "nbm_sparse",
        ]:
            (outputs, outputs_nn) = self._compute_forward(inputs)
            loss = self._compute_criterion(outputs, targets)
            output_penalty_loss = self._output_penalty(outputs_nn) * _model_hparams.get(
                "output_penalty", 0
            )
            loss += output_penalty_loss

            extra_metric_names = ["output_penalty"]
            extra_metric_values = [output_penalty_loss.detach()]

        elif _model_name == "spam":
            outputs = self._compute_forward(inputs)
            loss = self._compute_criterion(outputs, targets)
            reg_loss = _model.tensor_regularization()
            loss += reg_loss * _model_hparams.get("regularization_scale", 0)

            basis_l1_loss = _model.basis_l1_regularization()
            loss += basis_l1_loss * _model_hparams.get("basis_l1_regularization", 0)

            extra_metric_names = ["regularization", "basis_l1_loss"]
            extra_metric_values = [reg_loss.detach(), basis_l1_loss.detach()]
        else:
            outputs = self._compute_forward(inputs)
            loss = self._compute_criterion(outputs, targets)

            extra_metric_names = []
            extra_metric_values = []

        self._log_additional_metrics(extra_metric_names, extra_metric_values)

        return loss, outputs

    def _log_progress_bar_metrics(
        self, outputs: Tensor, labels: Tensor, loss: Tensor, key: str = "train"
    ) -> None:
        _outputs, _loss = outputs.detach(), loss.detach()
        _loss_handle = getattr(self, f"_loggingname_{key}_loss")
        _m1_handle = getattr(self, f"_loggingname_{key}_m1")
        _m2_handle = getattr(self, f"_loggingname_{key}_m2")

        self.log(
            _loss_handle,
            _loss.cpu().item(),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )

        if self._binary_classification:
            _outputs = torch.sigmoid(_outputs).squeeze(-1)
        elif self._regression:
            _outputs = _outputs.squeeze(-1)

        if key == "train":
            self.train_m1.update(_outputs, labels)
            self.log(_m1_handle, self.train_m1, on_epoch=True, prog_bar=True)
            self.train_m2.update(_outputs, labels)
            self.log(_m2_handle, self.train_m2, on_epoch=True, prog_bar=True)
        elif key == "val":
            self.val_m1.update(_outputs, labels)
            self.log(_m1_handle, self.val_m1, on_epoch=True, prog_bar=True)
            self.val_m2.update(_outputs, labels)
            self.log(_m2_handle, self.val_m2, on_epoch=True, prog_bar=True)
        elif key == "test":
            self.test_m1.update(_outputs, labels)
            self.log(_m1_handle, self.test_m1, on_epoch=True, prog_bar=True)
            self.test_m2.update(_outputs, labels)
            self.log(_m2_handle, self.test_m2, on_epoch=True, prog_bar=True)

    def _log_additional_metrics(
        self, metric_names: List[str], metric_values: List[Tensor]
    ) -> None:
        for _metric, _value in zip(metric_names, metric_values):
            lname = getattr(self, f"_loggingname_train_{_metric}_loss")
            self.log(
                lname,
                _value.cpu().item(),
                on_epoch=True,
                sync_dist=True,
                reduce_fx="mean",
                prog_bar=True,
            )

    def training_step(
        self, batch: Tensor, batch_idx: int, key: str = "train"
    ) -> Tensor:
        inputs, targets = batch[0], batch[1]
        self.model.train()

        loss, outputs = self._get_outputs_and_losses(inputs, targets, key)
        self._log_progress_bar_metrics(outputs, targets, loss, key=key)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int, key: str = "val") -> None:
        inputs, targets = batch[0], batch[1]
        self.model.eval()

        loss, outputs = self._get_outputs_and_losses(inputs, targets, key)
        self._log_progress_bar_metrics(outputs, targets, loss, key=key)

    def test_step(self, batch: Tensor, batch_idx: int, key: str = "test") -> None:
        inputs, targets = batch[0], batch[1]
        self.model.eval()

        loss, outputs = self._get_outputs_and_losses(inputs, targets, key)
        self._log_progress_bar_metrics(outputs, targets, loss, key=key)

    def configure_optimizers(self) -> None:
        _model = self.rgetattr(self, self.MODEL_ATTR)
        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                _model.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                _model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                _model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer '{self.hparams.optimizer}'")

        total_steps = self.hparams.datamodule.train_dataset_size
        total_batch_size = (
            self.hparams.datamodule.batch_size
            * self.hparams.trainer.gpus
            * self.hparams.trainer.num_nodes
        )
        max_steps = (
            ceil(total_steps / total_batch_size) * self.hparams.trainer.max_epochs
        )

        if self.hparams.warmup is None:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=max_steps
            )
        else:
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                max_epochs=max_steps,
                warmup_epochs=floor(max_steps * self.hparams.warmup),
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }

    def _output_penalty(self, output: Tensor) -> Tensor:
        return (torch.pow(output, 2).mean(dim=-1)).mean()

    @staticmethod
    def rgetattr(obj: Any, attr: str, *args: str) -> Any:
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))


@hydra.main(config_path="config")
def main(config: DictConfig) -> None:

    logger = logging.getLogger(__name__)
    logger.info("Training interpretable models on tabular datasets")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    f_trainer = pl.Trainer(callbacks=[lr_monitor], **config.trainer)

    train_dataset = build_dataset(split="train", **config.datamodule.dataset)
    val_dataset = build_dataset(split="validation", **config.datamodule.dataset)
    test_dataset = build_dataset(split="test", **config.datamodule.dataset)
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.datamodule.batch_size,
    )

    # adding data parameters to config

    with open_dict(config.tabular_prediction_module):
        config.tabular_prediction_module.trainer = config.trainer
        config.tabular_prediction_module.datamodule = config.datamodule
        config.tabular_prediction_module.datamodule.train_dataset_size = len(
            train_dataset
        )
        config.tabular_prediction_module.datamodule.val_dataset_size = len(val_dataset)
        config.tabular_prediction_module.datamodule.test_dataset_size = len(
            test_dataset
        )
        config.tabular_prediction_module.ckpt_dir = os.getcwd()

    model = instantiate(config.tabular_prediction_module)
    f_trainer.fit(model=model, datamodule=datamodule)
    f_trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
