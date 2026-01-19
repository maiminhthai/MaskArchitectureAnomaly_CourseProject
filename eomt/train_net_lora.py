import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import logging
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule
from training.mask_classification_semantic import MaskClassificationSemantic
from training.logit_norm_loss import LogitNormMaskClassificationLoss
from training.lora import inject_lora
from main import LightningCLI
import math
import torch.nn.functional as F
from typing import List
from lightning.pytorch.callbacks import ModelCheckpoint


class LoRACLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add LoRA specific arguments as top-level subcommand args
        parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA adapters")
        parser.add_argument("--logit_norm_temperature", type=float, default=0.04, help="Temperature for Logit Normalization")
        parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling")
        parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
        parser.add_argument("--lora_targets", type=List[str], default=["class_head", "mask_head"], help="Modules names to inject LoRA into")

        # Call super to add standard LightningCLI arguments (model, data, trainer, etc.)
        # and the links already defined in main.py:LightningCLI
        super().add_arguments_to_parser(parser)

    def fit(self, model, **kwargs):
        # Extract LoRA args from config
        # subclass_mode_model=True means model is already instantiated by CLI
        config = self.config[self.config["subcommand"]]
        lora_rank = config.get("lora_rank", 32)
        logit_norm_temperature = config.get("logit_norm_temperature", 0.04)
        lora_alpha = config.get("lora_alpha", 64)
        lora_dropout = config.get("lora_dropout", 0.1)
        lora_targets = config.get("lora_targets", ["class_head", "mask_head"])

        # Inject LoRA
        if hasattr(model, "network"):
            targets = list(set(lora_targets))
            logging.info(f"Injecting LoRA into: {targets}")
            logging.info(f"Injecting LoRA with r={lora_rank}, alpha={lora_alpha}, targets={targets}")
            model.network = inject_lora(
                model.network, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=targets
            )

        # Replace criterion with LogitNorm
        if hasattr(model, "criterion"):
            logging.info(f"Replacing criterion with LogitNormMaskClassificationLoss (temp={logit_norm_temperature})")
            model.criterion = LogitNormMaskClassificationLoss(
                temperature=logit_norm_temperature,
                num_points=model.criterion.num_points,
                oversample_ratio=model.criterion.oversample_ratio,
                importance_sample_ratio=model.criterion.importance_sample_ratio,
                mask_coefficient=model.criterion.mask_coefficient,
                dice_coefficient=model.criterion.dice_coefficient,
                class_coefficient=model.criterion.class_coefficient,
                num_labels=model.criterion.num_labels,
                no_object_coefficient=model.criterion.eos_coef,
            )

        # Original fit logic (logging code, etc.)
        from gitignore_parser import parse_gitignore
        if self.trainer.logger is not None and hasattr(self.trainer.logger, "experiment") and hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        # Support validation step logic from main.py
        from main import _should_check_val_fx
        from types import MethodType
        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)


def cli_main():
    cli = LoRACLI(
        LightningModule, # Use base class to support subclassing in YAML
        LightningDataModule,
        subclass_mode_model=True, 
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    save_top_k=1,        # Only keep the single latest checkpoint
                    every_n_epochs=1,    # Trigger saving every epoch
                    monitor=None,        # Do not monitor metrics (implies "latest is best")
                    filename="{epoch}-{step}",
                ),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )

if __name__ == "__main__":
    cli_main()
