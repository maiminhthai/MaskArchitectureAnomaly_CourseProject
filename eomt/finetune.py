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
from main import LightningCLI
from typing import List
from lightning.pytorch.callbacks import ModelCheckpoint


class FinetuneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add finetune specific arguments as top-level subcommand args
        parser.add_argument("--logit_norm_temperature", type=float, default=0.01, help="Temperature for Logit Normalization")
        parser.add_argument("--targets", type=List[str], default=["class_head", "mask_head"], help="Modules names to finetune")

        # Call super to add standard LightningCLI arguments (model, data, trainer, etc.)
        # and the links already defined in main.py:LightningCLI
        super().add_arguments_to_parser(parser)

    def fit(self, model, **kwargs):
        # subclass_mode_model=True means model is already instantiated by CLI
        config = self.config[self.config["subcommand"]]
        logit_norm_temperature = config.get("logit_norm_temperature", 0.01)
        targets = config.get("targets", ["class_head", "mask_head"])

        for param in model.network.parameters():
            param.requires_grad = False
                
        # Unfreeze heads
        for module_name in targets:
            if hasattr(model.network, module_name):
                logging.info(f"Unfreezing {module_name}")
                for param in getattr(model.network, module_name).parameters():
                    param.requires_grad = True


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
        if self.trainer.logger is not None and hasattr(self.trainer.logger, "experiment") and hasattr(self.trainer.logger.experiment, "log_code"):
            from gitignore_parser import parse_gitignore
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

        if not config.get("compile_disabled", False):
            # model = torch.compile(model)
            pass

        self.trainer.fit(model, **kwargs)


def cli_main():
    cli = FinetuneCLI(
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