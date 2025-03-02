import os
from pathlib import Path
import rootutils
import json
from typing import List

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import Logger
import logging
from src.utils.logging_utils import setup_logger

# Set up logging
log = logging.getLogger(__name__)

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def evaluate(cfg: DictConfig):
    log_dir = Path(cfg.paths.log_dir)
    # set up logger
    setup_logger(log_dir/"evaluation.log")

    # Set up paths
    ckpt_path = Path(cfg.ckpt_path)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    log.info(f"Loading model from {ckpt_path}")
    model = hydra.utils.instantiate(cfg.model)
    model = model.__class__.load_from_checkpoint(ckpt_path)

    # Set up data module
    log.info("Setting up data module")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.setup("test")

    # Instantiate loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Set up trainer
    log.info("Instantiating Trainer")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    # Perform evaluation
    log.info("Starting evaluation")
    validation_output = trainer.validate(model=model, datamodule=data_module)
    
    log.info(f"Validation output: {validation_output}")

    # Save results
    results_file = save_dir / "eval_results.json"
    log.info(f"Saving evaluation results to {results_file}")
    try:
        with results_file.open("w") as f:
            json.dump(validation_output, f, indent=2)
    except Exception as exp:
        log.error(f"Exception while writing json results: {exp}")

    log.info(f"Evaluation complete. Results saved in {results_file}")

if __name__ == '__main__':
    evaluate()