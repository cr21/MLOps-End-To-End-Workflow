import os
from pathlib import Path
import sys
import rootutils
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict, Any
import shutil

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Rest of your imports
from src.utils.logging_utils import setup_logger, task_wrapper
from src.train import train_model, test_model, instantiate_callbacks, instantiate_loggers

# Set up logging
log = logging.getLogger(__name__)

def find_best_hparams(log_dir: Path) -> Dict[str, Any]:
    best_acc = 0
    best_hparams = {}
    
    for run_dir in log_dir.glob("*/*"):
        metrics_files = list(run_dir.glob("logs/csv_logs/version_*/metrics.csv"))
        if not metrics_files:
            continue
        
        for metrics_file in metrics_files:
            metrics_df = pd.read_csv(metrics_file)
            test_acc = metrics_df[metrics_df['test/acc_best'].notna()]['test/acc_best'].max()
            
            if test_acc > best_acc:
                best_acc = test_acc
                hparam_file = metrics_file.parent / "hparams.yaml"
                if hparam_file.exists():
                    best_hparams = OmegaConf.load(hparam_file)
    
    log.info(f"Best test accuracy: {best_acc}")
    log.info(f"Best hyperparameters: {best_hparams}")
    return best_hparams

def update_model_config(cfg: DictConfig, hparams: Dict[str, Any]) -> DictConfig:
    # Update only the model-specific parameters
    if "model" not in cfg:
        cfg.model = {}
    
    for key, value in hparams.items():
        if key in ["base_model", "num_classes", "pretrained",
                   "learning_rate", "weight_decay", "patience", "factor", "min_lr"]:
            cfg.model[key] = value
        # elif key in []:
        #     if "optimizer" not in cfg.model:
        #         cfg.model.optimizer = {}
        #     cfg.model.optimizer[key] = value
    
    return cfg

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up the root directory and logger
    log_dir = Path(cfg.paths.log_dir)
    setup_logger(log_dir/"train_optimized_hparam.log")

    # Find the best hyperparameters
    multirun_log_dir = log_dir / "train" / "multiruns"
    best_hparams = find_best_hparams(multirun_log_dir)
    
    # Create markdown_results directory if it doesn't exist
    markdown_results_dir = Path("markdown_results") / cfg.name
    markdown_results_dir.mkdir(exist_ok=True)
    
    # Write best hparams to a markdown file inside markdown_results folder
    with open(markdown_results_dir / "best_hparams.md", "w") as f:
        f.write(OmegaConf.to_yaml(best_hparams))
    
    # Update the config with the best hyperparameters
    cfg = update_model_config(cfg, best_hparams)

    # Construct the checkpoint path
    checkpoints_dir = Path("checkpoints") / cfg.name

    # Delete existing checkpoints directory and recreate it
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Recreated checkpoints directory: {checkpoints_dir}")

    # Create data module
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    # Create model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # Instantiate callbacks and loggers
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))

    # Create trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    # Train the model with best hyperparameters
    if cfg.get("train"):
        train_metrics = train_model(cfg, trainer, model, datamodule)
    else:
        train_metrics = {}

    # Test the model
    if cfg.get("test"):
        test_metrics = test_model(cfg, trainer, model, datamodule)
    else:
        test_metrics = {}

    all_metrics = {**train_metrics, **test_metrics}
    log.info(f"Final metrics:\n{all_metrics}")

if __name__ == "__main__":
    main()
