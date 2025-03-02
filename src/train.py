import os
from pathlib import Path
import torch
import rootutils
# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")
import pytorch_lightning as pl
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hydra
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger
from typing import List
from utils.logging_utils import setup_logger  # Import the setup_logger function
from src.utils.s3_utility import upload_file_to_s3, remove_files
# Set up logging
log = logging.getLogger(__name__)


def instantiate_callbacks(callback_cfg: DictConfig) -> List[pl.Callback]:
    """Instantiate callbacks from Hydra configuration."""
    callbacks: List[pl.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiate loggers from Hydra configuration."""
    loggers: List[pl.LightningLoggerBase] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Generate and save a confusion matrix plot with percentage values."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Check for rows that sum to zero and handle them
    if np.any(cm.sum(axis=1) == 0):
        log.warning("Confusion matrix contains rows that sum to zero. Adjusting to avoid division by zero.")
        cm_percentage = np.zeros_like(cm, dtype=float)  # Set to zero if any row sums to zero
    else:
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize to get percentages

    plt.figure(figsize=(20, 16))  # Increased figure size for better visibility
    sns.heatmap(
        cm_percentage,
        annot=False,  # Set to True to show percentages
        fmt='.2%',  # Format for annotations
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'},  # Add a color bar label
        annot_kws={"size": 8},  # Adjust annotation font size
        linewidths=.5,  # Add lines between cells
        linecolor='black'  # Color of the lines between cells
    )
    
    plt.title(title, fontsize=20)  # Increase title font size
    plt.xlabel('Predicted Labels', fontsize=16)  # Increase x-axis label font size
    plt.ylabel('True Labels', fontsize=16)  # Increase y-axis label font size
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels for better visibility
    plt.yticks(rotation=0, fontsize=10)  # Rotate y-axis labels for better visibility
    plt.tight_layout(pad=2)  # Add padding around the plot
    plt.savefig(filename)
    plt.close()

def train_model(cfg: DictConfig, trainer: pl.Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule):
    """Train the model."""
    trainer.fit(model, datamodule)

    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        log.info(f"Best model saved at {trainer.checkpoint_callback.best_model_path}")
        log.info(f"Best model score {trainer.checkpoint_callback.best_model_score}")
        s3_model_save_location_path = cfg.s3_model_save_location
        print(f"S3 model save location path: {s3_model_save_location_path}")
        upload_file_to_s3(trainer.checkpoint_callback.best_model_path, s3_model_save_location_path)
        print(f"Model uploaded to S3 at {s3_model_save_location_path}")
        best_model  = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        log.warning("No best model found! Skipping.. and use existing model")
        best_model = model
    
    # Ensure the directory exists
    output_dir = Path(cfg.paths.static_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist

    # After training, get predictions and plot confusion matrix
    y_true, y_pred = get_predictions(trainer, model, datamodule, stage='train')
    plot_confusion_matrix(y_true.cpu(), y_pred.cpu(), class_names=datamodule.class_names, 
                          title="Confusion Matrix - Training", 
                          filename=output_dir / "confusion_matrix_train.png")  # Save to the created directory
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")
    return train_metrics
    

def test_model(cfg: DictConfig, trainer: pl.Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule):
    """Test the model."""
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        log.info(f"Best model saved at {trainer.checkpoint_callback.best_model_path}")
        log.info(f"Best model score {trainer.checkpoint_callback.best_model_score}")
        best_model  = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        log.warning("No best model found! Skipping.. and use existing model")
        best_model = model

    test_metrics = trainer.test(best_model, datamodule)

    # Ensure the directory exists
    output_dir = Path(cfg.paths.static_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist

    # After testing, get predictions and plot confusion matrix
    y_true, y_pred = get_predictions(trainer, model, datamodule, stage='test')
    plot_confusion_matrix(y_true.cpu(), y_pred.cpu(), class_names=datamodule.class_names, 
                          title="Confusion Matrix - Testing", 
                          filename=output_dir / "confusion_matrix_test.png")  # Save to the created directory

    log.info(f"Test metrics:\n{test_metrics}")  
    return test_metrics[0] if  test_metrics else {}


def get_predictions(trainer: pl.Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule, stage: str = 'train'):
    """Get predictions from the model."""
    dataloader = datamodule.train_dataloader() if stage == 'train' else datamodule.test_dataloader()
    y_true, y_pred = [], []

    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device of the model

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            x, y = batch  # Assuming the batch returns (inputs, labels)
            x, y = x.to(device), y.to(device)  # Move inputs and targets to the same device
            y_true.append(y)  # Collect true labels
            preds = model(x)  # Get model predictions
            y_pred.append(torch.argmax(preds, dim=1))  # Get predicted classes

    # Concatenate all true and predicted labels
    y_true = torch.cat(y_true).to('cpu')  # Move to CPU for plotting
    y_pred = torch.cat(y_pred).to('cpu')  # Move to CPU for plotting

    return y_true, y_pred

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Main function to set up and run training."""
    # Set up the root directory (if needed)
    log_dir = Path(cfg.paths.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Set up logger using logging_utils
    log_file = log_dir / "train.log"  # Define the log file path
    setup_logger(log_file)  # Call the setup_logger function

    # Create data module
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Create model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate callbacks and loggers
    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Create trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    # Train the model
    if cfg.train:
        log.info("Starting training!")
        train_model(cfg, trainer, model, datamodule)


    # Test the model
    if cfg.test:
        log.info("Starting testing!")
        test_model(cfg, trainer, model, datamodule)

    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimization_metrics")
    log.info(f"Optimized metric: {optimized_metric}")
    log.info(f"Trainer callback metrics: {trainer.callback_metrics}")
    if optimized_metric and optimized_metric in trainer.callback_metrics:
        return trainer.callback_metrics[optimized_metric]
if __name__ == "__main__":
    main()
