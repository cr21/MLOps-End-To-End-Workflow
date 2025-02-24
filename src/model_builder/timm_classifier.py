import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, MaxMetric
import logging
log = logging.getLogger(__name__)
class TimmClassifier(L.LightningModule):

    def _create_model(self, model_name, num_classes, pretrained=True,  **kwargs):
        """
        Create a model from a given model name and number of classes.
        Args:
            model_name (str): The name of the model to create.
            num_classes (int): The number of classes in the model.
            kwargs (dict): Additional keyword arguments for the model.
        Returns:
            torch.nn.Module: The created model.
        
        """
        log.info(f"Creating model with num_classes: {num_classes}")
        model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        model.reset_classifier(num_classes)
        log.info(f"Model created: {model}")
        return model
    
    def __init__(self, 
                base_model: str,
                num_classes: int, 
                learning_rate: float = 1e-3, 
                pretrained: bool = True, 
                weight_decay: float = 1e-5,
                patience: int = 3,
                factor: float = 0.1,
                min_lr: float = 1e-6,
                **kwargs):
        super().__init__()
        self.model = self._create_model(base_model, num_classes, pretrained, **kwargs)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="weighted")
        self.train_f1_score = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="weighted")
        self.val_f1_score = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1_score = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.save_hyperparameters()
        self.test_acc_best = MaxMetric()

    def forward(self, x):
        return self.model(x)
    
    def __common_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        return loss, out, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.__common_step(batch, batch_idx)
        self.log_dict({
            'train/loss': loss,
            'train/acc': self.train_acc(y_hat, y), 
            'train/f1_score': self.train_f1_score(y_hat, y)},
            on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.__common_step(batch, batch_idx)
        self.log_dict({
            'val/loss': loss, 
            'val/acc': self.val_acc(y_hat, y),
            'val/f1_score': self.val_f1_score(y_hat, y)}, 
            on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.__common_step(batch, batch_idx)
        self.log_dict({
            'test/loss': loss,
            'test/acc': self.test_acc(y_hat, y),
            'test/f1_score': self.test_f1_score(y_hat, y)},
            on_step=False, on_epoch=True, prog_bar=True)
        return loss

    
    def on_test_epoch_end(self):
        self.test_acc_best.update(self.test_acc.compute())
        self.log('test/acc_best', self.test_acc_best.compute(), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            }
        }
