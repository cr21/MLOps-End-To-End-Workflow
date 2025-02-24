import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

class GenericDataModule(L.LightningDataModule):
    def __init__(self, data: Union[str, Path] = "data", batch_size: int = 32, num_workers: int = 4, split: tuple[float, float] = (0.8, 0.1, 0.1),train_transform: transforms.Compose = None, test_transform: transforms.Compose = None, name: str = "generic_datamodule", image_size: tuple[int, int] = (224, 224), pin_memory: bool = True):
        super().__init__()
        self.data_dir = Path(data)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.name = name
        self.image_size = image_size
        self.pin_memory = pin_memory
        self.default_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.default_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = self.train_transform or self.default_train_transform
        self.test_transform = self.test_transform or self.default_test_transform
        self.class_names = None
    
    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            self.train_dataset = datasets.ImageFolder(self.data_dir / "train", transform=self.train_transform)
            self.val_dataset = datasets.ImageFolder(self.data_dir / "val", transform=self.test_transform)
            self.class_names = self.train_dataset.classes if hasattr(self.train_dataset, 'classes') else None
        else:
            self.test_dataset = datasets.ImageFolder(self.data_dir / "test", transform=self.test_transform)
            self.class_names = self.test_dataset.classes if hasattr(self.test_dataset, 'classes') else None
        print("CLASS NAMES: ", self.class_names)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=self.pin_memory)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=False,
                          pin_memory=self.pin_memory) 
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            pin_memory=self.pin_memory)
