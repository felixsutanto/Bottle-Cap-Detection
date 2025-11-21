"""Model training logic."""

from ultralytics import YOLO
from pathlib import Path
import wandb
import yaml
import os
import sys
import torch  # Added for device checking


class Trainer:
    """Training pipeline for bottle cap detection."""
    
    def __init__(self, config, resume_from=None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object.
            resume_from: Path to checkpoint to resume from.
        """
        self.config = config
        self.resume_from = resume_from
        
        # CRITICAL FIX: Force disable WandB at the system level if disabled in config.
        if not config.wandb.enabled:
            os.environ["WANDB_MODE"] = "disabled"
            os.environ["WANDB_SILENT"] = "true"
        
        # Initialize WandB if enabled
        if config.wandb.enabled:
            try:
                wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    name=config.wandb.name,
                    mode=config.wandb.mode,
                    config=self._config_to_dict(config)
                )
            except Exception as e:
                print(f"\n⚠️  WARNING: WandB initialization failed: {e}")
                print("⚠️  Proceeding with training WITHOUT experiment tracking.\n")
                os.environ["WANDB_MODE"] = "disabled"
                self.config.wandb.enabled = False
    
    def train(self):
        """
        Run training pipeline.
        
        Returns:
            Training results dictionary.
        """
        # Load model
        if self.resume_from:
            model = YOLO(self.resume_from)
        else:
            model = YOLO(f"{self.config.model.name}.pt")
        
        # Prepare data config
        data_yaml = self._create_data_yaml()
        
        # --- SMART DEVICE SELECTION ---
        target_device = self.config.training.device
        # Check if user asked for GPU (0 or cuda:0) but system only has CPU
        if str(target_device) in ["0", "cuda:0", "cuda"] and not torch.cuda.is_available():
            print("\n⚠️  WARNING: CUDA (GPU) was requested but is not available on this system.")
            print("⚠️  Switching training device to 'cpu'. Expect slower training speeds.\n")
            target_device = "cpu"
        # ------------------------------

        # Train
        results = model.train(
            data=data_yaml,
            epochs=self.config.training.epochs,
            batch=self.config.training.batch_size,
            imgsz=self.config.model.input_size,
            device=target_device,  # Use the smart device variable
            workers=self.config.training.workers,
            optimizer=self.config.training.optimizer,
            lr0=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            momentum=self.config.training.momentum,
            warmup_epochs=self.config.training.warmup_epochs,
            patience=self.config.training.patience,
            project="runs/train",
            name="exp",
            exist_ok=True,
            # Augmentations
            degrees=self.config.training.augmentation.get('degrees', 0.0),
            translate=self.config.training.augmentation.get('translate', 0.1),
            scale=self.config.training.augmentation.get('scale', 0.5),
            shear=self.config.training.augmentation.get('shear', 0.0),
            perspective=self.config.training.augmentation.get('perspective', 0.0),
            flipud=self.config.training.augmentation.get('flipud', 0.0),
            fliplr=self.config.training.augmentation.get('fliplr', 0.5),
            mosaic=self.config.training.augmentation.get('mosaic', 1.0),
            mixup=self.config.training.augmentation.get('mixup', 0.0),
        )
        
        return results.results_dict if hasattr(results, 'results_dict') else {'save_dir': str(results.save_dir)}
    
    def _create_data_yaml(self) -> str:
        """Create data.yaml file for YOLO training."""
        
        data_dict = {
            'path': str(Path(self.config.dataset.root_path).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.config.model.num_classes,
            'names': self.config.dataset.names or ['light_blue', 'dark_blue', 'other']
        }
        
        yaml_path = Path('data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_dict, f)
        
        return str(yaml_path)
    
    @staticmethod
    def _config_to_dict(config) -> dict:
        """Convert config to dictionary for WandB."""
        return {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'augmentation': config.training.augmentation,
        }