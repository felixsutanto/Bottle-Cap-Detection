"""Configuration management for the bottle cap detection system."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    root_path: str
    images_path: str
    labels_path: str
    train_split: float = 0.8
    val_split: float = 0.2
    names: list = field(default_factory=list)


@dataclass
class ColorClassificationConfig:
    """Color classification thresholds."""
    light_blue: Dict[str, int] = field(default_factory=dict)
    dark_blue: Dict[str, int] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "yolov8n"
    pretrained: bool = True
    num_classes: int = 3
    input_size: int = 640


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 3
    patience: int = 20
    device: str = "cuda:0"
    workers: int = 4
    augmentation: Dict[str, float] = field(default_factory=dict)
    optimizer: str = "SGD"
    lr_scheduler: str = "cosine"


@dataclass
class ValidationConfig:
    """Validation configuration."""
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300


@dataclass
class InferenceConfig:
    """Inference configuration."""
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 100
    save_results: bool = True
    output_dir: str = "./outputs"


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "bottlecap-detection"
    entity: Optional[str] = None
    name: Optional[str] = None
    mode: str = "online"


@dataclass
class Config:
    """Main configuration class."""
    dataset: DatasetConfig
    color_classification: ColorClassificationConfig
    model: ModelConfig
    training: TrainingConfig
    validation: ValidationConfig
    inference: InferenceConfig
    wandb: WandbConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file.
            
        Returns:
            Config object.
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            dataset=DatasetConfig(**config_dict['dataset']),
            color_classification=ColorClassificationConfig(**config_dict['color_classification']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            validation=ValidationConfig(**config_dict['validation']),
            inference=InferenceConfig(**config_dict['inference']),
            wandb=WandbConfig(**config_dict.get('wandb', {})),
        )