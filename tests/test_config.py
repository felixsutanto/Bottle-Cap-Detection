"""Tests for configuration management."""

import pytest
import yaml
from pathlib import Path
from bsort.config import Config


class TestConfig:
    """Test configuration loading."""
    
    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create sample configuration file."""
        config_dict = {
            'dataset': {
                'root_path': './data',
                'images_path': './data/images',
                'labels_path': './data/labels',
                'train_split': 0.8,
                'val_split': 0.2
            },
            'color_classification': {
                'light_blue': {
                    'hue_min': 170, 'hue_max': 210,
                    'sat_min': 20, 'sat_max': 60,
                    'val_min': 60, 'val_max': 100
                },
                'dark_blue': {
                    'hue_min': 200, 'hue_max': 240,
                    'sat_min': 40, 'sat_max': 100,
                    'val_min': 20, 'val_max': 60
                }
            },
            'model': {
                'name': 'yolov8n',
                'pretrained': True,
                'num_classes': 3,
                'input_size': 640
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'patience': 20,
                'device': 'cuda:0',
                'workers': 4,
                'augmentation': {},
                'optimizer': 'SGD',
                'lr_scheduler': 'cosine'
            },
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_det': 300
            },
            'inference': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_det': 100,
                'save_results': True,
                'output_dir': './outputs'
            },
            'wandb': {
                'enabled': True,
                'project': 'bottlecap-detection',
                'entity': None,
                'name': None
            }
        }
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        return config_path
    
    def test_load_config(self, sample_config):
        """Test configuration loading from YAML."""
        config = Config.from_yaml(str(sample_config))
        
        assert config.model.name == 'yolov8n'
        assert config.model.num_classes == 3
        assert config.training.epochs == 100
        assert config.training.batch_size == 16
    
    def test_config_structure(self, sample_config):
        """Test configuration has all required fields."""
        config = Config.from_yaml(str(sample_config))
        
        assert hasattr(config, 'dataset')
        assert hasattr(config, 'color_classification')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'validation')
        assert hasattr(config, 'inference')
        assert hasattr(config, 'wandb')