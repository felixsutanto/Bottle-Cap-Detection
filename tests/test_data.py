"""Tests for data processing and preprocessing."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from bsort.data.preprocessing import ColorClassifier, DataPreprocessor
from bsort.config import Config


class TestColorClassifier:
    """Test color classification logic."""
    
    @pytest.fixture
    def classifier(self):
        """Create a mock config and classifier."""
        config = type('Config', (), {
            'color_classification': type('ColorConfig', (), {
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
            })()
        })()
        return ColorClassifier(config)
    
    def test_classify_light_blue(self, classifier):
        """Test light blue classification."""
        # Create light blue image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [200, 230, 255]  # Light blue in BGR
        
        bbox = [0.5, 0.5, 0.5, 0.5]  # Center box
        result = classifier.classify_color(image, bbox)
        
        assert result in [0, 2], "Should classify as light blue or other"
    
    def test_classify_dark_blue(self, classifier):
        """Test dark blue classification."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [100, 50, 0]  # Dark blue in BGR
        
        bbox = [0.5, 0.5, 0.5, 0.5]
        result = classifier.classify_color(image, bbox)
        
        assert result in [1, 2], "Should classify as dark blue or other"
    
    def test_classify_other_color(self, classifier):
        """Test other color classification."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [0, 255, 0]  # Green in BGR
        
        bbox = [0.5, 0.5, 0.5, 0.5]
        result = classifier.classify_color(image, bbox)
        
        assert result == 2, "Should classify as other"
    
    def test_invalid_bbox(self, classifier):
        """Test handling of invalid bounding box."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Out of bounds bbox
        bbox = [1.5, 1.5, 0.5, 0.5]
        result = classifier.classify_color(image, bbox)
        
        assert result == 2, "Should return 'other' for invalid bbox"
    
    def test_empty_roi(self, classifier):
        """Test handling of empty ROI."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Zero-size bbox
        bbox = [0.5, 0.5, 0.0, 0.0]
        result = classifier.classify_color(image, bbox)
        
        assert result == 2, "Should return 'other' for empty ROI"


class TestDataPreprocessor:
    """Test data preprocessing pipeline."""
    
    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Create temporary dataset for testing."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create dummy image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "test1.jpg"), image)
        
        # Create dummy label
        with open(labels_dir / "test1.txt", 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        
        return images_dir, labels_dir
    
    def test_process_dataset(self, temp_dataset, tmp_path):
        """Test dataset processing."""
        images_dir, labels_dir = temp_dataset
        
        # Create mock config
        config = type('Config', (), {
            'dataset': type('DatasetConfig', (), {
                'images_path': str(images_dir),
                'labels_path': str(labels_dir),
            })(),
            'color_classification': type('ColorConfig', (), {
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
            })()
        })()
        
        preprocessor = DataPreprocessor(config)
        stats = preprocessor.process_dataset()
        
        assert stats['total_images'] == 1
        assert stats['total_annotations'] == 1
        assert sum(stats['class_distribution'].values()) == 1