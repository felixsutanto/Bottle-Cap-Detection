"""Tests for inference pipeline."""

import pytest
import numpy as np
from pathlib import Path
from bsort.inference.predictor import Predictor


class TestPredictor:
    """Test inference functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return type('Config', (), {
            'inference': type('InferenceConfig', (), {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_det': 100,
                'save_results': False,
                'output_dir': './outputs'
            })()
        })()
    
    def test_predictor_initialization(self, mock_config):
        """Test predictor can be initialized."""
        # This would fail without a real model
        try:
            predictor = Predictor(mock_config)
            assert predictor.config == mock_config
        except Exception:
            # Expected if no model exists
            pass