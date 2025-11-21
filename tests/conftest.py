"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import cv2


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box in YOLO format."""
    return [0.5, 0.5, 0.2, 0.2]  # [x_center, y_center, width, height]