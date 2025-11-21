"""Inference pipeline."""

from ultralytics import YOLO
from pathlib import Path
import cv2


class Predictor:
    """Inference pipeline for bottle cap detection."""
    
    def __init__(self, config, weights_path=None):
        """
        Initialize predictor.
        
        Args:
            config: Configuration object.
            weights_path: Path to model weights.
        """
        self.config = config
        
        # Load model
        if weights_path:
            self.weights_path = weights_path
        else:
            self.weights_path = 'runs/train/exp/weights/best.pt'
            
        # Future-proofing: Check if we are loading ONNX for edge inference
        if str(self.weights_path).endswith('.onnx'):
             # NOTE: For RPi5 deployment, we would switch to OnnxRuntime here
             # self.model = YOLO(self.weights_path, task='detect') 
             # or custom ORT session
             self.model = YOLO(self.weights_path)
        else:
            self.model = YOLO(self.weights_path)
    
    def predict(self, source, save=True, show=False):
        """
        Run inference on source.
        
        Args:
            source: Input source (image path, video path, or camera index).
            save: Save inference results.
            show: Display inference results.
            
        Returns:
            Detection results.
        """
        results = self.model.predict(
            source=source,
            conf=self.config.inference.conf_threshold,
            iou=self.config.inference.iou_threshold,
            max_det=self.config.inference.max_det,
            save=save,
            show=show,
            project=self.config.inference.output_dir,
            name="predict",
            exist_ok=True,
        )
        
        return results