"""Model validation logic."""

from ultralytics import YOLO


class Validator:
    """Validation pipeline for model evaluation."""
    
    def __init__(self, config, weights_path=None):
        """
        Initialize validator.
        
        Args:
            config: Configuration object.
            weights_path: Path to model weights.
        """
        self.config = config
        
        # Load model
        if weights_path:
            self.model = YOLO(weights_path)
        else:
            self.model = YOLO('runs/train/exp/weights/best.pt')
    
    def validate(self, dataset_split='val'):
        """
        Validate model on dataset.
        
        Args:
            dataset_split: Dataset split to validate on.
            
        Returns:
            Validation metrics dictionary.
        """
        metrics = self.model.val(
            data='data.yaml',
            split=dataset_split,
            conf=self.config.validation.conf_threshold,
            iou=self.config.validation.iou_threshold,
        )
        
        return metrics.results_dict