"""Data preprocessing and color-based relabeling."""

import cv2
import numpy as np
import shutil
import random
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


class ColorClassifier:
    """Classify bottle cap colors based on HSV thresholds."""
    
    CLASS_NAMES = {
        0: "light_blue",
        1: "dark_blue",
        2: "other"
    }
    
    def __init__(self, config):
        """
        Initialize color classifier.
        
        Args:
            config: Configuration object with color thresholds.
        """
        self.config = config
    
    def classify_color(self, image: np.ndarray, bbox: List[float]) -> int:
        """
        Classify color of detected object using Center Crop strategy.
        
        Args:
            image: Input image in BGR format.
            bbox: Bounding box in YOLO format [x_center, y_center, width, height].
            
        Returns:
            Class ID (0: light_blue, 1: dark_blue, 2: other).
        """
        h, w = image.shape[:2]
        
        # Convert YOLO format to pixel coordinates
        x_center, y_center, box_w, box_h = bbox
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)
        
        # Clip to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 2  # Default to "other"

        # --- CENTER CROP (Critical for Green Background) ---
        # We crop the center 50% of the ROI to avoid the green background board
        # which often confuses the color histogram.
        roi_h, roi_w = roi.shape[:2]
        cx, cy = roi_w // 2, roi_h // 2
        crop_w, crop_h = int(roi_w * 0.5), int(roi_h * 0.5)
        
        # Ensure crop is valid
        if crop_w > 0 and crop_h > 0:
            roi = roi[cy - crop_h//2 : cy + crop_h//2, cx - crop_w//2 : cx + crop_w//2]
        
        if roi.size == 0:
            return 2 # Fallback if crop failed
        # ---------------------------------
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV values
        mean_h, mean_s, mean_v = cv2.mean(hsv)[:3]
        
        # Check light blue
        lb = self.config.color_classification.light_blue
        if (lb['hue_min'] <= mean_h <= lb['hue_max'] and
            lb['sat_min'] <= mean_s <= lb['sat_max'] and
            lb['val_min'] <= mean_v <= lb['val_max']):
            return 0
        
        # Check dark blue
        db = self.config.color_classification.dark_blue
        if (db['hue_min'] <= mean_h <= db['hue_max'] and
            db['sat_min'] <= mean_s <= db['sat_max'] and
            db['val_min'] <= mean_v <= db['val_max']):
            return 1
        
        return 2  # Other


class DataPreprocessor:
    """Preprocess dataset with color-based relabeling and splitting."""
    
    def __init__(self, config):
        """
        Initialize data preprocessor.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.classifier = ColorClassifier(config)
    
    def process_dataset(self) -> Dict[str, Any]:
        """
        Process dataset: relabel colors and split into train/val folders.
        
        Returns:
            Dictionary with processing statistics.
        """
        root_path = Path(self.config.dataset.root_path)
        source_images_path = Path(self.config.dataset.images_path)
        source_labels_path = Path(self.config.dataset.labels_path)
        
        # Define target structure
        splits = ['train', 'val']
        dirs = {
            'images': {split: root_path / 'images' / split for split in splits},
            'labels': {split: root_path / 'labels' / split for split in splits}
        }
        
        # Create directories
        for split in splits:
            dirs['images'][split].mkdir(parents=True, exist_ok=True)
            dirs['labels'][split].mkdir(parents=True, exist_ok=True)
            
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {0: 0, 1: 0, 2: 0},
            'split_counts': {'train': 0, 'val': 0}
        }
        
        # Get all images (exclude those already in train/val subfolders to prevent recursion)
        image_files = [
            f for f in list(source_images_path.glob("*.jpg")) + list(source_images_path.glob("*.png"))
            if f.parent.name not in splits
        ]
        
        # Shuffle for random split
        random.seed(42) # Deterministic split
        random.shuffle(image_files)
        
        # Calculate split index
        split_idx = int(len(image_files) * self.config.dataset.train_split)
        
        for i, img_path in tqdm(enumerate(image_files), total=len(image_files), desc="Processing & Splitting"):
            label_path = source_labels_path / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Determine split
            split = 'train' if i < split_idx else 'val'
            stats['split_counts'][split] += 1
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Load annotations
            with open(label_path, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines()]
            
            # Relabel based on color
            relabeled_annotations = []
            for ann in annotations:
                if not ann: continue
                class_id, x, y, w, h = map(float, ann)
                new_class_id = self.classifier.classify_color(image, [x, y, w, h])
                
                relabeled_annotations.append(f"{new_class_id} {x} {y} {w} {h}")
                stats['class_distribution'][new_class_id] += 1
                stats['total_annotations'] += 1
            
            # 1. Write new label to target folder
            target_label_path = dirs['labels'][split] / f"{img_path.stem}.txt"
            with open(target_label_path, 'w') as f:
                f.write('\n'.join(relabeled_annotations))
            
            # 2. Copy image to target folder
            target_image_path = dirs['images'][split] / img_path.name
            shutil.copy2(img_path, target_image_path)
            
            stats['total_images'] += 1
            
        return stats