"""
Command-line interface for bottle cap detection system.

This module provides CLI commands for training, inference, and evaluation
of the bottle cap detection model.
"""

import click
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional

# We import these lazily or normally depending on CLI latency requirements.
from bsort.config import Config
from bsort.training.trainer import Trainer
from bsort.inference.predictor import Predictor
from bsort.training.validator import Validator
from bsort.data.preprocessing import DataPreprocessor


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Bottle Cap Detection System - CLI Interface."""
    pass


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Path to checkpoint to resume training",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode",
)
def train(config: str, resume: Optional[str], debug: bool) -> None:
    """
    Train the bottle cap detection model.

    Args:
        config: Path to configuration YAML file.
        resume: Path to checkpoint to resume training from.
        debug: Enable debug mode for verbose logging.

    Example:
        bsort train --config configs/settings.yaml
        bsort train --config configs/settings.yaml --resume runs/exp1/weights/last.pt
    """
    click.echo("🚀 Starting training pipeline...")
    
    # Load configuration
    cfg = Config.from_yaml(config)
    
    if debug:
        click.echo(f"Configuration: {cfg}")
    
    # Preprocess data (color-based relabeling)
    click.echo("📊 Checking/Preprocessing dataset...")
    preprocessor = DataPreprocessor(cfg)
    preprocessor.process_dataset()
    
    # Initialize trainer
    trainer = Trainer(cfg, resume_from=resume)
    
    # Start training
    click.echo("🎯 Training model...")
    results = trainer.train()
    
    click.echo(f"✅ Training complete! Results saved to {results.get('save_dir', 'output directory')}")
    if 'metrics/mAP50(B)' in results:
        click.echo(f"📈 Best mAP@50: {results['metrics/mAP50(B)']:.4f}")


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--image",
    type=click.Path(exists=True),
    default=None,
    help="Path to input image (or directory)",
)
@click.option(
    "--video",
    type=click.Path(exists=True),
    default=None,
    help="Path to input video",
)
@click.option(
    "--source",
    type=str,
    default=None,
    help="Source for inference (image path, video path, or camera index)",
)
@click.option(
    "--weights",
    type=click.Path(exists=True),
    default=None,
    help="Path to model weights",
)
@click.option(
    "--save",
    is_flag=True,
    default=True,
    help="Save inference results",
)
@click.option(
    "--show",
    is_flag=True,
    help="Display inference results",
)
def infer(
    config: str,
    image: Optional[str],
    video: Optional[str],
    source: Optional[str],
    weights: Optional[str],
    save: bool,
    show: bool,
) -> None:
    """
    Run inference on images or videos.

    Args:
        config: Path to configuration YAML file.
        image: Path to input image.
        video: Path to input video.
        source: General source path (image, video, or camera).
        weights: Path to model weights.
        save: Save inference results.
        show: Display inference results.

    Example:
        bsort infer --config configs/settings.yaml --image sample.jpg
        bsort infer --config configs/settings.yaml --video sample.mp4 --show
        bsort infer --config configs/settings.yaml --source 0  # webcam
    """
    click.echo("🔍 Starting inference...")
    
    # Load configuration
    cfg = Config.from_yaml(config)
    
    # Determine source
    inference_source = source or image or video
    if not inference_source:
        click.echo("❌ Error: No input source specified. Use --image, --video, or --source")
        return
    
    # Initialize predictor
    predictor = Predictor(cfg, weights_path=weights)
    
    # Run inference
    click.echo(f"📸 Processing: {inference_source}")
    results = predictor.predict(
        source=inference_source,
        save=save,
        show=show,
    )
    
    detection_count = len(results) if results else 0
    click.echo(f"✅ Inference complete! Processed {detection_count} items.")
    
    if save and cfg.inference.output_dir:
        output_dir = Path(cfg.inference.output_dir)
        click.echo(f"💾 Results saved to: {output_dir}")


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--weights",
    type=click.Path(exists=True),
    default=None,
    help="Path to model weights",
)
@click.option(
    "--dataset",
    type=click.Choice(["train", "val", "test"]),
    default="val",
    help="Dataset split to evaluate",
)
def eval(config: str, weights: Optional[str], dataset: str) -> None:
    """
    Evaluate model on validation/test dataset.

    Args:
        config: Path to configuration YAML file.
        weights: Path to model weights.
        dataset: Dataset split to evaluate (train, val, or test).

    Example:
        bsort eval --config configs/settings.yaml --dataset val
        bsort eval --config configs/settings.yaml --weights best.pt --dataset test
    """
    click.echo(f"📊 Evaluating model on {dataset} dataset...")
    
    # Load configuration
    cfg = Config.from_yaml(config)
    
    # Initialize validator
    validator = Validator(cfg, weights_path=weights)
    
    # Run evaluation
    metrics = validator.validate(dataset_split=dataset)
    
    # Display results
    click.echo("\n" + "=" * 50)
    click.echo("Evaluation Results:")
    click.echo("=" * 50)
    click.echo(f"mAP@50:       {metrics.get('metrics/mAP50(B)', 0):.4f}")
    click.echo(f"mAP@50-95:    {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    click.echo(f"Precision:    {metrics.get('metrics/precision(B)', 0):.4f}")
    click.echo(f"Recall:       {metrics.get('metrics/recall(B)', 0):.4f}")
    click.echo("=" * 50)
    
    # Per-class metrics
    if "per_class" in metrics:
        click.echo("\nPer-Class Metrics:")
        for class_name, class_metrics in metrics["per_class"].items():
            click.echo(f"\n{class_name}:")
            click.echo(f"  AP@50: {class_metrics.get('ap50', 0):.4f}")
            click.echo(f"  Precision: {class_metrics.get('precision', 0):.4f}")
            click.echo(f"  Recall: {class_metrics.get('recall', 0):.4f}")


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--weights",
    type=click.Path(exists=True),
    required=True,
    help="Path to model weights",
)
@click.option(
    "--format",
    type=click.Choice(["onnx", "tensorrt", "torchscript"]),
    default="onnx",
    help="Export format",
)
@click.option(
    "--simplify",
    is_flag=True,
    help="Simplify ONNX model",
)
@click.option(
    "--int8",
    is_flag=True,
    help="Enable INT8 quantization (Recommended for RPi)",
)
def export(config: str, weights: str, format: str, simplify: bool, int8: bool) -> None:
    """
    Export model to different formats for deployment.

    Args:
        config: Path to configuration YAML file.
        weights: Path to model weights.
        format: Export format (onnx, tensorrt, torchscript).
        simplify: Simplify ONNX model (removes redundant operators).
        int8: Enable INT8 quantization.

    Example:
        bsort export --config configs/settings.yaml --weights best.pt --format onnx --simplify
        bsort export --config configs/settings.yaml --weights best.pt --format onnx --int8
    """
    click.echo(f"📦 Exporting model to {format} format (INT8: {int8})...")
    
    # Lazy import
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(weights)
    
    # Prepare export arguments
    export_args = {"format": format}
    if format == "onnx":
        export_args["simplify"] = simplify
        
    if int8:
        export_args["int8"] = True
        export_args["data"] = "data.yaml" # Required for calibration
        click.echo("⚠️  INT8 quantization requires 'data.yaml' for calibration.")
        
    # Export
    export_path = model.export(**export_args)
    
    click.echo(f"✅ Model exported successfully to: {export_path}")
    
    # Estimate file size
    try:
        if export_path:
            size_mb = Path(export_path).stat().st_size / (1024 * 1024)
            click.echo(f"📦 Model size: {size_mb:.2f} MB")
    except (OSError, TypeError):
        click.echo("📦 Model size: Unknown")


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
def preprocess(config: str) -> None:
    """
    Preprocess dataset with color-based relabeling.

    Args:
        config: Path to configuration YAML file.

    Example:
        bsort preprocess --config configs/settings.yaml
    """
    click.echo("🔄 Preprocessing dataset...")
    
    # Load configuration
    cfg = Config.from_yaml(config)
    
    # Run preprocessing
    preprocessor = DataPreprocessor(cfg)
    stats = preprocessor.process_dataset()
    
    click.echo("\n" + "=" * 50)
    click.echo("Preprocessing Complete!")
    click.echo("=" * 50)
    click.echo(f"Total images processed: {stats.get('total_images', 0)}")
    click.echo(f"Total annotations: {stats.get('total_annotations', 0)}")
    click.echo("\nClass Distribution:")
    if 'class_distribution' in stats:
        for class_name, count in stats['class_distribution'].items():
            total = stats.get('total_annotations', 1)
            if total == 0: total = 1
            percentage = (count / total) * 100
            click.echo(f"  {class_name}: {count} ({percentage:.1f}%)")
    click.echo("=" * 50)


@main.command()
@click.option(
    "--weights",
    type=click.Path(exists=True),
    required=True,
    help="Path to model weights",
)
@click.option(
    "--device",
    default="cpu",
    help="Device to run benchmark on (cpu or cuda:0)",
)
@click.option(
    "--imgsz",
    default=640,
    help="Inference image size",
)
@click.option(
    "--iterations",
    default=100,
    help="Number of iterations for averaging",
)
def benchmark(weights: str, device: str, imgsz: int, iterations: int) -> None:
    """
    Benchmark inference speed on the current device.
    
    Args:
        weights: Path to model weights (.pt or .onnx).
        device: Device to run on (cpu, cuda).
        imgsz: Image size for inference.
        iterations: Number of iterations to run.
        
    Example:
        bsort benchmark --weights best.onnx --device cpu
    """
    click.echo(f"⏱️  Benchmarking {weights} on {device}...")
    
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(weights)
    
    # Generate dummy input
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup
    click.echo("🔥 Warming up...")
    for _ in range(10):
        model.predict(dummy_img, verbose=False, device=device)
        
    # Benchmark
    click.echo(f"🚀 Running {iterations} iterations...")
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        model.predict(dummy_img, verbose=False, device=device)
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # ms
        
    avg_time = np.mean(times)
    fps = 1000 / avg_time
    
    click.echo("\n" + "=" * 50)
    click.echo("Benchmark Results:")
    click.echo("=" * 50)
    click.echo(f"Model:      {weights}")
    click.echo(f"Device:     {device}")
    click.echo(f"Avg Time:   {avg_time:.2f} ms per frame")
    click.echo(f"FPS:        {fps:.2f}")
    click.echo("=" * 50)
    
    if avg_time <= 10:
        click.echo("✅ Success: Meets the <10ms requirement!")
    else:
        click.echo("⚠️  Warning: Slower than 10ms target. Consider INT8 quantization.")


if __name__ == "__main__":
    main()