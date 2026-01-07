# Bottle Cap Detection System (BSORT)

A high-performance computer vision system for real-time bottle cap detection and color classification, optimized for deployment on edge devices including Raspberry Pi 5. The system leverages YOLOv8 Nano architecture with color-based classification to distinguish between light blue, dark blue, and other colored bottle caps with inference speeds under 10 milliseconds per frame.

## Overview

This project addresses the challenge of automated bottle cap sorting in industrial environments where speed and accuracy are paramount. The system combines object detection with intelligent color classification, preprocessing raw annotations to automatically categorize bottle caps based on their HSV color properties. The architecture is specifically designed to meet stringent latency requirements for edge deployment while maintaining high detection accuracy.

## Key Features

The system provides comprehensive functionality spanning the entire machine learning workflow from data preprocessing through deployment. The color-based relabeling pipeline automatically processes datasets by analyzing the HSV color space within detected regions, applying configurable thresholds to classify caps into distinct categories. This approach eliminates manual annotation errors and ensures consistent labeling across the dataset.

Training capabilities include integration with Weights & Biases for experiment tracking, support for data augmentation strategies tailored to small object detection, and automated train-validation splitting with deterministic seeding. The system supports resuming training from checkpoints and provides comprehensive evaluation metrics including per-class average precision scores.

For deployment scenarios, the system offers ONNX export with optional INT8 quantization to achieve the target inference latency of under 10 milliseconds per frame. The inference pipeline supports multiple input sources including images, videos, and real-time camera feeds. Docker containerization ensures consistent deployment across different environments, while the command-line interface provides intuitive access to all system capabilities.

## Technical Architecture

The system architecture follows a modular design pattern with clear separation of concerns. The data preprocessing module implements a center-crop strategy to isolate bottle cap regions from background elements, particularly addressing challenges posed by green sorting boards that can confuse color classification algorithms. This preprocessing step calculates mean HSV values within the cropped region and applies configurable thresholds to determine color categories.

The model component utilizes YOLOv8 Nano, selected specifically for its favorable balance between detection accuracy and inference speed on resource-constrained devices. The training pipeline incorporates cosine annealing learning rate scheduling, SGD optimization with momentum, and early stopping based on validation performance to prevent overfitting.

Configuration management employs a hierarchical YAML-based system that separates dataset parameters, color classification thresholds, model hyperparameters, training settings, and inference configurations. This design enables rapid experimentation and deployment configuration changes without code modifications.

## Installation

The system requires Python 3.10 or higher and can be installed through multiple approaches depending on your deployment scenario. For development environments, clone the repository and install the package in editable mode to enable code modifications and testing.

```bash
git clone https://github.com/yourusername/bottlecap-detection.git
cd bottlecap-detection
pip install -e .
```

For production deployment, install the package directly without development dependencies.

```bash
pip install .
```

To include development tools for testing and code quality checks, install with the optional development dependencies.

```bash
pip install -e ".[dev]"
```

### Docker Deployment

The project includes a production-ready Dockerfile that packages the system with all dependencies and runtime requirements. Build the Docker image using the following command.

```bash
docker build -t bsort:latest .
```

Run the containerized system with volume mounts for data and output directories.

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs bsort:latest infer --config configs/settings.yaml --image data/sample.jpg
```

## Configuration

System behavior is controlled through the `configs/settings.yaml` file, which defines all operational parameters. The dataset configuration section specifies paths to images and labels, along with the train-validation split ratio. Class names must align with the preprocessing logic where index 0 represents light blue caps, index 1 represents dark blue caps, and index 2 represents other colors.

Color classification thresholds utilize OpenCV's HSV color space where hue ranges from 0 to 179, while saturation and value range from 0 to 255. These thresholds require careful tuning based on your specific bottle cap colors and lighting conditions. The provided default values target typical cyan and deep blue bottle caps but should be adjusted through experimentation with your dataset.

Model configuration parameters control the YOLOv8 variant selection, with YOLOv8 Nano mandated for edge deployment scenarios. Input size defaults to 640 pixels, providing a good balance between detection accuracy and processing speed. Training configuration encompasses epoch count, batch size, learning rate, and data augmentation parameters. The augmentation settings are specifically tuned for small object detection, with mixup disabled to prevent label confusion.

Inference parameters define confidence thresholds for detection filtering and non-maximum suppression thresholds for duplicate removal. These values significantly impact the precision-recall tradeoff and should be adjusted based on your application requirements.

## How to Run

This section provides step-by-step instructions for the most common workflows, from initial setup through production deployment. Each workflow assumes you have completed the installation steps and have your dataset prepared in YOLO format.

### Quick Start: Complete Pipeline

For users who want to run the entire pipeline from preprocessing through inference, follow these steps in sequence. This workflow represents the standard approach for training a new model on your dataset.

First, ensure your dataset is organized with images in the `data/images` directory and corresponding label files in the `data/labels` directory. Each label file should use YOLO format with one line per object. Initial class labels can be arbitrary as the preprocessing step will reclassify based on color analysis.

Next, review and adjust the configuration file at `configs/settings.yaml` to match your dataset paths and color thresholds. Pay particular attention to the HSV threshold values in the color classification section, as these directly impact classification accuracy. If you are unsure about appropriate values, start with the provided defaults and iterate based on preprocessing results.

Execute the preprocessing command to analyze your dataset and apply color-based relabeling. This step creates the organized train-validation split structure required for training.

```bash
bsort preprocess --config configs/settings.yaml
```

The preprocessing output will display statistics about your dataset including total images processed, annotation counts, and the distribution across the three color classes. Review these statistics to ensure the color classification is performing as expected. If the distribution appears skewed or unexpected, revisit your HSV threshold settings.

With preprocessing complete, initiate model training using the prepared dataset. The training process will run for the configured number of epochs, automatically saving checkpoints and tracking validation metrics.

```bash
bsort train --config configs/settings.yaml
```

Training progress appears in the console with regular updates on loss values and validation metrics. The system saves the best performing model based on validation mAP@50 scores to `runs/train/exp/weights/best.pt`. Training duration varies based on dataset size and hardware capabilities, typically requiring several hours on GPU hardware for datasets with thousands of images.

After training completes, evaluate the model performance on the validation dataset to assess accuracy and identify potential improvements.

```bash
bsort eval --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --dataset val
```

The evaluation output provides comprehensive metrics including overall mAP scores, precision, recall, and per-class performance breakdowns. Use these metrics to determine whether additional training, data collection, or threshold adjustments are necessary.

Once satisfied with model performance, run inference on new images or videos to verify real-world performance.

```bash
bsort infer --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --image path/to/test_image.jpg --show
```

The system displays annotated results with bounding boxes and class labels for each detected bottle cap. Review these visual results to ensure the model generalizes well to unseen data.

### Workflow: Training from Scratch

When starting with a new dataset or experimenting with different configurations, this workflow provides detailed steps for the training process. This approach assumes you have already preprocessed your data or are working with a dataset that has been organized into the required structure.

Begin by verifying your data organization. The training process expects to find images in `data/images/train` and `data/images/val` directories, with corresponding labels in `data/labels/train` and `data/labels/val`. If your dataset lacks this structure, run the preprocessing command first.

Review the training configuration section in `configs/settings.yaml`. Key parameters to consider include the number of epochs, batch size, and learning rate. For initial experiments, the default values provide a reasonable starting point. Larger datasets may benefit from increased epochs, while smaller datasets might require reduced learning rates to prevent overfitting.

If you have limited GPU memory, reduce the batch size accordingly. The YOLOv8 Nano architecture is designed to be memory efficient, but batch sizes of 32 or higher may exceed available memory on consumer GPUs. A batch size of 16 typically works well across different hardware configurations.

Enable Weights & Biases tracking if you want to monitor training progress remotely or compare multiple experiments. Update the wandb section in your configuration with your project name and entity. If you prefer not to use external tracking, set the enabled flag to false.

Start the training process with your configured parameters.

```bash
bsort train --config configs/settings.yaml
```

The training process begins with several warmup epochs using a reduced learning rate, then transitions to the full learning rate with cosine annealing decay. Monitor the console output for loss values that should generally decrease over time. Validation metrics are computed at regular intervals, providing early feedback on model performance.

If training is interrupted for any reason, resume from the last saved checkpoint to avoid losing progress.

```bash
bsort train --config configs/settings.yaml --resume runs/train/exp/weights/last.pt
```

The resume functionality loads the model weights, optimizer state, and epoch counter from the checkpoint, continuing training seamlessly from where it stopped.

Once training completes, compare the performance of different checkpoints by evaluating both the final model and the best model on validation data. Sometimes the final model may overfit slightly compared to the best checkpoint saved during training.

```bash
bsort eval --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --dataset val
bsort eval --config configs/settings.yaml --weights runs/train/exp/weights/last.pt --dataset val
```

### Workflow: Inference on New Data

When you have a trained model and need to process new images or videos, this workflow demonstrates the various inference options available. The inference pipeline supports batch processing of images, video analysis, and real-time camera feeds.

For single image inference with visual display of results, use the following command. This approach is ideal for quick validation and debugging.

```bash
bsort infer --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --image path/to/image.jpg --show
```

The system loads the image, applies the model with configured confidence and IOU thresholds, and displays the annotated result in a window. Press any key to close the window and proceed.

To process an entire directory of images, specify the directory path instead of a single file. The system iterates through all images in the directory, applying detection to each one.

```bash
bsort infer --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --image path/to/image_directory/ --save
```

Results are saved to the output directory specified in your configuration, typically `outputs/inference/predict`. Each output image includes bounding boxes, class labels, and confidence scores for detected bottle caps.

For video analysis, provide a path to the video file. The inference pipeline processes each frame sequentially, applying detection and optionally saving the annotated video.

```bash
bsort infer --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --video path/to/video.mp4 --save
```

The annotated video is saved to the output directory, maintaining the original frame rate and resolution. This functionality is useful for analyzing recorded sorting operations or demonstration purposes.

To perform real-time inference using a connected camera, specify the camera index as the source. Camera index 0 typically refers to the default system camera.

```bash
bsort infer --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --source 0 --show
```

The system opens a live view window displaying the camera feed with real-time detection overlays. This mode is particularly useful for testing deployment scenarios and verifying performance under actual operating conditions. Press 'q' to exit the live inference mode.

If inference results show too many false positives, increase the confidence threshold in your configuration file. Conversely, if the model misses valid detections, try lowering the threshold. The IOU threshold controls how aggressively duplicate detections are suppressed, with higher values being more permissive of overlapping boxes.

### Workflow: Edge Deployment Optimization

When preparing models for deployment on resource-constrained devices like Raspberry Pi 5, optimization through model export and quantization is essential. This workflow guides you through creating deployment-ready model files.

Start by exporting your trained model to ONNX format, which provides broad compatibility across different inference frameworks and platforms.

```bash
bsort export --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --format onnx --simplify
```

The simplify flag removes redundant operators from the ONNX graph, reducing model complexity and improving inference speed. The resulting ONNX file is saved in the same directory as the original weights file.

For maximum performance on edge devices, apply INT8 quantization during export. This optimization reduces model size by approximately 75 percent and significantly accelerates inference, particularly on devices with limited computational resources.

```bash
bsort export --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --format onnx --int8
```

INT8 quantization requires calibration data to determine appropriate quantization parameters. Ensure your `data.yaml` file points to representative training images. The export process analyzes these images to compute optimal quantization scales for each layer.

After exporting the optimized model, verify its performance using the benchmark tool. This measurement confirms whether the optimized model meets your latency requirements.

```bash
bsort benchmark --weights runs/train/exp/weights/best.onnx --device cpu --iterations 100
```

The benchmark runs 100 inference iterations with synthetic input data, reporting average processing time per frame and frames per second throughput. For Raspberry Pi 5 deployment, target average times below 10 milliseconds to ensure real-time performance at standard video frame rates.

If benchmark results exceed the latency target, consider additional optimizations such as reducing input resolution in your configuration, further simplifying the model architecture, or ensuring you are using the INT8 quantized version. Input resolution has a significant impact on inference speed, with 640x640 providing a good balance for most applications.

Test the exported model with actual inference to verify that accuracy remains acceptable after optimization.

```bash
bsort infer --config configs/settings.yaml --weights runs/train/exp/weights/best.onnx --image path/to/test_image.jpg --show
```

Compare the detection results from the ONNX model against the original PyTorch model to assess any accuracy degradation. Minor differences in confidence scores are normal due to numerical precision changes, but significant detection misses or false positives may indicate issues with the quantization process.

### Workflow: Model Evaluation and Analysis

Thorough evaluation of model performance is crucial for understanding strengths, weaknesses, and potential areas for improvement. This workflow demonstrates comprehensive evaluation techniques.

Begin with standard validation set evaluation to establish baseline performance metrics.

```bash
bsort eval --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --dataset val
```

The output displays overall metrics including mAP@50, mAP@50-95, precision, and recall. These aggregate metrics provide a high-level view of model performance but may obscure class-specific issues.

Examine the per-class metrics carefully, as they reveal performance disparities between different bottle cap colors. If one class significantly underperforms the others, consider whether that class is underrepresented in your training data or whether the color classification thresholds need adjustment.

For production deployments, evaluate the model on a held-out test set that was not used during training or validation. This provides the most realistic estimate of real-world performance.

```bash
bsort eval --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --dataset test
```

Significant performance drops between validation and test sets indicate potential overfitting or distribution shift issues. In such cases, consider collecting more diverse training data or applying stronger regularization through data augmentation.

If you have access to production data that differs from your training distribution, create a custom evaluation dataset and assess model performance on this realistic data. This evaluation most accurately predicts deployment performance and identifies domain adaptation requirements.

Compare multiple model checkpoints to understand how performance evolves during training and identify the optimal stopping point.

```bash
bsort eval --config configs/settings.yaml --weights runs/train/exp1/weights/best.pt --dataset val
bsort eval --config configs/settings.yaml --weights runs/train/exp2/weights/best.pt --dataset val
bsort eval --config configs/settings.yaml --weights runs/train/exp3/weights/best.pt --dataset val
```

This comparison helps determine whether extended training improves results or if the model has plateaued. It also validates the effectiveness of different hyperparameter configurations.

### Docker Deployment Workflow

For reproducible deployments across different environments, use the Docker containerization approach. This workflow ensures consistent behavior regardless of the host system configuration.

Build the Docker image from the project root directory. The build process installs all dependencies and packages the complete system.

```bash
docker build -t bsort:latest .
```

The build creates a self-contained image with Python 3.10, all required libraries, and the BSORT package. Image size is optimized by using the slim Python base image and cleaning up unnecessary files.

To run preprocessing inside the container, mount your data directory as a volume.

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/configs:/app/configs bsort:latest preprocess --config configs/settings.yaml
```

The volume mounts ensure that the container can access your dataset and write preprocessed results back to the host filesystem. Without these mounts, data would be isolated inside the container and lost when it terminates.

For training inside the container, mount additional directories for output preservation.

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/configs:/app/configs -v $(pwd)/runs:/app/runs bsort:latest train --config configs/settings.yaml
```

The `--gpus all` flag enables GPU access inside the container, significantly accelerating training. Omit this flag if running on CPU-only systems, though training will be considerably slower.

Run inference with the containerized system by mounting the image or video you want to process along with an output directory for results.

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs -v $(pwd)/runs:/app/runs bsort:latest infer --config configs/settings.yaml --weights runs/train/exp/weights/best.pt --image data/test.jpg --save
```

The annotated output images appear in the mounted outputs directory on your host system, accessible immediately after the container completes.

For interactive development inside the container, start a bash shell with the necessary mounts.

```bash
docker run -it -v $(pwd):/app bsort:latest bash
```

This approach provides a consistent development environment that matches production deployment, reducing environment-related bugs and compatibility issues.

## Dataset Requirements

The system expects datasets organized in YOLO format with images in one directory and corresponding label files in another. Each label file should contain one line per object with format `class_id x_center y_center width height` where coordinates are normalized to the range 0 to 1.

Initial annotations can use any class labels as the preprocessing step will reclassify them based on color analysis. However, ensure bounding boxes accurately encompass the entire bottle cap to enable proper color extraction. The preprocessing pipeline creates a structured directory layout with separate train and validation subdirectories for both images and labels.

## Performance Considerations

Achieving the 10 millisecond inference latency target on Raspberry Pi 5 requires careful optimization. The YOLOv8 Nano architecture provides the foundational efficiency, but additional optimizations through ONNX export and INT8 quantization are essential for edge deployment. These optimizations reduce model size by approximately 75 percent while maintaining acceptable accuracy degradation of less than 2 percent mAP.

The color classification preprocessing stage adds minimal overhead as it operates only during dataset preparation rather than inference time. This design decision separates the computationally intensive color analysis from the real-time detection pipeline, ensuring consistent inference performance.

For production deployments, consider implementing batch processing where applicable to amortize model loading overhead across multiple frames. However, be mindful that larger batch sizes increase memory requirements and may not be suitable for resource-constrained devices.

## Testing

The project includes a comprehensive test suite covering configuration loading, data preprocessing, model initialization, and inference pipelines. Run the complete test suite with coverage reporting using pytest.

```bash
pytest tests/ --cov=bsort --cov-report=term-missing
```

Individual test modules can be executed separately for focused development.

```bash
pytest tests/test_data.py -v
```

The test suite validates color classification logic with synthetic images of known colors, ensures configuration parsing handles all required fields correctly, and verifies that the preprocessing pipeline produces expected output structures. While some tests require trained model weights and will skip gracefully if unavailable, the majority of tests operate with mock configurations and synthetic data.

## Project Structure

The codebase follows a modular organization with clear separation between data handling, model management, training logic, and inference capabilities. The `bsort` package contains all core functionality organized into submodules. Configuration management is centralized in `config.py` using dataclasses for type safety and clarity. The CLI interface in `cli.py` provides the primary user interaction point, wrapping complex operations in simple commands.

Data preprocessing logic resides in `bsort/data/preprocessing.py` and implements both color classification and dataset organization. Training functionality in `bsort/training/trainer.py` handles model initialization, training loop execution, and checkpoint management. The inference pipeline in `bsort/inference/predictor.py` supports multiple input sources and output formats.

Testing infrastructure is organized in the `tests/` directory with separate modules for each major component. Configuration examples in `configs/` provide starting points for different deployment scenarios. The Docker configuration enables containerized deployment with minimal setup requirements.

## Troubleshooting

Common issues and their solutions are documented here to facilitate rapid problem resolution. If you encounter problems not covered in this section, consult the project issue tracker or open a new issue with detailed reproduction steps.

If preprocessing reports zero images processed, verify that your dataset paths in the configuration file are correct and that image files have standard extensions like `.jpg` or `.png`. The preprocessing script only processes files in the root of the images directory, not in subdirectories, to avoid processing already split data.

When training fails to start, check that your GPU drivers and CUDA installation are properly configured if using GPU acceleration. The system defaults to GPU device 0, so ensure this device exists on your system or modify the configuration to use CPU or a different GPU index.

If color classification produces unexpected results with most caps classified as "other", review your HSV threshold values. The default values target specific blue shades and may not generalize to all datasets. Use a color picker tool to analyze sample bottle cap images and adjust thresholds accordingly.

For inference performance issues where processing is slower than expected, verify that you are using the ONNX model with INT8 quantization rather than the original PyTorch weights. Additionally, confirm that the inference is running on the intended device and not falling back to a slower computational path.

Memory errors during training typically indicate that the batch size is too large for available GPU memory. Reduce the batch size in your configuration and restart training. The YOLOv8 Nano model should train successfully with batch sizes as low as 8, though larger batch sizes generally provide more stable training dynamics.

## Contributing

Contributions to improve detection accuracy, reduce inference latency, or enhance system capabilities are welcome. When submitting pull requests, ensure all tests pass and code follows the established style guidelines enforced by Black and isort formatters. Add tests for new functionality to maintain code coverage above 80 percent.

For bug reports or feature requests, open an issue with detailed reproduction steps and expected behavior descriptions. Include relevant configuration files and log outputs to facilitate troubleshooting.

## License

This project is released under the MIT License, permitting commercial and non-commercial use with attribution. See the LICENSE file for complete terms and conditions.

## Acknowledgments

This system builds upon the excellent work of the Ultralytics team in developing and maintaining the YOLOv8 architecture. The ONNX Runtime provides critical optimization capabilities for edge deployment scenarios.