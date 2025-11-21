Bottle Cap Detection SystemA real-time computer vision system to detect and classify bottle caps (Light Blue, Dark Blue, Other) optimized for edge devices like the Raspberry Pi 5.📋 Table of ContentsQuick StartDetailed SetupDataset PreparationTrainingInferenceEvaluationDeploymentCI/CDTroubleshooting🚀 Quick StartWindows Terminal (PowerShell / Command Prompt)Clone & Navigate:git clone <your-repo-url>
cd bottlecap-detection
Create & Activate Virtual Environment:# PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
# Note: If you get a permission error, run: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Command Prompt (cmd)
python -m venv venv
venv\Scripts\activate.bat
Install Dependencies:Note: Quotes are required around .[dev] in PowerShell.pip install -e ".[dev]"
Run the Pipeline:# Preprocess data
bsort preprocess --config configs/settings.yaml

# Train model
bsort train --config configs/settings.yaml

# Run inference (Make sure to point to a valid image file)
# Example: Picking a random validation image
bsort infer --config configs/settings.yaml --image data\images\val --save --show
Linux / MacOS# 1. Clone repository
git clone <your-repo-url>
cd bottlecap-detection

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Preprocess dataset
bsort preprocess --config configs/settings.yaml

# 4. Train model
bsort train --config configs/settings.yaml

# 5. Run inference
bsort infer --config configs/settings.yaml --image data/images/val/sample.jpg --show
🔧 Detailed Setup1. Environment SetupOption A: Using pip (Recommended for development)# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode
pip install -e ".[dev]"

# Verify installation
bsort --help
Option B: Using Docker (Recommended for deployment)# Build Docker image
docker build -t bottlecap-detection:latest .

# Run training inside Docker
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/outputs:/app/outputs \
           --gpus all \
           bottlecap-detection:latest \
           bsort train --config configs/settings.yaml

# Run inference inside Docker
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/outputs:/app/outputs \
           bottlecap-detection:latest \
           bsort infer --config configs/settings.yaml --image data/test.jpg
2. WandB Setup (Optional but Recommended)This project uses Weights & Biases for experiment tracking.wandb login
Update configs/settings.yaml with your entity if needed:wandb:
  enabled: true
  project: "bottlecap-detection"
  entity: "your-username"
📊 Dataset Preparation1. Download & StructureExtract your dataset into the data/ directory.data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
2. Preprocess Dataset (Crucial Step)The raw dataset usually comes with generic labels. We must run the color-based relabeling logic (using HSV histograms and Center Crop) to assign the correct classes (0: Light Blue, 1: Dark Blue, 2: Other).bsort preprocess --config configs/settings.yaml
This generates data/labels_relabeled/ and organizes images into train/ and val/ folders.3. Train/Val SplitEnsure your data.yaml (generated automatically during training) or manual setup points to the correct split structure.🎯 TrainingBasic Trainingbsort train --config configs/settings.yaml
Advanced Options# Resume from checkpoint
bsort train --config configs/settings.yaml \
            --resume runs/train/exp1/weights/last.pt

# Debug mode (verbose logging)
bsort train --config configs/settings.yaml --debug
ConfigurationEdit configs/settings.yaml to tune hyperparameters:training:
  epochs: 100
  batch_size: 16        # Reduce if OOM
  learning_rate: 0.01
  device: "cuda:0"      # or "cpu"
  patience: 20          # Early stopping
MonitoringResults are saved to runs/train/exp/ and logged to WandB.🔍 InferenceSingle Imagebsort infer --config configs/settings.yaml \
            --image data/images/val/test_image.jpg \
            --save \
            --show
Entire Folderbsort infer --config configs/settings.yaml \
            --image data/images/val \
            --save
Videobsort infer --config configs/settings.yaml \
            --video data/test.mp4 \
            --save
Webcam (Real-time)bsort infer --config configs/settings.yaml \
            --source 0
Custom Weightsbsort infer --config configs/settings.yaml \
            --image data/test.jpg \
            --weights runs/train/exp/weights/best.pt
📈 Evaluation# Validate on validation set
bsort eval --config configs/settings.yaml --dataset val

# Validate on test set
bsort eval --config configs/settings.yaml --dataset test
Metrics Explained:mAP@50: Mean Average Precision at IoU=0.50 (Standard metric)mAP@50-95: Strict metric averaging IoU from 0.50 to 0.95Precision: Accuracy of positive predictionsRecall: Ability to find all positive instances🚢 Deployment1. Export ModelFor edge devices, you must export the PyTorch model to an optimized format.ONNX (Recommended for Raspberry Pi)bsort export --config configs/settings.yaml \
             --weights best.pt \
             --format onnx \
             --simplify
TensorRT (NVIDIA Jetson)bsort export --config configs/settings.yaml \
             --weights best.pt \
             --format tensorrt
2. Raspberry Pi 5 DeploymentInstall Runtime on Pisudo apt-get update && sudo apt-get install python3-opencv python3-numpy
pip install onnxruntime
Run Inference on Pifrom bsort.inference.predictor import Predictor
# The predictor automatically detects .onnx extension and switches backend
predictor = Predictor(config, weights_path='best.onnx')
results = predictor.predict('test.jpg')
🔄 CI/CDThis project uses GitHub Actions (.github/workflows/ci.yml) to ensure code quality.Pipeline Steps:Code Quality: Black, Isort, Pylint (Score ≥ 8.0)Testing: Pytest with coverageBuild: Docker image creation testRun Locally:# Formatting check
black --check bsort/
isort --check-only bsort/

# Linting
pylint bsort/

# Unit Tests
pytest tests/ -v --cov=bsort
🐛 TroubleshootingIssueSolutionCUDA Out of MemoryReduce batch_size in settings.yaml (e.g., to 8 or 4).Slow TrainingIncrease workers if CPU allows. Reduce input_size to 416.Model OverfittingIncrease weight_decay (0.001), enable mosaic augmentation, or collect more data.Inference Too SlowCrucial: Use ONNX format with INT8 quantization on Edge devices. Check libgomp installation.Wrong Colors DetectedVerify HSV thresholds in settings.yaml are in OpenCV scale (H: 0-179, S/V: 0-255)."File Not Found" (Windows)Ensure you are using backslashes \ or quoted paths. Use specific image files, not example placeholders.📝 Best PracticesExperiment Tracking: Always use WandB to log runs.Model Versioning: Tag production models in Git (e.g., v1.0).Data Integrity: Do not manually edit labels_relabeled. Modify the ColorClassifier logic and re-run preprocessing instead.Edge Optimization: Never deploy .pt files to production. Always export to ONNX/TensorRT.📄 LicenseMIT License