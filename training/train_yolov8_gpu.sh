#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=yolov8_football
#SBATCH --time=0-12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/cluster/work/tmstorma/Football2025/training/logs/%j_train_output.txt
#SBATCH --error=/cluster/work/tmstorma/Football2025/training/logs/%j_train_error.err
#SBATCH --mail-user=tmstorma@stud.ntnu.no
#SBATCH --mail-type=ALL

# YOLOv8 Football Object Detection Training
# 4 classes: home, away, referee, ball
# Expected runtime: 2-4 hours on GPU

echo "=========================================="
echo "YOLOv8 Football Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load conda module
module purge
module load Anaconda3/2023.09-0

echo "Python version:"
python --version
echo ""

echo "Activating conda environment..."
source activate football_analysis
echo ""

# Install/upgrade required packages
echo "Installing/upgrading required packages..."
echo "Installing PyTorch 2.3.0 with CUDA 11.8 (compatible with P100 GPU)..."
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118 --quiet
pip install --upgrade ultralytics --quiet
echo "Package installation complete"
echo ""

# Print resource allocation
echo "Resources:"
echo "  GPU: 1"
echo "  Memory: 32GB"
echo "  CPUs: 8"
echo "  Time limit: 12 hours"
echo ""

# Set working directory
WORKDIR=/cluster/work/tmstorma/Football2025/training
cd ${WORKDIR}
echo "Working directory: ${WORKDIR}"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo ""

# Verify GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Run training script
echo "Starting YOLOv8 training..."
echo "=========================================="
python train_yolov8.py

# Check exit status
exit_code=$?
echo ""
echo "=========================================="
echo "Script exit code: $exit_code"
echo "End time: $(date)"
echo "=========================================="

# If successful, print results location
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Results saved in: /cluster/work/tmstorma/Football2025/training/runs/yolov8s_4class2/"
    echo ""
    echo "Key files:"
    echo "  - weights/best.pt (best model)"
    echo "  - weights/last.pt (last epoch)"
    echo "  - results.png (training curves)"
    echo "  - confusion_matrix.png"
    echo "  - val_batch*.jpg (validation predictions)"
fi

exit $exit_code
