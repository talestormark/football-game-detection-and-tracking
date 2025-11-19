#!/bin/bash
#SBATCH --job-name=football_tracking
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --output=/cluster/work/tmstorma/Football2025/tracking/logs/%j_tracking_output.txt
#SBATCH --error=/cluster/work/tmstorma/Football2025/tracking/logs/%j_tracking_error.err

# Configuration
# - 8 CPUs: Sufficient for DataLoader workers (max 8 recommended)
# - 32GB RAM: Enough for loading images and tracking state
# - 1 GPU: For YOLOv8 inference
# - 2 hours: Conservative estimate (tracking is faster than training)

# Create logs directory if it doesn't exist
mkdir -p /cluster/work/tmstorma/Football2025/tracking/logs

# Load required modules
module purge
module load Miniconda3/24.7.1-0

# Activate conda environment
source activate football_analysis

# Change to tracking directory
cd /cluster/work/tmstorma/Football2025/tracking

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Print GPU information
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# Run tracking validation
echo "=========================================="
echo "Running Validation Tracking"
echo "=========================================="
python run_tracking_validation.py

# Print completion time
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
