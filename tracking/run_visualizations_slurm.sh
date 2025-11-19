#!/bin/bash
#SBATCH --job-name=tracking_viz
#SBATCH --output=/cluster/work/tmstorma/Football2025/tracking/logs/%j_viz_output.txt
#SBATCH --error=/cluster/work/tmstorma/Football2025/tracking/logs/%j_viz_error.err
#SBATCH --partition=CPUQ
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=========================================="
echo "Tracking Visualization Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo ""

# Load Python environment
module purge
module load Anaconda3/2024.02-1

# Activate conda environment
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
conda activate football_analysis

# Install OpenCV if needed
echo "Installing dependencies..."
pip install opencv-python -q
echo "Dependencies installed"
echo ""

# Run visualization script
echo "=========================================="
echo "Generating Visualizations"
echo "=========================================="

cd /cluster/work/tmstorma/Football2025/tracking

python create_visualizations.py

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
