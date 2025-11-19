#!/bin/bash
#SBATCH --job-name=metrics_viz
#SBATCH --output=/cluster/work/tmstorma/Football2025/tracking/logs/%j_metrics_output.txt
#SBATCH --error=/cluster/work/tmstorma/Football2025/tracking/logs/%j_metrics_error.err
#SBATCH --partition=CPUQ
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load Python environment
module purge
module load Anaconda3/2024.02-1

# Activate conda environment
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
conda activate football_analysis

# Run metrics visualization script
cd /cluster/work/tmstorma/Football2025/tracking
python create_metrics_overlay.py
