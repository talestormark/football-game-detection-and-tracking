#!/bin/bash
#SBATCH --job-name=hota_eval
#SBATCH --account=ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=CPUQ
#SBATCH --output=/cluster/work/tmstorma/Football2025/tracking/logs/%j_hota_output.txt
#SBATCH --error=/cluster/work/tmstorma/Football2025/tracking/logs/%j_hota_error.err

cd /cluster/work/tmstorma/Football2025/tracking

module purge
module load Miniconda3/24.7.1-0

source activate football_analysis

echo "=========================================="
echo "HOTA Evaluation Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo ""

# Step 1: Install dependencies and TrackEval if not present
echo "Installing dependencies..."
pip install scipy -q
echo "Dependencies installed"
echo ""

if [ ! -d "TrackEval" ]; then
    echo "Installing TrackEval..."
    git clone https://github.com/JonathonLuiten/TrackEval.git
    cd TrackEval
    pip install -e .
    cd ..
    echo "TrackEval installed"
    echo ""
fi

# Step 2: Prepare data
echo "=========================================="
echo "Step 1: Preparing HOTA Data"
echo "=========================================="
python prepare_hota_data.py
echo ""

# Step 3: Run evaluation
echo "=========================================="
echo "Step 2: Running HOTA Evaluation"
echo "=========================================="
python run_hota_evaluation.py

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
