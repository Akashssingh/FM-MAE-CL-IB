#!/bin/bash
# Submit all 4 ablation combinations as separate SLURM jobs.
#
# Usage:
#   cd /home/a/akashsingh/FM-MAE-CL-IB
#   bash Objective_Functions_Ablations/submit_ablations.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # project root

SLURM_SCRIPT="Objective_Functions_Ablations/run_ablation.slurm"

echo "Submitting 4 ablation jobs..."
echo ""

# 1. logcosh + KL  (reference objective function from Arya et al.)
JOB1=$(RUN_NAME=logcosh_kl \
       LOSS_FN=logcosh \
       REGULARIZER=kl \
       KL_WEIGHT=1.0 \
       sbatch --parsable "${SLURM_SCRIPT}")
echo "  [1/4] logcosh + KL    → job ${JOB1}"

# 2. MSE (L2) + KL
JOB2=$(RUN_NAME=mse_kl \
       LOSS_FN=mse \
       REGULARIZER=kl \
       KL_WEIGHT=1.0 \
       sbatch --parsable "${SLURM_SCRIPT}")
echo "  [2/4] MSE (L2) + KL   → job ${JOB2}"

# 3. logcosh + MMD  (WAE-MMD with log-cosh reconstruction)
JOB3=$(RUN_NAME=logcosh_mmd \
       LOSS_FN=logcosh \
       REGULARIZER=mmd \
       MMD_WEIGHT=10.0 \
       sbatch --parsable "${SLURM_SCRIPT}")
echo "  [3/4] logcosh + MMD   → job ${JOB3}"

# 4. MSE (L2) + MMD  (WAE-MMD with L2 reconstruction)
JOB4=$(RUN_NAME=mse_mmd \
       LOSS_FN=mse \
       REGULARIZER=mmd \
       MMD_WEIGHT=10.0 \
       sbatch --parsable "${SLURM_SCRIPT}")
echo "  [4/4] MSE (L2) + MMD  → job ${JOB4}"

echo ""
echo "All jobs submitted: ${JOB1} ${JOB2} ${JOB3} ${JOB4}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  Objective_Functions_Ablations/results/ablation_results.csv"
echo "Logs:     Objective_Functions_Ablations/logs/"
