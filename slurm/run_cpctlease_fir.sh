#!/bin/bash
#SBATCH --job-name=cpctlease
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cpctlease_%j.log

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 scipy-stack/2024a

REPO_DIR="$HOME/scratch/cpctlease_code_package"
cd "$REPO_DIR"

source "$HOME/venvs/cpctlease/bin/activate"

export MPLCONFIGDIR=${SLURM_TMPDIR:-$PWD/.mpl-cache}
mkdir -p "$MPLCONFIGDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

cpctlease generate-all \
  --config configs/paper_literal.json \
  --outdir outputs/paper_literal \
  --workers ${SLURM_CPUS_PER_TASK} \
  --pin-blas-threads
