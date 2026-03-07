#!/bin/bash
#SBATCH --job-name=cpctlease
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cpctlease_%j.log

module load python/3.11 scipy-stack

python -m venv ~/venvs/cpctlease || true
source ~/venvs/cpctlease/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

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
