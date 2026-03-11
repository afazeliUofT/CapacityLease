#!/bin/bash
#SBATCH --job-name=cpctlease_audit_v2_32
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=cpctlease_audit_v2_32_%j.log

set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 scipy-stack/2024a

REPO_DIR="/home/rsadve1/scratch/cpctlease_code_package"
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

python -m py_compile tools/flexible_solver_audit.py tools/apply_audit_hardening_patch.py

OUTDIR="audit_runs/paper_literal_audit_v2"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

set +e
python tools/flexible_solver_audit.py \
  --repo-root . \
  --config configs/paper_literal.json \
  --golden-dir tests/golden/paper_literal \
  --outdir "$OUTDIR" \
  --baseline-workers 1 \
  --sensitivity-workers 32 \
  --chunksize 4 \
  --pin-blas-threads \
  --fail-on-issues
AUDIT_RC=$?
set -e

mkdir -p "$OUTDIR/review_bundle"

for REL in \
  overall_verdict.json \
  regression/csv_regression_compare.csv \
  regression/summary_regression_compare.csv \
  diagnostics/monopoly_diagnostics.csv \
  sensitivity/sensitivity_summary.csv \
  sensitivity/all_local_probes.csv \
  sensitivity/case_definitions.json \
  environment.json
do
  if [ -f "$OUTDIR/$REL" ]; then
    cp "$OUTDIR/$REL" "$OUTDIR/review_bundle/$(basename "$REL")"
  fi
done

git status --short > "$OUTDIR/review_bundle/git_status.txt" || true
git diff -- tools/flexible_solver_audit.py > "$OUTDIR/review_bundle/flexible_solver_audit.diff" || true
printf 'AUDIT_EXIT_CODE=%s\n' "$AUDIT_RC" > "$OUTDIR/JOB_STATUS.txt"

if [ -f "$OUTDIR/overall_verdict.json" ]; then
  cat "$OUTDIR/overall_verdict.json"
fi

exit "$AUDIT_RC"
