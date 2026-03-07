# Capacity-lease simulation package

This package generates the simulation-data tables and plot files for the paper source `CpctLeaseFinal.tex` using organized Python modules, explicit scalar bracketing, analytic Lambert-W profitability boundaries, and process-level parallelism suitable for 32- or 64-core Compute Canada jobs.

## What is included

- `capacity_lease/monopoly.py`: monopoly fixed-point solver and price sweep.
- `capacity_lease/market_clearing.py`: literal market-clearing MVNO model, profitability-set construction, and leased-capacity sweep.
- `capacity_lease/flexible.py`: flexible-participation best-response and MNO outer optimization.
- `capacity_lease/plotting.py`: six plot generators matching the paper's figure layout.
- `configs/paper_literal.json`: literal parameters transcribed from the TeX simulation tables.
- `configs/paper_literal_dense.json`: denser resolution preset.
- `slurm/compute_canada_example.sh`: example SLURM launcher.
- `notes/SOURCE_REVIEW.md`: concise source-review and consistency notes.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Fastest cluster usage

For process parallelism, avoid BLAS/OpenMP oversubscription:

```bash
export MPLCONFIGDIR=${PWD}/.mpl-cache
mkdir -p "$MPLCONFIGDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Then run:

```bash
cpctlease generate-all \
  --config configs/paper_literal.json \
  --outdir outputs/paper_literal \
  --workers 32 \
  --pin-blas-threads
```

For a denser run:

```bash
cpctlease generate-all \
  --config configs/paper_literal_dense.json \
  --outdir outputs/paper_literal_dense \
  --workers 64 \
  --pin-blas-threads
```

For a very fast smoke run that exercises the full pipeline end-to-end:

```bash
cpctlease generate-all \
  --config configs/paper_literal_quick.json \
  --outdir outputs/paper_literal_quick \
  --workers 8 \
  --pin-blas-threads
```

## Outputs

The command writes:

- `data/monopoly_curve.csv`
- `data/market_clearing_capacity_curve.csv`
- `data/flexible_capacity_curve.csv`
- `data/market_clearing_nM_curve.csv`
- `data/flexible_nM_curve.csv`
- `plots/monopoly_n_R.(png|pdf)`
- `plots/monopoly_Ag_r.(png|pdf)`
- `plots/MNO_MVNO_CapacityBlocks.(png|pdf)`
- `plots/Optimal_Prices_vs_Capacity.(png|pdf)`
- `plots/MVNO_MNO_Revenue.(png|pdf)`
- `plots/MVNO_MNO_Prices.(png|pdf)`
- `summary.json`

## Numerical choices

The code is written for stability first:

- Every 1-D root is explicitly bracketed before solving.
- Profitability boundaries are computed from Lambert-W instead of numeric root scans.
- Flexible-participation probabilities use a bivariate-normal tail probability rather than Monte Carlo.
- Global searches over `n_V` and `n_M` use coarse-to-fine refinement instead of local-only solvers.

## Important note about the source paper

The package follows the literal equations and table values in the TeX source. The paper's narrative figure descriptions appear to contain some numerical inconsistencies relative to those literal equations and tables. Those are documented in `notes/SOURCE_REVIEW.md`, but the implementation intentionally does not overwrite the paper's listed simulation parameters.
