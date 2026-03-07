from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path


def _pin_blas_threads() -> None:
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ.setdefault(name, "1")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simulation data and plots for the CpctLease paper.")
    parser.add_argument("command", choices=["generate-all"], help="Action to run.")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file.")
    parser.add_argument("--outdir", required=True, help="Directory for CSV data, plots, and summary JSON.")
    parser.add_argument("--workers", type=int, default=None, help="Override worker count from the config.")
    parser.add_argument(
        "--capacity-step",
        type=float,
        default=None,
        help="Optional override for capacity step in Mbps.",
    )
    parser.add_argument(
        "--pin-blas-threads",
        action="store_true",
        help="Set BLAS/OpenMP thread counts to 1 to avoid oversubscription with process parallelism.",
    )
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.pin_blas_threads:
        _pin_blas_threads()

    import pandas as pd  # delayed import so thread pinning happens first

    from .config import load_config
    from .flexible import solve_flexible_capacity_curve, solve_flexible_nm_curve
    from .market_clearing import solve_market_clearing_capacity_curve, solve_market_clearing_nm_curve
    from .monopoly import solve_monopoly_problem
    from .plotting import (
        plot_capacity_prices,
        plot_capacity_revenues,
        plot_monopoly_acceptance_rate,
        plot_monopoly_subscribers_revenue,
        plot_nm_prices,
        plot_nm_revenues,
    )
    from .utils import save_dataframe, save_json

    config = load_config(args.config)
    solver = config.solver
    if args.workers is not None:
        solver = replace(solver, workers=int(args.workers))
    if args.capacity_step is not None:
        solver = replace(solver, capacity_step_mbps=float(args.capacity_step))
    config = replace(config, solver=solver)

    outdir = Path(args.outdir)
    data_dir = outdir / "data"
    plots_dir = outdir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    monopoly = solve_monopoly_problem(config.model, config.solver)
    save_dataframe(monopoly.curve, data_dir / "monopoly_curve.csv")
    plot_monopoly_subscribers_revenue(monopoly.curve, plots_dir / "monopoly_n_R")
    plot_monopoly_acceptance_rate(monopoly.curve, plots_dir / "monopoly_Ag_r")

    mc_curve = solve_market_clearing_capacity_curve(config.model, config.solver)
    save_dataframe(mc_curve, data_dir / "market_clearing_capacity_curve.csv")

    flex_curve = solve_flexible_capacity_curve(config.model, config.solver, monopoly.optimal_revenue)
    save_dataframe(flex_curve, data_dir / "flexible_capacity_curve.csv")

    plot_capacity_revenues(mc_curve, flex_curve, monopoly.optimal_revenue, plots_dir / "MNO_MVNO_CapacityBlocks")
    plot_capacity_prices(mc_curve, flex_curve, plots_dir / "Optimal_Prices_vs_Capacity")

    mc_nm = solve_market_clearing_nm_curve(
        config.model,
        config.solver,
        config.solver.reported_market_clearing_capacity_mbps,
    )
    save_dataframe(mc_nm, data_dir / "market_clearing_nM_curve.csv")

    flex_nm = solve_flexible_nm_curve(
        config.model,
        config.solver,
        monopoly.optimal_revenue,
        config.solver.reported_flexible_capacity_mbps,
    )
    save_dataframe(flex_nm, data_dir / "flexible_nM_curve.csv")

    plot_nm_revenues(mc_nm, flex_nm, plots_dir / "MVNO_MNO_Revenue")
    plot_nm_prices(mc_nm, flex_nm, plots_dir / "MVNO_MNO_Prices")

    summary = {
        "config": {
            "model": {
                "group_sizes": config.model.group_sizes.tolist(),
                "alpha": config.model.alpha.tolist(),
                "beta": config.model.beta.tolist(),
                "monopoly_noise_mean": config.model.monopoly_noise_mean.tolist(),
                "monopoly_noise_sd": config.model.monopoly_noise_sd.tolist(),
                "mno_noise_mean": config.model.mno_noise_mean.tolist(),
                "mno_noise_sd": config.model.mno_noise_sd.tolist(),
                "mvno_noise_mean": config.model.mvno_noise_mean.tolist(),
                "mvno_noise_sd": config.model.mvno_noise_sd.tolist(),
                "capacity_mbps": config.model.capacity_mbps,
                "delta": config.model.delta,
                "zeta": config.model.zeta,
                "lambda_retention": config.model.lambda_retention,
                "participation_fee": config.model.participation_fee,
            },
            "solver": {
                "root_xtol": config.solver.root_xtol,
                "root_rtol": config.solver.root_rtol,
                "max_root_iter": config.solver.max_root_iter,
                "use_fast_bvn_if_available": config.solver.use_fast_bvn_if_available,
                "market_clearing_quantile_tail": config.solver.market_clearing_quantile_tail,
                "monopoly_price_points": config.solver.monopoly_price_points,
                "monopoly_plot_price_max": config.solver.monopoly_plot_price_max,
                "capacity_step_mbps": config.solver.capacity_step_mbps,
                "market_clearing_search_points": config.solver.market_clearing_search_points,
                "flexible_nM_search_points": config.solver.flexible_nM_search_points,
                "flexible_nV_search_points": config.solver.flexible_nV_search_points,
                "refine_levels": config.solver.refine_levels,
                "top_k_seeds": config.solver.top_k_seeds,
                "nM_curve_points": config.solver.nM_curve_points,
                "workers": config.solver.workers,
                "chunksize": config.solver.chunksize,
                "reported_market_clearing_capacity_mbps": config.solver.reported_market_clearing_capacity_mbps,
                "reported_flexible_capacity_mbps": config.solver.reported_flexible_capacity_mbps,
            },
        },
        "monopoly": {
            "critical_price": monopoly.critical_price,
            "optimal_price": monopoly.optimal_price,
            "optimal_subscribers": monopoly.optimal_subscribers,
            "optimal_rate": monopoly.optimal_rate,
            "optimal_revenue": monopoly.optimal_revenue,
        },
        "market_clearing_best_capacity": (
            None
            if mc_curve.empty or mc_curve["revenue_mno_total"].dropna().empty
            else mc_curve.loc[mc_curve["revenue_mno_total"].idxmax()].to_dict()
        ),
        "flexible_best_capacity": (
            None
            if flex_curve.empty or flex_curve["revenue_mno_total"].dropna().empty
            else flex_curve.loc[flex_curve["revenue_mno_total"].idxmax()].to_dict()
        ),
    }
    save_json(summary, outdir / "summary.json")


if __name__ == "__main__":
    main()
