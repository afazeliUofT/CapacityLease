import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from capacity_lease.config import load_config
from capacity_lease.monopoly import solve_monopoly_problem
from capacity_lease.market_clearing import profitability_interval


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "paper_literal.json")
    monopoly = solve_monopoly_problem(config.model, config.solver)
    assert monopoly.optimal_revenue > 0.0
    interval = profitability_interval(config.model, config.solver, 100.0)
    assert interval is not None
    assert interval[0] < interval[1]
    print("Smoke test passed.")


if __name__ == "__main__":
    main()