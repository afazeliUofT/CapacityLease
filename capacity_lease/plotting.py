from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd



def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def plot_monopoly_subscribers_revenue(monopoly_df: pd.DataFrame, out_base: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    df = monopoly_df[monopoly_df["feasible"]].copy()
    ax1.plot(df["price"], df["subscribers"], color="tab:blue", linewidth=2.0, label="Subscribers")
    ax1.set_xlabel("Subscription price p")
    ax1.set_ylabel("Subscribers n(p)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["price"], df["revenue"], color="tab:red", linewidth=2.0, label="Revenue")
    ax2.set_ylabel("Revenue R(p)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.suptitle("Monopoly: subscribers and revenue versus price")
    _save(fig, Path(out_base))



def plot_monopoly_acceptance_rate(monopoly_df: pd.DataFrame, out_base: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    df = monopoly_df[monopoly_df["feasible"]].copy()
    for idx in range(1, 5):
        ax1.plot(df["price"], df[f"A_group_{idx}"], linewidth=1.8, label=f"Group {idx}")
    ax1.set_xlabel("Subscription price p")
    ax1.set_ylabel("Acceptance probability")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(df["price"], df["rate_mbps"], color="tab:red", linewidth=2.0, label="Target rate")
    ax2.set_ylabel("Target rate r(p) [Mbps]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.suptitle("Monopoly: group acceptance and target rate versus price")
    _save(fig, Path(out_base))



def plot_capacity_revenues(mc_df: pd.DataFrame, flex_df: pd.DataFrame, monopoly_revenue: float, out_base: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    mc = mc_df[mc_df["valid"]].copy()
    fx = flex_df[flex_df["valid"]].copy()
    ax.plot(mc["capacity_leased_mbps"], mc["revenue_mno_total"], linewidth=2.0, label="Market clearing")
    ax.plot(fx["capacity_leased_mbps"], fx["revenue_mno_total"], linewidth=2.0, linestyle="--", label="Flexible participation")
    ax.axhline(monopoly_revenue, color="black", linewidth=1.5, linestyle=":", label="Monopoly optimum")
    ax.set_xlabel("Leased capacity C_V [Mbps]")
    ax.set_ylabel("MNO total revenue")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle("MNO maximum revenue versus leased capacity")
    _save(fig, Path(out_base))



def plot_capacity_prices(mc_df: pd.DataFrame, flex_df: pd.DataFrame, out_base: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    mc = mc_df[mc_df["valid"]].copy()
    fx = flex_df[flex_df["valid"]].copy()

    ax1.plot(mc["capacity_leased_mbps"], mc["pM"], color="tab:blue", linewidth=2.0, label="p_M market clearing")
    ax1.plot(fx["capacity_leased_mbps"], fx["pM"], color="tab:blue", linewidth=2.0, linestyle="--", label="p_M flexible")
    ax1.set_xlabel("Leased capacity C_V [Mbps]")
    ax1.set_ylabel("MNO price p_M", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(mc["capacity_leased_mbps"], mc["pV"], color="tab:red", linewidth=2.0, label="p_V market clearing")
    ax2.plot(fx["capacity_leased_mbps"], fx["pV"], color="tab:red", linewidth=2.0, linestyle="--", label="p_V flexible")
    ax2.set_ylabel("MVNO price p_V", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    handles = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax1.legend(handles, labels, loc="best")
    fig.suptitle("Optimal prices versus leased capacity")
    _save(fig, Path(out_base))



def plot_nm_revenues(mc_nm_df: pd.DataFrame, flex_nm_df: pd.DataFrame, out_base: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    mc = mc_nm_df[mc_nm_df["valid"]].copy()
    fx = flex_nm_df[flex_nm_df["valid"]].copy()

    ax1.plot(mc["nM"], mc["revenue_mno_total"], color="tab:blue", linewidth=2.0, label="MNO revenue (MC)")
    ax1.plot(fx["nM"], fx["revenue_mno_total"], color="tab:blue", linewidth=2.0, linestyle="--", label="MNO revenue (flex)")
    ax1.set_xlabel("MNO subscribers n_M")
    ax1.set_ylabel("MNO total revenue", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(mc["nM"], mc["revenue_mvno_gross"], color="tab:red", linewidth=2.0, label="MVNO gross revenue (MC)")
    ax2.plot(fx["nM"], fx["revenue_mvno_gross"], color="tab:red", linewidth=2.0, linestyle="--", label="MVNO gross revenue (flex)")
    ax2.set_ylabel("MVNO gross revenue", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    handles = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax1.legend(handles, labels, loc="best", fontsize=9)
    fig.suptitle("Revenues versus MNO subscriber count")
    _save(fig, Path(out_base))



def plot_nm_prices(mc_nm_df: pd.DataFrame, flex_nm_df: pd.DataFrame, out_base: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    mc = mc_nm_df[mc_nm_df["valid"]].copy()
    fx = flex_nm_df[flex_nm_df["valid"]].copy()

    ax1.plot(mc["nM"], mc["pM"], color="tab:blue", linewidth=2.0, label="p_M (MC)")
    ax1.plot(fx["nM"], fx["pM"], color="tab:blue", linewidth=2.0, linestyle="--", label="p_M (flex)")
    ax1.set_xlabel("MNO subscribers n_M")
    ax1.set_ylabel("MNO price p_M", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(mc["nM"], mc["pV"], color="tab:red", linewidth=2.0, label="p_V (MC)")
    ax2.plot(fx["nM"], fx["pV"], color="tab:red", linewidth=2.0, linestyle="--", label="p_V (flex)")
    ax2.set_ylabel("MVNO price p_V", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    handles = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax1.legend(handles, labels, loc="best", fontsize=9)
    fig.suptitle("Prices versus MNO subscriber count")
    _save(fig, Path(out_base))
