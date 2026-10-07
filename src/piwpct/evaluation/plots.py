"""Publication-quality plots for PI-WPCT evaluation, mechanism diagnostics, and robustness."""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def set_plot_style():
    """Configure clean publication-quality styling."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
    })


def plot_risk_curves(
    wps: List[float],
    r_light_risks: List[float],
    r_charm_risks: List[float],
    risk_ratios: List[float],
    save_path: str,
) -> None:
    """Plot R_light(w), R_charm(w) and risk ratio vs working point."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(wps, r_light_risks, marker="o", color="#1f77b4", label=r"Light Risk $R_{\rm light}(w)$")
    ax1.plot(wps, r_charm_risks, marker="s", color="#ff7f0e", label=r"Charm Risk $R_{\rm charm}(w)$")
    ax1.set_xlabel(r"Target $b$-efficiency Working Point $w$")
    ax1.set_ylabel(r"Risk Contribution $R_i(w)$")
    ax1.set_title("Physics-Impact Risk Components vs Working Point")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(wps, risk_ratios, marker="^", color="#2ca02c", label=r"Risk Ratio $R_{\rm light} / R_{\rm charm}$")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="Equal Allocation")
    ax2.set_xlabel(r"Target $b$-efficiency Working Point $w$")
    ax2.set_ylabel(r"Ratio $R_{\rm light}(w) / R_{\rm charm}(w)$")
    ax2.set_title("Dynamic Allocation Ratio vs Working Point")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_lambda_trajectories(
    trajectories: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Plot evolution of loss weights and purity across training epochs."""
    set_plot_style()
    epochs = trajectories.get("epoch", [])
    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, trajectories.get("lambda_light", []), marker="o", color="#1f77b4", label=r"$\lambda_{\rm light}(w)$")
    ax1.plot(epochs, trajectories.get("lambda_charm", []), marker="s", color="#ff7f0e", label=r"$\lambda_{\rm charm}(w)$")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss Weight")
    ax1.set_title("Evolution of Dynamic Loss Weights During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, trajectories.get("purity", []), marker="d", color="#9467bd", label=r"Purity $P(w)$")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Purity")
    ax2.set_title("Downstream Selection Purity Proxy Trajectory")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_conformal_coverage(
    per_class_coverage: Dict[int, float],
    target_coverage: float,
    class_names: Optional[List[str]] = None,
    save_path: str = "plots/conformal_coverage.png",
) -> None:
    """Plot exact Mondrian conformal empirical coverage vs target per class."""
    set_plot_style()
    names = class_names or ["Light", "Charm", "b"]
    if len(names) <= max(per_class_coverage.keys(), default=-1):
        names = ["Class %d" % index for index in sorted(per_class_coverage)]
    classes = sorted(per_class_coverage.keys())
    covs = [per_class_coverage[c] for c in classes]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([names[c] for c in classes], covs,
                  color=[colors[index % len(colors)] for index in range(len(classes))], width=0.5)
    ax.axhline(target_coverage, color="red", linestyle="--", linewidth=1.8, label=f"Target ({target_coverage*100:.0f}%)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Exact Mondrian Class-Conditional Coverage")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_robustness_curves(
    robustness_results: Dict[str, List[Dict[str, float]]],
    save_path: str,
) -> None:
    """Plot conformal coverage and background rejection degradation under systematic mismodellings."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. JES
    jes_data = robustness_results.get("jes", [])
    if jes_data:
        x = [d["factor"] for d in jes_data]
        cov = [d["marginal_coverage"] for d in jes_data]
        axes[0, 0].plot(x, cov, marker="o", color="#d62728")
        axes[0, 0].set_xlabel("JES Shift Factor")
        axes[0, 0].set_ylabel("Marginal Conformal Coverage")
        axes[0, 0].set_title("Coverage vs Jet Energy Scale (JES)")
        axes[0, 0].grid(True, alpha=0.3)

    # 2. JER
    jer_data = robustness_results.get("jer", [])
    if jer_data:
        x = [d["sigma"] for d in jer_data]
        cov = [d["marginal_coverage"] for d in jer_data]
        axes[0, 1].plot(x, cov, marker="s", color="#9467bd")
        axes[0, 1].set_xlabel("JER Smear Sigma")
        axes[0, 1].set_ylabel("Marginal Conformal Coverage")
        axes[0, 1].set_title("Coverage vs Jet Energy Resolution (JER)")
        axes[0, 1].grid(True, alpha=0.3)

    # 3. IP Smearing
    ip_data = robustness_results.get("ip_smear", [])
    if ip_data:
        x = [d["sigma"] for d in ip_data]
        rej_l = [d["rej_light"] for d in ip_data]
        axes[1, 0].plot(x, rej_l, marker="^", color="#1f77b4")
        axes[1, 0].set_xlabel("Impact Parameter Smear Sigma")
        axes[1, 0].set_ylabel("Light Jet Rejection (1 / mistag)")
        axes[1, 0].set_title("Light Rejection vs IP Smearing")
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Track Dropping
    td_data = robustness_results.get("track_drop", [])
    if td_data:
        x = [d["drop_rate"] for d in td_data]
        cov = [d["marginal_coverage"] for d in td_data]
        axes[1, 1].plot(x, cov, marker="d", color="#2ca02c")
        axes[1, 1].set_xlabel("Constituent Track Drop Rate")
        axes[1, 1].set_ylabel("Marginal Conformal Coverage")
        axes[1, 1].set_title("Coverage vs Tracking Inefficiency")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
