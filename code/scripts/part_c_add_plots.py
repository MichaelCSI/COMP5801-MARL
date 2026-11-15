
"""Additional Part C figures from existing seed-level analysis.

Run from the repository's code/ directory:
    python -m scripts.part_c_extra_plots

Never reruns training or evaluation.
Five training seeds are the statistical unit.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t



ROOT = Path("results") / "part_c"
OUT = ROOT / "extra_plots"

TEST = ROOT / "test_combined" / "analysis" / "per_seed.csv"
VALIDATION = ROOT / "validation_combined" / "analysis" / "per_seed.csv"

ARMS = (("M0", "M0"),("M1", "M1 p=1.0"),("M1-RND", "M1-RND p=1.0"),)




def ci95(values):
    """Return seed-level mean and 95% t CI half-width."""

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError("At least two independent training seeds are required")

    mean = float(x.mean())
    half_width = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1)/ np.sqrt(len(x)))
    return mean, half_width


def save(fig, filename):
    """Save a figure to the supplementary results directory."""
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def core_by_arm(data, arm):
    """Select normal communication conditions for one arm."""
    return data[(data["arm"] == arm) & (data["hold_last"] == False) & (data["force_mask"] == False)]


# Per-seed improvement relative to M0

def plot_per_seed_improvement(test):

    base = test.loc[test["condition"] == "M0",["seed", "primary"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for arm, offset in (("M1", -0.09), ("M1-RND", 0.09),):
        candidate = test.loc[test["condition"] == f"{arm} p=1.0",["seed", "primary"]]
        pairs = base.merge(candidate, on="seed", suffixes=("_m0", "_arm"), validate="one_to_one").sort_values("seed")
        improvement = (100 * (pairs["primary_m0"] - pairs["primary_arm"])  / pairs["primary_m0"] )
        ax.scatter(pairs["seed"] + offset, improvement, s=65, label=arm)
        ax.plot(pairs["seed"] + offset, improvement, alpha=0.45)

    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set(xticks=sorted(base["seed"].unique()), xlabel="Independent training seed", ylabel="Peak improvement vs M0 (%)", title="Per-seed paired improvement at perfect delivery (p=1)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Positive = better than M0")
    save(fig, "01_per_seed_improvement.png")



#  Peak reduction versus electricity cost


def plot_peak_cost_tradeoff(test):

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, condition in ARMS:
        group = test[test["condition"] == condition]
        points = ax.scatter(group["primary"], group["cost"], alpha=0.5, s=45, label=f"{arm} seeds")
        facecolor = points.get_facecolors()[0]
        ax.scatter([group["primary"].mean()],[group["cost"].mean()], marker="*", s=240, color=facecolor, edgecolors="black", linewidths=0.6)

    ax.set(xlabel=("Primary KPI: average daily peak / R0 " "(lower is better)"), ylabel="Electricity-cost metric (lower is better)", title="Peak-versus-cost trade-off at p=1")
    ax.grid(alpha=0.2)
    ax.legend(title="Dots = seeds; stars = means")
    save(fig, "02_peak_cost_tradeoff.png")



#Absolute battery correlation


def plot_absolute_correlation(test):

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for arm in ("M1", "M1-RND"):

        group = core_by_arm(test, arm)

        probabilities = []
        means = []
        lower_bounds = []
        upper_bounds = []

        for p, data in sorted(
            group.groupby("p"),
            key=lambda pair: pair[0]
        ):

            mean, half_width = ci95(
                data["mean_abs_correlation"]
            )

            probabilities.append(float(p))
            means.append(mean)
            lower_bounds.append(mean - half_width)
            upper_bounds.append(mean + half_width)

        line, = ax.plot(
            probabilities,
            means,
            marker="o",
            label=arm
        )

        ax.fill_between(
            probabilities,
            lower_bounds,
            upper_bounds,
            color=line.get_color(),
            alpha=0.14
        )

    # M0 baseline
    m0_mean, m0_half_width = ci95(
        test.loc[
            test["condition"] == "M0",
            "mean_abs_correlation"
        ]
    )

    ax.axhline(
        m0_mean,
        linestyle="--",
        color="gray",
        label="M0"
    )

    ax.axhspan(
        m0_mean - m0_half_width,
        m0_mean + m0_half_width,
        color="gray",
        alpha=0.1
    )

    ax.set(
        xlabel="Message delivery probability p",
        ylabel="Mean absolute battery pair correlation |r|",
        title="Absolute battery coupling across communication conditions"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    save(fig, "03_absolute_correlation.png")



# HOLD versus standard message dropout


def plot_hold_effect(test):

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for p in (0.25, 0.5, 0.9):

        standard = test.loc[
            test["condition"] == f"M1 p={p}",
            ["seed", "primary"]
        ]

        hold = test.loc[
            test["condition"] == f"M1-HOLD p={p}",
            ["seed", "primary"]
        ]

        pairs = standard.merge(
            hold,
            on="seed",
            suffixes=("_standard", "_hold"),
            validate="one_to_one"
        )

        # Negative values indicate HOLD improves peak KPI.
        delta = (
            pairs["primary_hold"]
            - pairs["primary_standard"]
        )

        mean, half_width = ci95(delta)

        # Individual seed results
        ax.scatter(
            np.full(len(delta), p),
            delta,
            alpha=0.4,
            s=30
        )

        # Mean and 95% confidence interval
        ax.errorbar(
            [p],
            [mean],
            yerr=[half_width],
            fmt="o",
            markersize=8,
            capsize=5,
            color="black"
        )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    ax.set(
        xticks=[0.25, 0.5, 0.9],
        xlabel="Message delivery probability p",
        ylabel="Primary KPI: HOLD minus standard dropout",
        title="Paired HOLD effect (negative = HOLD helps)"
    )

    ax.grid(axis="y", alpha=0.2)

    save(fig, "04_hold_paired_effect.png")



#Validation versus test performance


def plot_validation_vs_test(validation, test):

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for label, data, offset in (
        ("Validation", validation, -0.10),
        ("Test", test, 0.10),
    ):

        means = []
        half_widths = []

        for _, condition in ARMS:

            mean, half_width = ci95(
                data.loc[
                    data["condition"] == condition,
                    "primary"
                ]
            )

            means.append(mean)
            half_widths.append(half_width)

        x_positions = np.arange(len(ARMS)) + offset

        ax.errorbar(
            x_positions,
            means,
            yerr=half_widths,
            fmt="o",
            capsize=5,
            markersize=8,
            label=label
        )

    ax.set(
        xticks=np.arange(len(ARMS)),
        xticklabels=[arm for arm, _ in ARMS],
        ylabel="Primary KPI (lower is better)",
        title="Validation vs held-out test at perfect delivery (p=1)"
    )

    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="95% CIs across five training seeds")

    save(fig, "05_validation_vs_test.png")





def main():

    if not TEST.is_file():
        raise FileNotFoundError(
            f"Missing {TEST}. Run part_c_report first."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(TEST)

    # Verify one aggregated record per seed and condition.
    if test.duplicated(
        ["arm", "condition", "seed"]
    ).any():
        raise ValueError(
            "Expected one aggregated record per arm/condition/seed"
        )

    print("Generating supplementary Part C graphs...")

    plot_per_seed_improvement(test)

    plot_peak_cost_tradeoff(test)

    plot_absolute_correlation(test)

    plot_hold_effect(test)

    # This graph requires validation data.
    if VALIDATION.is_file():

        validation = pd.read_csv(VALIDATION)

        plot_validation_vs_test(
            validation,
            test
        )

    else:
        print(
            f"Skipped validation comparison: missing {VALIDATION}"
        )

    print("All available supplementary plots generated.")


if __name__ == "__main__":
    main()
