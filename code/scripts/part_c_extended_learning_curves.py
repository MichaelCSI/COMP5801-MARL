import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from matplotlib.ticker import FuncFormatter, MaxNLocator





RUNS = Path("runs")
OUT = Path("results/part_c/extended/training_analysis")

ARMS = ["M0", "M1", "M1-RND", "I0", "M-FULL"]
ARM_DIRS = {arm: RUNS for arm in ARMS}
ARM_DIRS.update({arm: RUNS / "extended" for arm in ("I0", "M-FULL")})
ARM_SETTINGS = {
    "M0": (False, "central", "none", False),
    "M1": (True, "central", "fixed", False),
    "M1-RND": (True, "central", "random", False),
    "I0": (False, "local", "none", False),
    "M-FULL": (False, "central", "none", True),
}
SEEDS = [0, 1, 2, 3, 4]

COLORS = {
    "M0": "#1f77b4",
    "M1": "#ff7f0e",
    "M1-RND": "#2ca02c",
    "I0": "#d62728",
    "M-FULL": "#9467bd",
}

EXPECTED_STEPS = 200000
ROLLOUT_STEPS = 1000
SMOOTHING_WINDOW = 10

METRICS = ["ep_return_mean", "value_loss", "policy_loss", "dist_entropy",
           "actor_grad_norm", "critic_grad_norm", "ratio", "steps_per_sec"]



# Load and validate training logs


def load_logs():

    records = []
    configurations = []

    for arm in ARMS:
        for seed in SEEDS:
            path = ARM_DIRS[arm] / f"{arm}_s{seed}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)

            with path.open("r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]

            config = entries[0]["config"]
            expected = dict(arm=arm, seed=seed, total_steps=EXPECTED_STEPS,
                            rollout_steps=ROLLOUT_STEPS)
            expected.update(zip(("use_messages", "critic", "train_schedule",
                                 "full_peer_info"), ARM_SETTINGS[arm]))
            for key, value in expected.items():
                if config.get(key) != value:
                    raise ValueError(f"{path}: {key}={config.get(key)!r}, expected {value!r}")
            if entries[0].get("env") != "citylearn":
                raise ValueError(f"{path}: expected CityLearn environment")
            configurations.append({**config, "env": entries[0]["env"], "source_log": str(path)})

            updates = [entry for entry in entries if "env_steps" in entry]
            df = pd.DataFrame(updates)
            df = df.sort_values("env_steps")
            expected_steps = list(range(ROLLOUT_STEPS, EXPECTED_STEPS + 1, ROLLOUT_STEPS))
            if df["env_steps"].tolist() != expected_steps:
                raise ValueError(f"Incomplete or duplicate step sequence: {path}")
            if df["update"].tolist() != list(range(1, len(expected_steps) + 1)):
                raise ValueError(f"Incorrect update sequence: {path}")
            if not (df["arm"].eq(arm).all() and df["seed"].eq(seed).all()):
                raise ValueError(f"Incorrect update identifiers: {path}")
            for metric in METRICS:
                df[metric] = pd.to_numeric(df[metric], errors="raise")
                if not np.isfinite(df[metric].to_numpy(dtype=float)).all():
                    raise ValueError(f"Non-finite {metric}: {path}")
            if (df["steps_per_sec"] <= 0).any():
                raise ValueError(f"Non-positive rollout throughput: {path}")

            records.append(df)
            print(f"Loaded {path.name}: "f"{len(df)} updates, "f"{df['env_steps'].max()} steps")
    data = pd.concat(records, ignore_index=True)
    for key in ("normalize_obs", "obs_clip", "device", "onpolicy_overrides",
                "force_mask_train", "train_p", "rnd_p_choices"):
        if len({json.dumps(c[key], sort_keys=True) for c in configurations}) != 1:
            raise ValueError(f"Inconsistent logged training setting: {key}")
    pd.DataFrame(configurations).to_csv(OUT / "training_configurations.csv",index=False)

    return data



# Seed-level stats


def aggregate_metric(data, metric, smooth=False):
    subset = data[["arm", "seed", "env_steps", metric]].copy()
    subset = subset.sort_values(["arm", "seed", "env_steps"])

    if smooth:
        subset[metric] = (subset.groupby(["arm", "seed"])[metric]
            .transform(lambda x: x.rolling(SMOOTHING_WINDOW,min_periods=SMOOTHING_WINDOW).mean()))

    summary = (subset.groupby(["arm", "env_steps"])[metric].agg(["mean", "std", "count"]).reset_index())
    summary = summary[summary["count"] == len(SEEDS)].copy()

    summary["ci95"] = ( t.ppf(0.975, summary["count"] - 1) * summary["std"] / np.sqrt(summary["count"]))
    summary["lower"] = summary["mean"] - summary["ci95"]
    summary["upper"] = summary["mean"] + summary["ci95"]
    return summary


def save_figure(fig, filename):
    fig.tight_layout(rect=(0, 0, 1, 0.95) if fig._suptitle else None)
    path = OUT / filename
    fig.savefig(path, dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    
def format_steps(x, pos=None):
    x = float(x)
    if abs(x) >= 1_000_000:
        value = x / 1_000_000
        if value.is_integer():
            return f"{int(value)}M"
        return f"{value:.1f}M"

    if abs(x) >= 1_000:
        value = x / 1_000
        if value.is_integer():
            return f"{int(value)}k"
        return f"{value:.1f}k"

    if x.is_integer():
        return str(int(x))

    return f"{x:.0f}"


def apply_step_axis_format(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


# Graph 1: Training learning curves

def plot_learning_curves(data):
    fig, ax = plt.subplots(figsize=(9, 5))
    raw = aggregate_metric(data,"ep_return_mean",smooth=False)
    smoothed = aggregate_metric(data,"ep_return_mean",smooth=True)
    for arm in ARMS:
        raw_arm = raw[raw["arm"] == arm]
        group = smoothed[smoothed["arm"] == arm]
        color = COLORS[arm]
        ax.plot(raw_arm["env_steps"],raw_arm["mean"],color=color,alpha=0.18,linewidth=0.8)# Unsmoothed mean
        ax.plot( group["env_steps"],group["mean"],color=color,linewidth=2,label=arm)# Smoothed mean
        ax.fill_between(group["env_steps"].to_numpy(dtype=float), group["lower"].to_numpy(dtype=float), group["upper"].to_numpy(dtype=float),color=color,alpha=0.15)# Confidence Interval across training seeds
        
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Five-controller training curves (five seeds per controller)")
    ax.grid(alpha=0.2)
    ax.legend(title="10-update rolling mean; pointwise 95% CI")
    apply_step_axis_format(ax)
    save_figure(fig,"01_training_learning_curves.png")



# Graph 2: Individual seed learning curves


def plot_individual_seeds(data):

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, arm in zip(axes, ARMS):
        group = data[data["arm"] == arm]
        for seed in SEEDS:
            run = group[ group["seed"] == seed ].sort_values("env_steps")
            smoothed = run["ep_return_mean"].rolling(SMOOTHING_WINDOW,min_periods=SMOOTHING_WINDOW ).mean()
            ax.plot(run["env_steps"],smoothed, linewidth=1.3,label=f"Seed {seed}")
            
        ax.set_title(arm)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.2)
        apply_step_axis_format(ax)

    axes[0].set_ylabel("Mean episode return")
    axes[3].set_ylabel("Mean episode return")
    for ax in axes[:len(ARMS)]:
        ax.tick_params(labelbottom=True)
    axes[-1].set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", title="Training seed", frameon=False)
    fig.suptitle("Training variability (10-update rolling means)")
    save_figure( fig,"02_individual_seed_learning_curves.png")



# Graph 3: Optimization diagnostics

def plot_optimization_diagnostics(data):

    metrics = [ ("value_loss", "Value loss"),("policy_loss", "Policy loss"), ("dist_entropy", "Policy entropy"), ("actor_grad_norm", "Actor gradient norm"), 
               ("critic_grad_norm", "Critic gradient norm"), ("ratio", "PPO probability ratio")]
    fig, axes = plt.subplots( 2, 3,figsize=(15, 8))
    for ax, (metric, label) in zip( axes.flat, metrics):
        summary = aggregate_metric( data, metric, smooth=True )
        for arm in ARMS:
            group = summary[summary["arm"] == arm ]
            ax.plot( group["env_steps"], group["mean"], color=COLORS[arm],label=arm,linewidth=1.5)
            ax.fill_between(group["env_steps"].to_numpy(dtype=float), group["lower"].to_numpy(dtype=float), group["upper"].to_numpy(dtype=float),color=COLORS[arm], alpha=0.12)

        ax.set_title(label)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.2)
        apply_step_axis_format(ax)


    axes[0, 0].legend()
    fig.suptitle("PPO optimization diagnostics (10-update means; pointwise 95% seed-level CI)" )
    save_figure( fig,"03_optimization_diagnostics.png")



# Training duration and return summary


def summarize_training(data):

    records = []
    for (arm, seed), group in data.groupby(["arm", "seed"]):
        group = group.sort_values("env_steps")
        first_return = group["ep_return_mean"].head(10).mean()
        final_return = group[ "ep_return_mean"].tail(10).mean()
        final_steps = group["env_steps"].iloc[-1]
        steps_per_update = group["env_steps"].diff()
        steps_per_update.iloc[0] = group["env_steps"].iloc[0]

        throughput = group["steps_per_sec"].astype(float)
        if not np.isfinite(throughput).all() or (throughput <= 0).any():
            raise ValueError(f"Invalid throughput for {arm}, seed {seed}")

        # Collection timer excludes PPO compute/train and other overhead.
        estimated_seconds = float((steps_per_update / throughput).sum())
        effective_sps = final_steps / estimated_seconds

        records.append({
            "arm": arm,
            "seed": seed,
            "updates": len(group),
            "total_steps": final_steps,
            "initial_return_first_10_updates": first_return,
            "final_return_last_10_updates": final_return,
            "return_change": final_return - first_return,
            "effective_steps_per_sec": effective_sps,
            "estimated_rollout_collection_seconds": estimated_seconds,
            "estimated_rollout_collection_minutes": estimated_seconds / 60,
        })

    per_seed = pd.DataFrame(records)

    per_seed.to_csv(OUT / "training_summary_per_seed.csv",index=False)
    summary = (
        per_seed.groupby("arm")
        .agg(
            seeds=("seed", "count"),
            mean_initial_return=("initial_return_first_10_updates", "mean"),
            mean_final_return=( "final_return_last_10_updates", "mean"),
            sd_final_return=("final_return_last_10_updates", "std"),
            mean_return_change=("return_change", "mean"),
            mean_steps_per_sec=("effective_steps_per_sec", "mean"),
            mean_rollout_collection_minutes=("estimated_rollout_collection_minutes", "mean" ),
            min_rollout_collection_minutes=("estimated_rollout_collection_minutes", "min" ),
            max_rollout_collection_minutes=( "estimated_rollout_collection_minutes", "max" ),
            total_rollout_collection_minutes=("estimated_rollout_collection_minutes", "sum" )).reset_index() 
        )

    summary.to_csv( OUT / "training_summary_by_arm.csv", index=False )
    print("\nPER-SEED TRAINING SUMMARY\n")
    print(per_seed.round(4).to_string(index=False))
    print("\nCONTROLLER TRAINING SUMMARY\n")
    print( summary.round(4).to_string(index=False) )
    print("\nTotal estimated rollout collection hours (excludes optimization and overhead):",round(per_seed["estimated_rollout_collection_minutes"].sum() / 60, 3))



# Main


def main():

    OUT.mkdir(parents=True, exist_ok=True)
    data = load_logs()
    print(f"\nTotal training update records: {len(data)}")
    expected_records = len(ARMS) * len(SEEDS) * (EXPECTED_STEPS // ROLLOUT_STEPS)
    if len(data) != expected_records:
        raise ValueError(f"Expected {expected_records} records, got {len(data)}")
    plot_learning_curves(data)
    plot_individual_seeds(data)
    plot_optimization_diagnostics(data)
    summarize_training(data)
    print("\nTraining analysis completed successfully.")


if __name__ == "__main__":
    main()
