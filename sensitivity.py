"""Run a small dynamic sensitivity analysis for the micro model.

Inputs:
    A Config object or config path, an output folder, sample count, simulated
    duration, and a random seed.

Outputs:
    A CSV table of Sobol total-order indices over time and one PNG plot.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_DIR = Path(tempfile.gettempdir()) / "swarm_foraging_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import MicroModel
from config import Config


PARAMETER_BOUNDS = {
    "alpha": (0.5, 1.0),
    "lambda_loss": (1.0, 5.0),
    "recency": (0.01, 0.20),
    "congestion_tolerance": (0.01, 0.10),
}


def run_sensitivity_analysis(
    cfg: Config,
    out_dir: str | Path,
    samples: int = 12,
    seconds: float = 3000.0,
    sample_every: float = 100.0,
    seed: int | None = None,
) -> list[Path]:
    """Run sensitivity simulations and write the CSV plus plot.

    Inputs are the model configuration, output folder, sample count, simulated
    seconds, sampling interval, and optional seed. Outputs are returned as file
    paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_value = int(seed if seed is not None else cfg.get("run", "random_seed", default=7))
    problem = sobol_problem()
    design = make_sobol_design(problem, samples, seed_value)
    milestones = time_milestones(seconds)
    energy = run_parameter_sweep(cfg, design["values"], seconds, sample_every, milestones, seed_value)
    data = analyze_parameter_importance(problem, design, energy, milestones)

    csv_path = out / "dynamic_sobol_indices.csv"
    data.to_csv(csv_path, index=False)
    png_path = out / "fig_dynamic_sobol.png"
    plot_sensitivity(data, png_path)
    return [png_path, csv_path]


def sobol_problem() -> dict[str, object]:
    """Return the parameter space used by the Sobol analysis."""
    return {
        "num_vars": len(PARAMETER_BOUNDS),
        "names": list(PARAMETER_BOUNDS.keys()),
        "bounds": [list(bounds) for bounds in PARAMETER_BOUNDS.values()],
    }


def make_sobol_design(problem: dict[str, object], samples: int, seed: int) -> dict[str, object]:
    """Create A, B, and A_Bi matrices for total-order Sobol estimation."""
    samples = max(2, int(samples))
    rng = np.random.default_rng(seed)
    names = list(problem["names"])
    bounds = np.asarray(problem["bounds"], dtype=float)
    unit_a = rng.random((samples, len(names)))
    unit_b = rng.random((samples, len(names)))
    a = scale_unit_samples(unit_a, bounds)
    b = scale_unit_samples(unit_b, bounds)
    ab_matrices = []
    for index in range(len(names)):
        mixed = a.copy()
        mixed[:, index] = b[:, index]
        ab_matrices.append(mixed)
    values = np.vstack([a, b, *ab_matrices])
    return {"values": values, "samples": samples, "a_slice": (0, samples), "b_slice": (samples, 2 * samples)}


def scale_unit_samples(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Scale unit-cube samples into the configured parameter bounds."""
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def time_milestones(seconds: float) -> list[float]:
    """Choose four evenly spaced times for comparing sensitivity over the run."""
    seconds = float(seconds)
    return [0.25 * seconds, 0.50 * seconds, 0.75 * seconds, seconds]


def run_parameter_sweep(
    cfg: Config,
    parameter_values: np.ndarray,
    seconds: float,
    sample_every: float,
    milestones: list[float],
    seed: int,
) -> np.ndarray:
    """Run the micro model for every sampled parameter set and collect energy."""
    stride = max(1, int(round(float(sample_every) / cfg.dt)))
    outputs = np.zeros((len(parameter_values), len(milestones)), dtype=float)
    names = list(PARAMETER_BOUNDS.keys())

    for index, values in enumerate(parameter_values):
        params = dict(zip(names, values))
        model = MicroModel(cfg=cfg, seed=seed, **params)
        trace = model.run(seconds=seconds, stride=stride)
        for t_index, target_time in enumerate(milestones):
            closest = (trace["time_s"] - target_time).abs().idxmin()
            outputs[index, t_index] = float(trace.loc[closest, "energy"])
    return outputs


def analyze_parameter_importance(
    problem: dict[str, object],
    design: dict[str, object],
    energy: np.ndarray,
    milestones: list[float],
) -> pd.DataFrame:
    """Compute Jansen total-order Sobol indices for each parameter and time."""
    names = list(problem["names"])
    rows: list[dict[str, float | str]] = []
    samples = int(design["samples"])
    a_start, a_end = design["a_slice"]
    y_a = energy[a_start:a_end, :]
    for t_index, time_s in enumerate(milestones):
        variance = float(np.var(energy[:, t_index], ddof=1))
        for p_index, name in enumerate(names):
            start = 2 * samples + p_index * samples
            stop = start + samples
            y_ab = energy[start:stop, t_index]
            if variance <= 1e-12:
                importance = 0.0
            else:
                importance = float(np.mean((y_a[:, t_index] - y_ab) ** 2) / (2.0 * variance))
            rows.append({"time_s": time_s, "parameter": name, "importance": importance, "method": "sobol_total_order"})
    return pd.DataFrame(rows)


def plot_sensitivity(data: pd.DataFrame, path: str | Path) -> Path:
    """Draw the sensitivity values over time and save the figure."""
    path = Path(path)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    labels = {
        "alpha": r"$\alpha$ utility curvature",
        "lambda_loss": r"$\lambda$ loss aversion",
        "recency": "Learning recency",
        "congestion_tolerance": "Congestion tolerance",
    }
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for color, (parameter, group) in zip(colors, data.groupby("parameter", sort=False)):
        ax.plot(group["time_s"], group["importance"], marker="o", linewidth=1.8, color=color, label=labels.get(parameter, parameter))

    ax.set_title("Dynamic Sobol sensitivity")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Parameter importance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
