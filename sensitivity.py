"""Run a small dynamic sensitivity analysis for the micro model.

Inputs:
    A Config object or config path, an output folder, sample count, simulated
    duration, and a random seed.

Outputs:
    A CSV table of parameter importance over time and one PNG plot. If SALib is
    installed, the values are Sobol total-order indices. Otherwise the module
    writes a lightweight correlation-based screening result and labels it as
    such.
"""

from __future__ import annotations

from pathlib import Path
import warnings

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
    problem = salib_problem()
    parameter_values, method = make_parameter_samples(problem, samples, seed_value)
    milestones = time_milestones(seconds)
    energy = run_parameter_sweep(cfg, parameter_values, seconds, sample_every, milestones, seed_value)
    data = analyze_parameter_importance(problem, parameter_values, energy, milestones, method)

    csv_path = out / "sensitivity_dynamic.csv"
    data.to_csv(csv_path, index=False)
    png_path = out / "fig_dynamic_sensitivity.png"
    plot_sensitivity(data, png_path)
    return [png_path, csv_path]


def salib_problem() -> dict[str, object]:
    """Return the parameter space shared by the Sobol and fallback analyses."""
    return {
        "num_vars": len(PARAMETER_BOUNDS),
        "names": list(PARAMETER_BOUNDS.keys()),
        "bounds": [list(bounds) for bounds in PARAMETER_BOUNDS.values()],
    }


def make_parameter_samples(problem: dict[str, object], samples: int, seed: int) -> tuple[np.ndarray, str]:
    """Create parameter samples for the sensitivity run.

    SALib is used when available. The fallback uses random uniform samples and
    is reported as screening, not Sobol analysis.
    """
    samples = max(2, int(samples))
    try:
        from SALib.sample import saltelli

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = saltelli.sample(problem, samples, calc_second_order=False)
        return np.asarray(values, dtype=float), "sobol_total_order"
    except Exception:
        rng = np.random.default_rng(seed)
        bounds = np.asarray(problem["bounds"], dtype=float)
        unit = rng.random((samples, len(problem["names"])))
        values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
        return values, "screening_correlation"


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
    parameter_values: np.ndarray,
    energy: np.ndarray,
    milestones: list[float],
    method: str,
) -> pd.DataFrame:
    """Compute importance values for each parameter at each sampled time."""
    names = list(problem["names"])
    rows: list[dict[str, float | str]] = []

    if method == "sobol_total_order":
        from SALib.analyze import sobol

        for t_index, time_s in enumerate(milestones):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sobol.analyze(problem, energy[:, t_index], calc_second_order=False, print_to_console=False)
            for name, value in zip(names, result["ST"]):
                rows.append({"time_s": time_s, "parameter": name, "importance": float(value), "method": method})
        return pd.DataFrame(rows)

    for t_index, time_s in enumerate(milestones):
        y = energy[:, t_index]
        raw_scores = []
        for p_index in range(parameter_values.shape[1]):
            x = parameter_values[:, p_index]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                raw_scores.append(0.0)
            else:
                raw_scores.append(abs(float(np.corrcoef(x, y)[0, 1])))
        total = sum(raw_scores)
        scores = [score / total if total > 1e-12 else 0.0 for score in raw_scores]
        for name, value in zip(names, scores):
            rows.append({"time_s": time_s, "parameter": name, "importance": float(value), "method": method})
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

    method = str(data["method"].iloc[0]) if not data.empty else "sensitivity"
    title = "Dynamic Sobol sensitivity" if method == "sobol_total_order" else "Dynamic sensitivity screening"
    ax.set_title(title)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Parameter importance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
