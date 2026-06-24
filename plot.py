"""Create the comparison plots and CSV traces.

Inputs are a Config object, output location, and optional mean rest time. The
outputs are Figures 8-10 and their CSV summaries or traces.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import Config
from macro import MacroModel
from agents import MicroModel


ENERGY_SCALE = 1e5


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def _stride_steps(cfg: Config) -> int:
    stride_s = float(cfg.get("run", "csv_stride_s", default=5.0))
    return max(1, int(round(stride_s / cfg.dt)))


def _micro_frames(cfg: Config, rest_time_s: float, stride: int) -> list[pd.DataFrame]:
    runs = int(cfg.get("run", "micro_runs", default=10))
    seed0 = int(cfg.get("run", "random_seed", default=7))
    return [
        MicroModel(cfg, rest_time_s=rest_time_s, seed=seed).run(stride=stride)
        for seed in range(seed0, seed0 + runs)
    ]


def _micro_mean_and_std(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not frames:
        raise ValueError("at least one micro simulation run is required")

    min_len = min(len(frame) for frame in frames)
    mean = frames[0][["time_s", "rest_time_s"]].iloc[:min_len].copy()
    std = mean.copy()
    skip = {"time_s", "rest_time_s", "positions", "food_positions"}
    value_cols = [col for col in frames[0].columns if col not in skip]

    for col in value_cols:
        values = np.vstack([frame[col].to_numpy()[:min_len] for frame in frames])
        mean[col] = values.mean(axis=0)
        std[col] = values.std(axis=0, ddof=1) if len(frames) > 1 else 0.0
    return mean, std


def _energy_1e5(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return values / ENERGY_SCALE


def _set_energy_limits(ax, *series: pd.Series | np.ndarray) -> None:
    values = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    lower = min(0.0, float(values.min()))
    upper = max(0.0, float(values.max()))
    margin = 1.0 if abs(upper - lower) < 1e-12 else 0.08 * (upper - lower)
    ax.set_ylim(lower - margin, upper + margin)


def write_macro_csv(cfg: Config, out_dir: str | Path, rest_time_s: float | None = None) -> Path:
    out = ensure_dir(out_dir)
    model = MacroModel(cfg, rest_time_s=rest_time_s)
    result = model.run(stride=_stride_steps(cfg))
    path = out / f"macro_trace_tau_r_{int(model.rest_time_s)}.csv"
    result.trace.to_csv(path, index=False)
    return path


def plot_fig8(cfg: Config, out_dir: str | Path) -> tuple[Path, Path]:
    """Write the Figure 8 energy sweep as a PNG and CSV file."""
    _paper_style()
    out = ensure_dir(out_dir)
    rest_times = [float(x) for x in cfg.get("behaviour", "rest_times_s")]
    if not rest_times:
        raise ValueError("behaviour.rest_times_s must contain at least one rest time")
    stride = _stride_steps(cfg)
    rows = []

    for tau_r in rest_times:
        macro = MacroModel(cfg, rest_time_s=tau_r).run(stride=stride).trace
        micro_frames = _micro_frames(cfg, tau_r, stride)
        micro_final_energy = np.array([frame["energy"].iloc[-1] for frame in micro_frames], dtype=float)
        micro_final_std = float(micro_final_energy.std(ddof=1)) if len(micro_final_energy) > 1 else 0.0
        rows.append(
            {
                "rest_time_s": tau_r,
                "model_energy": float(macro["energy"].iloc[-1]),
                "model_energy_1e5": float(_energy_1e5(macro["energy"].iloc[-1])),
                "simulation_energy": float(micro_final_energy.mean()),
                "simulation_energy_1e5": float(_energy_1e5(micro_final_energy.mean())),
                "simulation_energy_std": micro_final_std,
                "simulation_energy_std_1e5": float(_energy_1e5(micro_final_std)),
                "simulation_runs": len(micro_final_energy),
            }
        )

    data = pd.DataFrame(rows)
    csv_path = out / "fig8_macro_summary.csv"
    data.to_csv(csv_path, index=False)

    rest_time_x = data["rest_time_s"].to_numpy(dtype=float)
    sim_y = data["simulation_energy_1e5"].to_numpy(dtype=float)
    sim_err = data["simulation_energy_std_1e5"].to_numpy(dtype=float)
    model_y = data["model_energy_1e5"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    sim_handle = ax.errorbar(
        rest_time_x,
        sim_y,
        yerr=sim_err,
        fmt="o--",
        color="blue",
        ecolor="black",
        elinewidth=0.7,
        capsize=3,
        markersize=5,
        linewidth=0.7,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="simulation",
    )
    model_handle, = ax.plot(rest_time_x, model_y, color="black", linewidth=0.7, label="model")

    max_tau = max(rest_times) if rest_times else 0.0
    pad = 0.05 * max(1.0, max_tau)
    ax.set_xlim(min(0.0, min(rest_times)) - pad, max_tau + pad)
    ax.set_xticks(rest_times)
    _set_energy_limits(
        ax,
        sim_y - sim_err,
        sim_y + sim_err,
        model_y,
    )
    ax.set_xlabel(r"$\tau_r$  (seconds)")
    ax.set_ylabel(r"energy of swarm  ($10^5$ units)")
    ax.legend([sim_handle, model_handle], ["simulation", "model"], loc="lower left", bbox_to_anchor=(0.22, 0.08), frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(pad=0.8)
    png_path = out / "fig8_macro_energy.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path, csv_path


def _micro_average(cfg: Config, rest_time_s: float, stride: int) -> pd.DataFrame:
    mean, _ = _micro_mean_and_std(_micro_frames(cfg, rest_time_s, stride))
    return mean



def _save_fig9(out: Path, tau_r: float, macro: pd.DataFrame, micro: pd.DataFrame, micro_std: pd.DataFrame | None = None) -> Path:
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 4.0))
    macro_time = macro["time_s"].to_numpy(dtype=float)
    micro_time = micro["time_s"].to_numpy(dtype=float)
    macro_y = _energy_1e5(macro["energy"].to_numpy(dtype=float))
    micro_y = _energy_1e5(micro["energy"].to_numpy(dtype=float))
    micro_err = (
        _energy_1e5(micro_std["energy"].to_numpy(dtype=float))
        if micro_std is not None and "energy" in micro_std
        else None
    )
    errorevery = max(1, len(micro) // 25)
    sim_handle = ax.errorbar(
        micro_time,
        micro_y,
        yerr=micro_err,
        color="0.65",
        linewidth=0.45,
        errorevery=errorevery,
        capsize=1.5,
        label="simulation",
    )
    model_handle, = ax.plot(macro_time, macro_y, color="red", linewidth=0.9, label="model")
    max_time = max(float(macro_time[-1]), float(micro_time[-1]))
    ax.set_xlim(0, max_time)
    ax.set_xticks(np.linspace(0, max_time, 6))
    if micro_err is None:
        _set_energy_limits(ax, macro_y, micro_y)
    else:
        _set_energy_limits(ax, macro_y, micro_y - micro_err, micro_y + micro_err)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"energy of swarm  ($10^5$ units)")
    ax.legend([sim_handle, model_handle], ["simulation", "model"], loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(pad=0.8)
    path = out / f"fig9_energy_tau_r_{int(tau_r)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fig9(cfg: Config, out_dir: str | Path, rest_time_s: float | None = None) -> Path:
    out = ensure_dir(out_dir)
    tau_r = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
    stride = _stride_steps(cfg)
    macro = MacroModel(cfg, rest_time_s=tau_r).run(stride=stride).trace
    micro, micro_std = _micro_mean_and_std(_micro_frames(cfg, tau_r, stride))
    return _save_fig9(out, tau_r, macro, micro, micro_std)


def _save_fig10(out: Path, tau_r: float, macro: pd.DataFrame, micro: pd.DataFrame) -> Path:
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 4.05))
    colors = {"searching": "tab:red", "resting": "tab:green", "homing": "tab:blue"}
    macro_time = macro["time_s"].to_numpy(dtype=float)
    micro_time = micro["time_s"].to_numpy(dtype=float)
    for state, color in colors.items():
        ax.plot(micro_time, micro[state].to_numpy(dtype=float), color=color, linewidth=0.35, alpha=0.75)

    for values, state in [
        (macro["searching"], "searching"),
        (macro["resting"], "resting"),
        (macro["homing"], "homing"),
    ]:
        ax.plot(macro_time, values.to_numpy(dtype=float), color=colors[state], linestyle="--", linewidth=1.0)
    max_time = max(float(macro_time[-1]), float(micro_time[-1]))
    max_count = max(float(micro[state].max()) for state in colors)
    max_count = max(max_count, *(float(macro[state].max()) for state in colors))
    upper = max(1, int(np.ceil(max_count)))
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, upper)
    ax.set_xticks(np.linspace(0, max_time, 5))
    ax.set_yticks(np.arange(0, upper + 1, 1))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Robots")
    state_handles = [
        Line2D([0], [0], color=color, linewidth=1.2, label=state.title())
        for state, color in colors.items()
    ]
    series_handles = [
        Line2D([0], [0], color="0.3", linewidth=0.6, label="Simulation"),
        Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.0, label="Macro model"),
    ]
    state_legend = ax.legend(
        handles=state_handles,
        title="State",
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    ax.add_artist(state_legend)
    ax.legend(
        handles=series_handles,
        title="Series",
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    fig.tight_layout(pad=0.8)
    path = out / f"fig10_states_tau_r_{int(tau_r)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fig10(cfg: Config, out_dir: str | Path, rest_time_s: float | None = None) -> Path:
    out = ensure_dir(out_dir)
    tau_r = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
    stride = _stride_steps(cfg)
    macro = MacroModel(cfg, rest_time_s=tau_r).run(stride=stride).trace
    micro = _micro_average(cfg, tau_r, stride)
    return _save_fig10(out, tau_r, macro, micro)


def make_all_plots(cfg: Config, out_dir: str | Path) -> list[Path]:
    out = ensure_dir(out_dir)
    p8, csv = plot_fig8(cfg, out)
    tau_r = float(cfg.get("behaviour", "default_rest_time_s"))
    stride = _stride_steps(cfg)
    macro = MacroModel(cfg, rest_time_s=tau_r).run(stride=stride).trace
    micro, micro_std = _micro_mean_and_std(_micro_frames(cfg, tau_r, stride))
    p9 = _save_fig9(out, tau_r, macro, micro, micro_std)
    p10 = _save_fig10(out, tau_r, macro, micro)
    macro_csv = out / f"macro_trace_tau_r_{int(tau_r)}.csv"
    macro.to_csv(macro_csv, index=False)
    return [p8, csv, p9, p10, macro_csv]
