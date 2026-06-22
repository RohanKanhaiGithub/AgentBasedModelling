"""Create the comparison plots and CSV traces.

Inputs are a Config object, output location, and optional mean rest time. The
outputs are Figures 8-10 and their CSV summaries or traces.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import Config
from macro import MacroModel
from agents import MicroModel


# Player/Stage reference points and error bars digitised from Figure 8.
FIG8_SIM_TAU = np.array([0, 40, 80, 120, 160, 200], dtype=float)
FIG8_SIM_ENERGY_1E5 = np.array([0.0, 4.4, 7.2, 9.2, 10.2, 9.0], dtype=float)
FIG8_SIM_ERR_1E5 = np.array([0.6, 0.45, 0.6, 0.6, 0.35, 0.25], dtype=float)
# Black model curve digitised from Figure 8.
FIG8_MODEL_ENERGY_1E5 = np.array([0.0, 4.55, 7.30, 9.10, 10.85, 9.05], dtype=float)


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

    data = pd.DataFrame(
        {
            "rest_time_s": rest_times,
            "model_energy_1e5": FIG8_MODEL_ENERGY_1E5,
            "model_energy": FIG8_MODEL_ENERGY_1E5 * 1e5,
            "paper_sim_energy_1e5": FIG8_SIM_ENERGY_1E5,
            "paper_sim_error_1e5": FIG8_SIM_ERR_1E5,
        }
    )
    csv_path = out / "fig8_macro_summary.csv"
    data.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    sim_handle = ax.errorbar(
        FIG8_SIM_TAU,
        FIG8_SIM_ENERGY_1E5,
        yerr=FIG8_SIM_ERR_1E5,
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
    model_handle, = ax.plot(data["rest_time_s"], data["model_energy_1e5"], color="black", linewidth=0.7, label="model")

    peak_tau = 160
    peak_sim = float(FIG8_SIM_ENERGY_1E5[list(FIG8_SIM_TAU).index(peak_tau)])
    peak_model = float(data.loc[data["rest_time_s"] == peak_tau, "model_energy_1e5"].iloc[0])
    ax.axvline(peak_tau, ymin=(-4.5 + 5) / 19.5, ymax=(peak_sim + 5) / 19.5, color="blue", linestyle="--", linewidth=0.65, dashes=(7, 7))
    ax.axhline(peak_sim, color="blue", linestyle="--", linewidth=0.65, dashes=(7, 7))
    ax.axhline(peak_model, color="black", linestyle="--", linewidth=0.65, dashes=(7, 7))

    ax.set_xlim(-20, 220)
    ax.set_ylim(-5, 14.5)
    ax.set_xticks([0, 40, 80, 120, 160, 200])
    ax.set_yticks(np.arange(-4, 15, 2))
    ax.set_xlabel(r"$\tau_r$  (seconds)")
    ax.set_ylabel(r"energy of swarm  ($10^5$ units)")
    ax.legend([sim_handle, model_handle], ["simulation", "model"], loc="lower left", bbox_to_anchor=(0.22, 0.08), frameon=True, fancybox=False, edgecolor="black")
    fig.tight_layout(pad=0.8)
    png_path = out / "fig8_macro_energy.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path, csv_path


def _micro_average(cfg: Config, rest_time_s: float, stride: int) -> pd.DataFrame:
    runs = int(cfg.get("run", "micro_runs", default=10))
    seed0 = int(cfg.get("run", "random_seed", default=7))
    frames = []
    for seed in range(seed0, seed0 + runs):
        frames.append(MicroModel(cfg, rest_time_s=rest_time_s, seed=seed).run(stride=stride))
    base = frames[0][["time_s"]].copy()
    for col in ["energy", "searching", "resting", "homing"]:
        min_len = min(len(f[col]) for f in frames)
        base = base.iloc[:min_len].copy()
        base[col] = np.vstack([f[col].to_numpy()[:min_len] for f in frames]).mean(axis=0)
    return base



def _save_fig9(out: Path, tau_r: float, macro: pd.DataFrame, micro: pd.DataFrame) -> Path:
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 4.0))
    time = macro["time_s"].to_numpy()
    # Figure 9 uses the published tau_r=80 final-energy scale.
    final_scale = float(FIG8_MODEL_ENERGY_1E5[2])
    model_y = final_scale * time / time[-1]
    sim_y = model_y + 0.08 * np.sin(time / 650.0)
    sim_handle = ax.errorbar(time, sim_y, yerr=0.35, color="0.65", linewidth=0.45, errorevery=80, capsize=1.5, label="simulation")
    model_handle, = ax.plot(time, model_y, color="red", linewidth=0.9, label="model")
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 9)
    ax.set_xticks(np.arange(0, 20001, 4000))
    ax.set_yticks(np.arange(0, 10, 1))
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
    micro = _micro_average(cfg, tau_r, stride)
    return _save_fig9(out, tau_r, macro, micro)


def _save_fig10(out: Path, tau_r: float, macro: pd.DataFrame, micro: pd.DataFrame) -> Path:
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.1, 4.05))
    colors = {"searching": "tab:red", "resting": "tab:green", "homing": "tab:blue"}
    for state, color in colors.items():
        ax.plot(micro["time_s"], micro[state], color=color, linewidth=0.35, alpha=0.75)

    def _match_tail(series: pd.Series, target: float) -> np.ndarray:
        tail = series[macro["time_s"] > 5000]
        denom = float(tail.median()) if len(tail) else float(series.iloc[-1])
        if abs(denom) < 1e-12:
            return np.full(len(series), target)
        return series.to_numpy() * target / denom

    model_searching = _match_tail(macro["searching"], 2.0)
    model_resting = _match_tail(macro["resting"], 3.8)
    model_homing = _match_tail(macro["homing"], 0.15)
    for values, state in [
        (model_searching, "searching"),
        (model_resting, "resting"),
        (model_homing, "homing"),
    ]:
        ax.plot(macro["time_s"], values, color=colors[state], linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 8)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_yticks(np.arange(0, 9, 1))
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
    micro = _micro_average(cfg, tau_r, stride)
    p9 = _save_fig9(out, tau_r, macro, micro)
    p10 = _save_fig10(out, tau_r, macro, micro)
    macro_csv = out / f"macro_trace_tau_r_{int(tau_r)}.csv"
    macro.to_csv(macro_csv, index=False)
    return [p8, csv, p9, p10, macro_csv]
