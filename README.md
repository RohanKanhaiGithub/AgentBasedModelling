# Swarm foraging paper reproduction

This project implements the collective-foraging model described by Liu, Winfield and Sa. It includes aggregate and individual-agent models, comparison plots, CSV traces, and an animation.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Project files

All source files, configuration, and outputs are stored directly in the repository root.

- `cli.py`: input is a terminal command; output is the selected plot, CSV, GIF, or sensitivity files.
- `config.py`: input is `default_config.yaml`; output is a `Config` object used by the model.
- `default_config.yaml`: input is edited parameter values; output is the run configuration read by the scripts.
- `map.py`: input is the configuration; output is arena geometry, timings, and transition probabilities.
- `agents.py`: input is configuration, seed, and optional behaviour parameters; output is a sampled micro-model trace.
- `macro.py`: input is configuration and rest time; output is a sampled aggregate-model trace.
- `plot.py`: input is configuration and output folder; output is Figures 8–12 plus CSV traces.
- `plots.py`: input is an import request for `plots`; output is the plotting functions from `plot.py`.
- `animation.py`: input is configuration and GIF settings; output is `swarm_animation.gif`.
- `sensitivity.py`: input is configuration, sample count, and simulated seconds; output is a dynamic sensitivity plot and CSV.
- `emergence_analysis.py`: input is short-run emergence settings; output is the final emergence report folder.
- `requirements.txt`: input to `pip`; output is the installed Python environment.

## Agent movement

Every robot starts at a random radius and angle in the search area. Searching uses a correlated random walk: the robot keeps its previous heading, adds a random turn between `-0.25` and `+0.25` radians, and moves forward by `speed × time step`. Grabbing and avoidance use the same wandering movement. Depositing and homing also receive the random turn, then steer toward the arena centre. Resting robots move 10% of their remaining distance toward the centre on each step.

Movement coordinates are used for the animation. PFSM probabilities and timers determine food discovery, encounters, target loss, state changes, and energy.

## Command-line interface

Create Figures 8–12 and their CSV files:

```bash
python cli.py plots
```

This now also writes:

- `fig11_strategy_evolution.png`
- `fig11_strategy_evolution.csv`
- `fig12_energy_divergence.png`
- `fig12_energy_divergence.csv`

Create a macro trace with an 80-second mean rest time:

```bash
python cli.py csv --rest 80
```

Create the animation using `default_config.yaml`:

```bash
python cli.py animation
```

Run the sensitivity analysis:

```bash
python cli.py sensitivity
```

If `SALib` is installed, this command writes Sobol total-order indices. If it is not installed, it still runs and labels the output as a lightweight screening analysis.

Use shorter or longer sensitivity runs:

```bash
python cli.py sensitivity --seconds 3000 --samples 12 --sample-every 100
```

Set the simulated time, playback duration, frame rate, and rest time:

```bash
python cli.py animation --seconds 6000 --playback-seconds 45 --fps 15 --rest 80
```

Create every output:

```bash
python cli.py all
```

Options:

- `--config FILE` chooses the YAML file. The default is `default_config.yaml`.
- `--out PATH` chooses an output directory or a `.gif` path for `animation`.
- `--rest SECONDS` sets the mean rest time.
- `--seconds SECONDS` sets the simulated animation time.
- `--samples N` sets the sensitivity-analysis base sample count.
- `--sample-every SECONDS` sets the sensitivity-analysis sampling interval.
- `--playback-seconds SECONDS` sets the approximate GIF duration.
- `--fps FPS` sets the animation frame rate.

## Output files

- `fig8_macro_energy.png`
- `fig8_macro_summary.csv`
- `fig9_energy_tau_r_80.png`
- `fig10_states_tau_r_80.png`
- `fig11_strategy_evolution.png`
- `fig11_strategy_evolution.csv`
- `fig12_energy_divergence.png`
- `fig12_energy_divergence.csv`
- `macro_trace_tau_r_80.csv`
- `swarm_animation.gif`
- `fig_dynamic_sensitivity.png`
- `sensitivity_dynamic.csv`

The GIF color key uses blue for searching, orange for grabbing, green for depositing, red for homing, purple for resting, brown for avoidance, and gold stars for food.

## Emergence analysis

The emergence script uses short runs, repeated seeds, and parameter sweeps. The final report focuses on three defensible patterns: finite-size scaling, emergent search/rest allocation, and percolation-like contact-network growth.

Run the final emergence report:

```bash
python emergence_analysis.py --final
```

Useful options:

- `--duration 3000` sets the simulated seconds per run.
- `--seeds 5` sets the number of repeated seeds.
- `--sample-every 1.0` samples once per simulated second.
- `--out results/emergence_final` chooses the output folder.

Final emergence outputs:

- `results/emergence_final/fig_1_scaling_law_collection_vs_N.png`
- `results/emergence_final/fig_2_search_rest_ratio_energy.png`
- `results/emergence_final/fig_3_contact_network_percolation.png`
- `results/emergence_final/scaling_law_collection_vs_N.csv`
- `results/emergence_final/search_rest_ratio_energy.csv`
- `results/emergence_final/contact_network_percolation.csv`
- `results/emergence_final/emergence_summary.txt`

The final analysis avoids unsupported claims. It reports finite-size scaling, power-law-like scaling over the tested range when supported by the fitted exponents, and percolation-like contact-network growth. Avalanche, jamming, and gamma_r ablation figures are left out of the final report because they are weaker evidence under short simulation runs.
