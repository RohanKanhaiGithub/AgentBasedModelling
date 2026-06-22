# Swarm foraging paper reproduction

This project implements the collective-foraging model described by Liu, Winfield and Sa. It includes an aggregate swarm model, an individual-agent model, comparison plots, CSV traces, and an animation.

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment in the project folder:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Project files

All project files and outputs are stored directly in the repository root.

- `cli.py` receives terminal commands and options. It calls the selected plot, CSV, or animation function and prints each output path.
- `config.py` reads `default_config.yaml` and returns a `Config` object used by the models.
- `default_config.yaml` provides the arena, robot, energy, food, behaviour, random seed, and run settings.
- `map.py` receives the configuration and returns arena geometry, state durations, and transition probabilities.
- `macro.py` receives the configuration, rest time, duration, and sampling stride. It returns a table of population states, food, energy, and transition rates.
- `agents.py` receives the configuration, rest time, seed, duration, and sampling stride. It returns a table of individual-agent results and optional animation positions.
- `plot.py` receives the configuration and output path. It writes Figures 8–10 and the CSV summaries or traces.
- `animation.py` receives the configuration, output path, simulated time, playback time, frame rate, and rest time. It writes `swarm_animation.gif`.
- `requirements.txt` lists the Python packages required by the project.

## Agent movement

Every robot begins in the searching state at a random radius and angle in the search area.

Searching follows a **correlated random walk**. On each simulation step, the robot keeps its current heading and adds a uniformly sampled turn between `-0.25` and `+0.25` radians, approximately `-14.3°` to `+14.3°`. It then moves forward by `speed × time step`. Keeping the previous heading produces a gradually curving path instead of an unrelated direction on every step.

When a robot enters the grabbing state, it selects the nearest displayed food star. It points directly at that food position and moves along a straight line at the configured speed. It stops at the target if it arrives before the grabbing timer finishes. If the target is lost, the robot clears the target and returns to searching. If grabbing succeeds, the target disappears and the robot starts depositing.

Avoidance temporarily interrupts searching, grabbing, depositing, or homing while preserving the interrupted timer. Depositing and homing robots steer toward the arena centre. Resting robots move 10% of their remaining distance toward the centre on each step. A robot that reaches the outer boundary reverses its heading and is placed back on the boundary.

The coordinates control the animation, while the PFSM probabilities and timers control food discovery, robot encounters, target loss, state changes, and energy. Food remains a swarm-level count; the stars provide positions for displaying available food and straight grabbing movement.

The state transitions are:

1. **Searching** changes to **grabbing** when food is found, **avoidance** after a robot encounter, or **homing** when its search credit ends.
2. **Grabbing** changes to **deposit** after a successful pickup, **avoidance** after an encounter, or **searching** if the target is lost.
3. **Avoidance** returns to the interrupted state after its timer finishes, unless that state's timer has already completed.
4. **Deposit** and **homing** both lead to **resting**.
5. **Resting** returns to **searching** after the configured rest time.

The GIF includes an on-screen color key: blue for searching, orange for grabbing, green for depositing, red for homing, purple for resting, brown for avoidance, and gold stars for food.

## Command-line interface

Run commands from the project folder while the virtual environment is active.

Create Figures 8–10 and their CSV files:

```bash
python cli.py plots
```

Create the macro trace with an 80-second mean rest time:

```bash
python cli.py csv --rest 80
```

Create the animation using `default_config.yaml`:

```bash
python cli.py animation
```

Choose the simulated time, playback time, frame rate, and mean rest time:

```bash
python cli.py animation --seconds 6000 --playback-seconds 45 --fps 15 --rest 80
```

Create every output:

```bash
python cli.py all
```

Available options:

- `--config FILE` chooses the YAML input file. The default is `default_config.yaml`.
- `--out PATH` chooses an output directory. For `animation`, it can also be a `.gif` path.
- `--rest SECONDS` sets the mean rest time.
- `--seconds SECONDS` sets the simulated time shown in the animation.
- `--playback-seconds SECONDS` sets the approximate GIF playback duration.
- `--fps FPS` sets the animation frame rate.

Run `python cli.py --help` for the command summary.

## Output files

Commands write these files directly to the project folder by default:

- `fig8_macro_energy.png`
- `fig8_macro_summary.csv`
- `fig9_energy_tau_r_80.png`
- `fig10_states_tau_r_80.png`
- `macro_trace_tau_r_80.csv`
- `swarm_animation.gif`
