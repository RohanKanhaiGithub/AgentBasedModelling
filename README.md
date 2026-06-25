# Swarm Foraging Model

This folder now contains one combined model. It keeps Model A's learning,
energy accounting, risk-sensitive rest decisions, and game-theory-style strategy
selection, then adds Model R's sector-wise food-memory strategy to the same
agent pipeline.

The movement logic is intentionally left close to the working base models:
robots wander while searching, move toward assigned food while grabbing, and
turn back toward the nest while depositing or homing.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Files

- `cli.py`: simple command-line entry point.
- `default_config.yaml`: paper parameters and run settings.
- `config.py`: loads YAML into a typed `Config` object.
- `map.py`: computes arena geometry, timers, and transition probabilities.
- `agents.py`: individual robot PFSM, sector strategy, learning, and energy.
- `macro.py`: aggregate timer-queue version of the paper model.
- `plot.py`: static figures and CSV traces.
- `animation.py`: GIF animation for the micro model.
- `sensitivity.py`: dynamic parameter sensitivity analysis.
- `requirements.txt`: Python package dependencies.

Generated figures, CSVs, GIFs, caches, and the old duplicate base-model folders
are not needed for the source model.

## Run

Create all static plots and CSV traces:

```bash
python cli.py plots
```

Write one macro CSV:

```bash
python cli.py csv --rest 80
```

Create the animation:

```bash
python cli.py animation
```

Run the sensitivity analysis:

```bash
python cli.py sensitivity --seconds 3000 --samples 12 --sample-every 100
```

Create plots plus animation:

```bash
python cli.py all
```

Common options:

- `--config FILE`: YAML config path. Default: `default_config.yaml`.
- `--out PATH`: output directory, or GIF path for `animation`.
- `--rest SECONDS`: override the rest time used by macro/animation runs.
- `--seconds SECONDS`: simulated seconds for animation or sensitivity.
- `--playback-seconds SECONDS`: approximate GIF playback duration.
- `--fps FPS`: GIF frame rate.
- `--samples N`: base sample count for sensitivity.
- `--sample-every SECONDS`: sensitivity sampling interval.

## Agent Decision Pipeline

The full individual-agent logic lives in `agents.py`.

1. `MicroModel.step()` starts each tick by counting robot states, computing food
   discovery probability `gamma_f`, collision probability `gamma_r`, and food
   loss probability `gamma_l`.
2. Each robot updates its display position. This affects animation and sector
   steering, but the PFSM probabilities still come from the paper equations.
3. Energy cost is charged. Resting robots pay `resting_cost`; all other states
   pay `active_cost`.
4. The robot advances through its current PFSM state:
   searching, grabbing, depositing, homing, avoidance, or resting.
5. Food count and swarm energy are updated after all robots finish the tick.

## Sector-Wise Strategy

Model R's sector idea is now part of each `Agent` through `sector_scores`.

- `get_sector(x, y)` maps a robot's position into one of four arena sectors.
- During unrewarded search, `_cool_current_sector()` slowly reduces the current
  sector score. This keeps robots from overcommitting to a poor patch.
- When grabbing completes successfully, `_reward_current_sector()` increases the
  score for the sector where that food was found.
- While searching, `_steer_toward_best_sector()` gently bends the robot's
  heading toward its best-scoring sector.

This is a movement bias, not a replacement for the PFSM. Robots still discover
food, collide, lose targets, deposit, home, and rest through the same stochastic
state machine.

## Learning, Game Theory, And Risk

Model A's strategy layer is kept in the rest transition because that is the
natural point where a robot has completed a trip and has evidence to evaluate.

- `_step_resting()` implements the immediate go/search-versus-wait decision. It
  uses recent subjective collision and food encounter memories. If congestion is
  above `congestion_tolerance` and the expected reward is lower than expected
  active plus collision cost, the robot may snooze instead of waking.
- `_go_resting()` records trip experience after deposit or homing. It updates
  each robot's recent `gamma_r` and `gamma_f` memories.
- Risk aversion is applied in `_go_resting()` through `alpha`. The trip energy
  delta is converted into subjective utility with diminishing sensitivity.
- Loss aversion is also applied in `_go_resting()` through `lambda_loss`. A
  negative trip delta is weighted more strongly than an equally sized gain.
- Strategy learning happens after that utility calculation. The selected rest
  strategy's propensity is updated with recency weighting, then the next rest
  strategy is sampled from normalized propensities.

In short: sector memory shapes where a searching robot drifts, the PFSM decides
what happens to it, the trip outcome updates risk/loss-sensitive utility, and
the strategy propensities decide how long the robot rests next.

## Outputs

The main commands can write:

- `fig8_macro_energy.png` and `fig8_macro_summary.csv`
- `fig9_energy_tau_r_80.png`
- `fig10_states_tau_r_80.png`
- `fig11_strategy_evolution.png` and `fig11_strategy_evolution.csv`
- `fig12_energy_divergence.png` and `fig12_energy_divergence.csv`
- `macro_trace_tau_r_80.csv`
- `swarm_animation.gif`
- `fig_dynamic_sobol.png` and `dynamic_sobol_indices.csv`

The animation color key uses blue for searching, orange for grabbing, green for
depositing, red for homing, purple for resting, brown for avoidance, and gold
stars for displayed food.
