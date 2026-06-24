"""Dynamic Time-Series Sobol Sensitivity Analysis for Swarm Foraging.

This script uses Variance-Based Global Sensitivity Analysis (Sobol' Indices)
to determine how cognitive biases drive the swarm's macro-level energy efficiency
over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

from config import Config
from agents import MicroModel

def run_dynamic_sobol(config_path: str = "default_config.yaml"):
    cfg = Config.load(config_path)
    
    # 1. Define the Parameter Space for Cognitive Biases
    problem = {
        'num_vars': 4,
        'names': ['alpha', 'lambda_loss', 'recency', 'congestion_tolerance'],
        'bounds': [
            [0.5, 1.0],   # alpha: Diminishing sensitivity
            [1.0, 5.0],   # lambda_loss: Loss aversion multiplier
            [0.01, 0.20], # recency: Roth-Erev learning forgetting rate
            [0.01, 0.10]  # congestion_tolerance: El Farol snooze threshold
        ]
    }
    
    # 2. Generate Saltelli Samples
    # N=64 generates N * (2D + 2) runs. For D=4, this is 64 * 10 = 640 simulations.
    # Note: For publication-grade results, N should be > 256. 64 is good for testing.
    param_values = saltelli.sample(problem, 64)
    num_runs = len(param_values)
    print(f"Executing {num_runs} ABM runs for Global Sensitivity Analysis...")

    # We will sample the swarm's energy at 4 specific time milestones
    time_milestones_s = [5000, 10000, 15000, 20000]
    
    # Pre-allocate output matrix: rows = simulation runs, cols = time milestones
    Y = np.zeros([num_runs, len(time_milestones_s)])

    # 3. Execute the ABM for every parameter combination
    for i, params in enumerate(param_values):
        if i % 50 == 0:
            print(f"Running simulation {i}/{num_runs}...")
            
        model = MicroModel(
            cfg=cfg,
            alpha=params[0],
            lambda_loss=params[1],
            recency=params[2],
            congestion_tolerance=params[3],
            seed=42 # Lock seed to isolate variance strictly to parameters
        )
        
        # Run model headless (stride=400 captures data roughly every 100 seconds)
        trace = model.run(stride=400)
        
        # Extract energy at the requested time milestones
        for t_idx, target_time in enumerate(time_milestones_s):
            # Find the closest time step in the trace
            closest_row = trace.iloc[(trace['time_s'] - target_time).abs().argsort()[:1]]
            Y[i, t_idx] = closest_row['energy'].values[0]

    # 4. Analyze Sobol Indices Dynamically Over Time
    total_order_indices = {name: [] for name in problem['names']}
    
    print("Analyzing variance and computing Sobol indices...")
    for t_idx, time_point in enumerate(time_milestones_s):
        # Run SALib Sobol analyzer on the energy outputs for this specific time slice
        Si = sobol.analyze(problem, Y[:, t_idx], print_to_console=False)
        
        for p_idx, name in enumerate(problem['names']):
            # Append the Total-Order index (ST)
            total_order_indices[name].append(Si['ST'][p_idx])

    # 5. Plot the Dynamic Evolution of Parameter Importance
    _plot_dynamic_sobol(time_milestones_s, total_order_indices, problem['names'])


def _plot_dynamic_sobol(times, st_data, param_names):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in"
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = [
        r'$\alpha$ (Utility Linearity)', 
        r'$\lambda$ (Loss Aversion)', 
        r'Learning Rate (Recency)', 
        r'Congestion Tolerance'
    ]

    for name, color, label in zip(param_names, colors, labels):
        ax.plot(times, st_data[name], marker='o', linewidth=2.5, color=color, label=label)

    ax.set_title("Evolution of Parameter Importance (Total-Order Sobol Indices)")
    ax.set_xlabel("Simulation Time (seconds)")
    ax.set_ylabel(r"Total-Order Sobol Index ($S_T$)")
    ax.set_xticks(times)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, edgecolor='black')
    
    fig.tight_layout()
    out_path = "fig_dynamic_sobol.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity Analysis complete. Plot saved to {out_path}")


if __name__ == "__main__":
    run_dynamic_sobol()
