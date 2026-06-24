# ABM Extension


### 1. Bounded Rationality: Finite Subjective Memory 
In the original model, robots act with perfect global knowledge, reacting to mathematically objective probabilities of finding food ($\gamma_f$) and colliding with teammates ($\gamma_r$). 
* **The Upgrade:** Robots are no longer omniscient. Each agent is equipped with a rolling **5-trip memory buffer**. They track their active steps, collisions, and food encounters to build a strictly local, noisy, and subjective worldview. Their decisions are based purely on these imperfect historical estimates.

### 2. Cognitive Biases: Cumulative Prospect Theory 
Instead of maximizing linear energy income like perfect rational actors, the robots' decision-making is governed by Kahneman and Tversky’s Nobel Prize-winning **Prospect Theory**.
* **Reference Dependence:** Agents evaluate the success of a foraging trip relative to their current internal energy, not absolute wealth.
* **Loss Aversion:** The psychological "pain" of an empty-handed trip or excessive collisions is multiplied by a loss-aversion factor ($\lambda = 2.25$). 
* **Risk Attitude Shifts:** "Full" robots become highly risk-averse, opting for longer rest times to protect gains. "Starving" or frustrated robots become highly risk-seeking, cutting rests short to desperately forage.

### 3. Learning & Adaptation: Roth-Erev Reinforcement Learning 
The original paper noted that tuning individual parameters like resting time ($\tau_r$) relied on a trial and error process. 
* **The Upgrade:** We implemented **Roth-Erev Reinforcement Learning**. Robots no longer use a static $\tau_r$. Instead, they test an array of resting strategies (ranging from aggressive 20s rests to conservative 160s rests). The Prospect Theory subjective utility serves as the internal reward signal, updating their propensity scores. Over time, the swarm naturally evolves into segmented behavioral niches.

### 4. Strategic Interaction: The El Farol Congestion Game (Requirement 8)
We transformed physical space limitations into a strategic congestion game, modeled after the famous **El Farol Bar Problem** (Minority Game).
* **The Upgrade:** Foraging is only highly profitable when the arena is empty. When an agent's rest timer expires, it calculates the Subjective Expected Value (EV) of waking up based on its bounded memory. If it predicts a high-traffic "rush hour" with negative EV, it executes a strategic **Snooze**, opting to stay home and wait out the congestion rather than blindly colliding with the swarm.

### 5. Robust Global Sensitivity Analysis: Dynamic Time-Series Sobol' (Requirement 11)
Standard analysis in the original paper uses a One-At-a-Time (OAT) parameter sweep (Figure 8). 
* **The Upgrade:** We execute a **Variance-Based Global Sensitivity Analysis (GSA)** using Saltelli sampling and Sobol' Indices via `SALib`. Crucially, this is a *Dynamic Time-Series Analysis*. The `sensitivity.py` script sweeps the cognitive parameters (Loss Aversion, Learning Rate, Utility Linearity, Congestion Tolerance) and plots their Total-Order Sobol Indices ($S_T$) over the simulation's lifespan, proving mathematically which psychological traits drive swarm efficiency at different phases of exploration.


* **Figure 11 (Strategy Evolution):** *[NEW]* Reaches into the agents' Roth-Erev learning arrays to plot how their propensities for different resting strategies evolve over time.
* **Figure 12 (Physics vs. Psychology Divergence):** *[NEW]* Directly compares the perfectly rational, equation-based `MacroModel` control group against the new `MicroModel` Neuro-Economic swarm, visualizing the macro-level cost of psychological biases.
* **Dynamic Sobol' Plot:** Shows the changing variance importance of cognitive parameters over the 20,000-second simulation.

---
