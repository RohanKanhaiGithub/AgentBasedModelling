### How Strategy Weights are Updated: Roth-Erev Learning & Prospect Theory

To understand exactly how the weights (referred to as **propensities**) are updated after every single run, we must examine the intersection of the **Roth-Erev Reinforcement Learning algorithm** and the **Cumulative Prospect Theory** utility function. 

In this neuro-economic model, an agent does not passively track how much energy it gained; it "feels" the outcome psychologically and uses that subjective utility to update its internal scorecard. Here is the exact step-by-step mathematical process of how the weights shift after a robot completes a trip and returns to the nest.

---

#### Step 1: The Initial State (The Scorecard)
Before the robot has learned anything, it holds a portfolio of four strategies. It assigns an arbitrary, equal starting weight (propensity) to each one. Let's assume the starting weight is $10.0$.
* **Strategy 0 (20s):** $q_0 = 10.0$
* **Strategy 1 (60s):** $q_1 = 10.0$
* **Strategy 2 (100s):** $q_2 = 10.0$
* **Strategy 3 (160s):** $q_3 = 10.0$

Because all weights are equal, the robot has exactly a 25% chance to pick any of them. Let's assume the robot runs a probability selection and randomly selects **Strategy 0 (20s)** for its next rest.

#### Step 2: The Physical Trip and the Objective Outcome
The robot rests for 20 seconds, wakes up, and deploys to the arena to forage. Because 20 seconds is a very short resting period, the arena is highly congested. The robot gets trapped in several collision avoidance loops. 

When it returns to the nest, the physics engine calculates its absolute net energy change ($x$). Because it wasted battery avoiding collisions rather than efficiently foraging, it returns with an absolute energy loss: 
**$x = -5.0$ units**.

#### Step 3: Calculating the Psychological Reward (Prospect Theory)
The Roth-Erev algorithm requires a "Reward Signal" to update the strategy weights. In a perfectly rational base model, the reward would simply be $-5.0$. However, our robots possess cognitive biases.

The robot passes $x = -5.0$ through the **Cumulative Prospect Theory** value function:
$$V(x) = -\lambda \cdot (-x)^\alpha$$
*(Assuming Loss Aversion $\lambda = 2.25$ and Risk Sensitivity $\alpha = 0.88$)*

$$V(-5) = -2.25 \cdot (5)^{0.88} \approx -2.25 \cdot 4.11 = \mathbf{-9.25}$$

Instead of evaluating a $-5.0$ loss, the robot *psychologically experiences* a **$-9.25$ utility penalty** because it is heavily loss-averse. This $-9.25$ is the actual subjective reward signal fed into the learning engine.

#### Step 4: The Roth-Erev Update Equation
Next, the robot opens its strategy scorecard to update the weights. The Roth-Erev equation utilizes a **Recency Parameter ($\rho$)**, typically set around $0.05$ (or 5%). This parameter introduces a "forgetting" mechanism—it slightly decays old memories so the robot can adapt to dynamically changing spatial conditions.

**For the strategy that was JUST played (Strategy 0):**
The new weight is the old weight minus the forgotten fraction, *plus* the new psychological reward:
$$q_{\text{played}}(t) = (1 - \rho) \cdot q_{\text{played}}(t-1) + \text{Reward}$$
$$q_0 = (1 - 0.05) \cdot 10.0 + (-9.25)$$
$$q_0 = 9.5 - 9.25 = \mathbf{0.25}$$

**For the strategies that were NOT played (Strategies 1, 2, 3):**
They do not receive the reward signal, but they still experience the 5% "forgetting" decay to prevent historically successful weights from growing to infinity over a 20,000-second timeline:
$$q_{\text{unplayed}}(t) = (1 - \rho) \cdot q_{\text{unplayed}}(t-1)$$
$$q_{1,2,3} = (1 - 0.05) \cdot 10.0 = \mathbf{9.50}$$

#### Step 5: Recalculating the Probabilities (Roulette Wheel Selection)
Now observe the robot's updated scorecard:
* **Strategy 0 (20s):** $q_0 = 0.25$
* **Strategy 1 (60s):** $q_1 = 9.50$
* **Strategy 2 (100s):** $q_2 = 9.50$
* **Strategy 3 (160s):** $q_3 = 9.50$

To determine which strategy the robot will deploy next, the algorithm converts these raw weights into selection probabilities by dividing each weight by the total sum of all current weights ($\Sigma q = 28.75$):
* **Probability of 20s:** $0.25 / 28.75 = \mathbf{0.8\%}$
* **Probability of 60s:** $9.50 / 28.75 = \mathbf{33.0\%}$
* **Probability of 100s:** $9.50 / 28.75 = \mathbf{33.0\%}$
* **Probability of 160s:** $9.50 / 28.75 = \mathbf{33.0\%}$

---

### The Macroscopic Result
After just *one* highly congested run, the loss aversion penalty impacted the robot's subjective utility so severely that the probability of it choosing the hyper-aggressive 20-second rest again plummeted from **25% down to less than 1%**. 

If we examine the **Strategy Evolution Plot (Figure 11)**, this exact mathematical process is precisely why the propensity line for the aggressive 20s strategy crashes downward almost immediately. The swarm mathematically learns to fear the energy drain of collisions, naturally allowing the safer, longer-rest strategies to absorb the probability distribution and organically regulate spatial density.
