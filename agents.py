"""Model individual robots with a stochastic PFSM.

Inputs are a Config object, an optional mean rest time, random seed, duration,
and sampling stride. A run returns a pandas DataFrame with state counts, food,
energy, transition rates, and optional positions for the animation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import pandas as pd

from config import Config
from map import PaperMap


@dataclass
class Agent:
    state: str = "searching"
    timer: int = 0
    search_credit: int = 0
    return_state: str = ""
    return_timer: int = 0
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    target_x: float | None = None
    target_y: float | None = None


class MicroModel:
    """Track each robot's state, timers, energy contribution, and display position."""

    def __init__(self, cfg: Config, rest_time_s: float | None = None, seed: int | None = None):
        self.cfg = cfg
        self.world = PaperMap.from_config(cfg)
        seed_value = seed if seed is not None else int(cfg.get("run", "random_seed", default=1))
        self.rng = random.Random(seed_value)
        self.food_rng = random.Random(seed_value + 1)
        self.rest_time_s = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))
        self.ts = self.world.steps(float(cfg.get("behaviour", "search_time_s")))
        self.ta = self.world.steps(float(cfg.get("behaviour", "avoidance_time_s")))
        self.tr = self.world.steps(self.rest_time_s)
        self.tg = self.world.steps(self.world.tau_grab)
        self.td = self.world.steps(self.world.tau_deposit)
        self.th = self.world.steps(self.world.tau_home)
        self.food = float(cfg.get("food", "initial_count"))
        self.energy = 0.0
        self.resting_cost = float(cfg.get("energy", "resting_cost"))
        self.active_cost = float(cfg.get("energy", "active_cost"))
        self.food_reward = float(cfg.get("energy", "food_reward"))
        self.agents = [self._new_agent() for _ in range(self.world.n_robots)]
        self.food_positions: list[tuple[float, float]] = []
        self._sync_food_positions()

    def _new_agent(self) -> Agent:
        r = self.rng.uniform(self.world.rinner, self.world.router)
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        return Agent("searching", self.ts, self.ts, "", 0, r * math.cos(theta), r * math.sin(theta), theta)

    def run(self, seconds: float | None = None, stride: int = 1, keep_frames: bool = False) -> pd.DataFrame:
        steps = self.world.steps(float(seconds if seconds is not None else self.cfg.duration_s))
        rows: list[dict[str, float]] = []
        stride = max(1, int(stride))
        for step in range(steps):
            stats = self.step()
            if step % stride == 0 or step == steps - 1:
                stats["time_s"] = (step + 1) * self.world.dt
                stats["rest_time_s"] = self.rest_time_s
                stats["energy"] = self.energy
                stats["food_items"] = self.food
                if keep_frames:
                    stats["positions"] = [(a.x, a.y, a.state) for a in self.agents]
                    stats["food_positions"] = list(self.food_positions)
                rows.append(stats)
        return pd.DataFrame(rows)

    def step(self) -> dict[str, float]:
        counts = self.counts()
        active = self.world.n_robots - counts["resting"]
        gamma_f = self.world.find_probability(self.food)
        gamma_r = self.world.collision_probability(active)
        competing = counts["searching"] + counts["grabbing"] + counts["avoidance"]
        gamma_l = self.world.loss_probability(self.food, competing, self.tg)

        if gamma_f + gamma_r > 0.98:
            scale = 0.98 / (gamma_f + gamma_r)
            gamma_f *= scale
            gamma_r *= scale
        if gamma_l + gamma_r > 0.98:
            scale = 0.98 / (gamma_l + gamma_r)
            gamma_l *= scale
            gamma_r *= scale

        entered_deposit = 0
        completed_deposit = 0

        for agent in self.agents:
            self._update_display_position(agent)

            if agent.state == "searching":
                entered_deposit += self._step_searching(agent, gamma_f, gamma_r)

            elif agent.state == "grabbing":
                entered_deposit += self._step_grabbing(agent, gamma_l, gamma_r)

            elif agent.state == "deposit":
                completed_deposit += self._step_deposit(agent, gamma_r)

            elif agent.state == "homing":
                self._step_homing(agent, gamma_r)

            elif agent.state == "avoidance":
                done_deposit = self._step_avoidance(agent)
                completed_deposit += done_deposit

            elif agent.state == "resting":
                self._step_resting(agent)

        self.food = max(0.0, self.food + self.world.p_new_per_step - entered_deposit)
        self._sync_food_positions()
        counts = self.counts()
        active = self.world.n_robots - counts["resting"]
        self.energy += self.food_reward * completed_deposit - self.world.dt * (
            self.resting_cost * counts["resting"] + self.active_cost * active
        )
        counts["gamma_f"] = gamma_f
        counts["gamma_r"] = gamma_r
        counts["gamma_l"] = gamma_l
        counts["entered_deposit"] = float(entered_deposit)
        counts["completed_deposit"] = float(completed_deposit)
        return counts

    def _step_searching(self, agent: Agent, gamma_f: float, gamma_r: float) -> int:
        if agent.search_credit <= 0:
            self._go_homing(agent)
            return 0
        u = self.rng.random()
        if u < gamma_r:
            self._go_avoidance(agent, "searching", agent.timer)
        elif u < gamma_r + gamma_f:
            agent.state = "grabbing"
            agent.timer = self.tg
            self._assign_food_target(agent)
        else:
            agent.search_credit -= 1
            agent.timer = agent.search_credit
            if agent.search_credit <= 0:
                self._go_homing(agent)
        return 0

    def _step_grabbing(self, agent: Agent, gamma_l: float, gamma_r: float) -> int:
        if agent.search_credit <= 0:
            self._go_homing(agent)
            return 0
        u = self.rng.random()
        if u < gamma_r:
            self._go_avoidance(agent, "grabbing", agent.timer)
            return 0
        if u < gamma_r + gamma_l:
            self._clear_food_target(agent)
            agent.state = "searching"
            agent.timer = max(0, agent.search_credit)
            return 0
        agent.search_credit -= 1
        agent.timer -= 1
        if agent.search_credit <= 0:
            self._go_homing(agent)
        elif agent.timer <= 0:
            self._consume_food_target(agent)
            agent.state = "deposit"
            agent.timer = self.td
            return 1
        return 0

    def _step_deposit(self, agent: Agent, gamma_r: float) -> int:
        if self.rng.random() < gamma_r:
            self._go_avoidance(agent, "deposit", agent.timer)
            return 0
        agent.timer -= 1
        if agent.timer <= 0:
            self._go_resting(agent)
            return 1
        return 0

    def _step_homing(self, agent: Agent, gamma_r: float) -> None:
        if self.rng.random() < gamma_r:
            self._go_avoidance(agent, "homing", agent.timer)
            return
        agent.timer -= 1
        if agent.timer <= 0:
            self._go_resting(agent)

    def _step_avoidance(self, agent: Agent) -> int:
        # The interrupted timer keeps running; it is not reset when avoidance ends.
        agent.timer -= 1
        if agent.return_state in {"searching", "grabbing"}:
            agent.search_credit = max(0, agent.search_credit - 1)
        if agent.return_state in {"grabbing", "deposit", "homing"}:
            agent.return_timer -= 1

        if agent.timer > 0:
            return 0

        previous = agent.return_state
        return_timer = agent.return_timer
        agent.return_state = ""
        agent.return_timer = 0

        if previous == "searching":
            if agent.search_credit <= 0:
                self._go_homing(agent)
            else:
                agent.state = "searching"
                agent.timer = agent.search_credit
            return 0

        if previous == "grabbing":
            if agent.search_credit <= 0:
                self._go_homing(agent)
                return 0
            if return_timer <= 0:
                self._consume_food_target(agent)
                agent.state = "deposit"
                agent.timer = self.td
                return 1
            agent.state = "grabbing"
            agent.timer = return_timer
            return 0

        if previous == "deposit":
            if return_timer <= 0:
                self._go_resting(agent)
                return 1
            agent.state = "deposit"
            agent.timer = return_timer
            return 0

        if previous == "homing":
            if return_timer <= 0:
                self._go_resting(agent)
            else:
                agent.state = "homing"
                agent.timer = return_timer
            return 0

        agent.state = "searching"
        agent.timer = max(0, agent.search_credit)
        return 0

    def _step_resting(self, agent: Agent) -> None:
        agent.timer -= 1
        if agent.timer <= 0:
            agent.state = "searching"
            agent.search_credit = self.ts
            agent.timer = self.ts

    def _go_avoidance(self, agent: Agent, previous: str, previous_timer: int) -> None:
        agent.return_state = previous
        agent.return_timer = previous_timer
        agent.state = "avoidance"
        agent.timer = self.ta

    def _go_homing(self, agent: Agent) -> None:
        self._clear_food_target(agent)
        agent.state = "homing"
        agent.timer = self.th
        agent.return_state = ""
        agent.return_timer = 0

    def _go_resting(self, agent: Agent) -> None:
        self._clear_food_target(agent)
        agent.state = "resting"
        agent.timer = self._rest_timer()
        agent.search_credit = 0
        agent.return_state = ""
        agent.return_timer = 0

    def _rest_timer(self) -> int:
        return max(0, self.tr)

    def counts(self) -> dict[str, float]:
        names = ["searching", "grabbing", "deposit", "homing", "resting", "avoidance"]
        out = {name: 0.0 for name in names}
        for agent in self.agents:
            out[agent.state] += 1.0
        return out

    def _update_display_position(self, agent: Agent) -> None:
        """Move one animation marker without changing PFSM transition rates."""
        if agent.state == "resting":
            agent.x *= 0.9
            agent.y *= 0.9
            return

        random_turn = self.rng.uniform(-0.25, 0.25)
        step = self.world.speed * self.world.dt

        if agent.state == "grabbing":
            self._move_toward_food(agent, step)
            return

        agent.heading += random_turn
        if agent.state in {"deposit", "homing"} or agent.return_state in {"deposit", "homing"}:
            target = math.atan2(-agent.y, -agent.x)
            agent.heading = 0.85 * agent.heading + 0.15 * target
        agent.x += step * math.cos(agent.heading)
        agent.y += step * math.sin(agent.heading)
        radius = math.hypot(agent.x, agent.y)
        if radius > self.world.router:
            agent.heading += math.pi
            scale = self.world.router / max(radius, 1e-9)
            agent.x *= scale
            agent.y *= scale

    def _move_toward_food(self, agent: Agent, step: float) -> None:
        if agent.target_x is None or agent.target_y is None:
            self._assign_food_target(agent)
        if agent.target_x is None or agent.target_y is None:
            return

        dx = agent.target_x - agent.x
        dy = agent.target_y - agent.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-12:
            return

        agent.heading = math.atan2(dy, dx)
        travel = min(step, distance)
        agent.x += travel * math.cos(agent.heading)
        agent.y += travel * math.sin(agent.heading)

    def _assign_food_target(self, agent: Agent) -> None:
        self._sync_food_positions()
        if not self.food_positions:
            return
        target = min(self.food_positions, key=lambda point: math.hypot(point[0] - agent.x, point[1] - agent.y))
        agent.target_x, agent.target_y = target

    def _clear_food_target(self, agent: Agent) -> None:
        agent.target_x = None
        agent.target_y = None

    def _consume_food_target(self, agent: Agent) -> None:
        if agent.target_x is not None and agent.target_y is not None:
            target = (agent.target_x, agent.target_y)
            shared = any(
                other is not agent and (other.target_x, other.target_y) == target
                for other in self.agents
            )
            if not shared and target in self.food_positions:
                self.food_positions.remove(target)
        self._clear_food_target(agent)

    def _sync_food_positions(self) -> None:
        desired = int(math.ceil(max(0.0, self.food) - 1e-12))
        while len(self.food_positions) < desired:
            radius = math.sqrt(
                self.food_rng.uniform(self.world.rinner**2, self.world.router**2)
            )
            angle = self.food_rng.uniform(0.0, 2.0 * math.pi)
            self.food_positions.append((radius * math.cos(angle), radius * math.sin(angle)))

        targeted = {
            (agent.target_x, agent.target_y)
            for agent in self.agents
            if agent.target_x is not None and agent.target_y is not None
        }
        index = len(self.food_positions) - 1
        while len(self.food_positions) > desired and index >= 0:
            if self.food_positions[index] not in targeted:
                self.food_positions.pop(index)
            index -= 1
