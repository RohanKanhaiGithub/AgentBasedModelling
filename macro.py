"""Run the aggregate swarm PFSM.

Inputs are a Config object, optional mean rest time, duration, and sampling
stride. The output contains the sampled state, food, energy, and rate trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import Config
from map import PaperMap


@dataclass
class MacroResult:
    trace: pd.DataFrame
    rest_time_s: float


class MacroModel:
    """Move robot populations through timed state queues and track food and energy."""

    def __init__(self, cfg: Config, rest_time_s: float | None = None):
        self.cfg = cfg
        self.world = PaperMap.from_config(cfg)
        self.rest_time_s = float(rest_time_s if rest_time_s is not None else cfg.get("behaviour", "default_rest_time_s"))

        self.ts = self.world.steps(float(cfg.get("behaviour", "search_time_s")))
        self.ta = self.world.steps(float(cfg.get("behaviour", "avoidance_time_s")))
        self.tr = self.world.steps(self.rest_time_s)
        self.tg = self.world.steps(self.world.tau_grab)
        self.td = self.world.steps(self.world.tau_deposit)
        self.th = self.world.steps(self.world.tau_home)

        self.initial_food = float(cfg.get("food", "initial_count"))
        self.resting_cost = float(cfg.get("energy", "resting_cost"))
        self.active_cost = float(cfg.get("energy", "active_cost"))
        self.food_reward = float(cfg.get("energy", "food_reward"))

    def run(self, seconds: float | None = None, stride: int = 1) -> MacroResult:
        seconds = float(seconds if seconds is not None else self.cfg.duration_s)
        steps = self.world.steps(seconds)
        stride = max(1, int(stride))

        search = np.zeros(self.ts + 1, dtype=float)
        search[self.ts] = self.world.n_robots
        avoid_search = np.zeros((self.ts + 1, self.ta + 1), dtype=float)
        grab = np.zeros(self.tg + 1, dtype=float)
        deposit = np.zeros(self.td + 1, dtype=float)
        homing = np.zeros(self.th + 1, dtype=float)
        resting = 0.0

        food = self.initial_food
        energy = 0.0
        rows: list[dict[str, float]] = []

        for step in range(steps):
            totals = self._totals(search, grab, deposit, homing, resting, avoid_search)
            gamma_f, gamma_r, gamma_l = self._rates(food, totals)

            next_search = np.zeros_like(search)
            next_avoid = np.zeros_like(avoid_search)
            next_grab = np.zeros_like(grab)
            next_deposit = np.zeros_like(deposit)
            next_homing = np.zeros_like(homing)

            # A zero rest time returns the population to searching immediately.
            if self.tr <= 0:
                rest_to_search = resting
            else:
                rest_to_search = resting / self.tr
            next_resting = resting - rest_to_search
            next_search[self.ts] += rest_to_search

            # A search timer reaching zero sends that population home.
            next_homing[self.th] += search[0]
            active_search = search[1:]
            to_avoid = gamma_r * active_search
            to_grab = gamma_f * active_search
            stay_search = active_search - to_avoid - to_grab
            next_search[:-1] += stay_search
            next_avoid[:-1, self.ta] += to_avoid
            next_grab[self.tg] += float(to_grab.sum())

            # Search credit continues to count down during avoidance.
            done_avoid = avoid_search[:, 0]
            next_homing[self.th] += float(done_avoid[0])
            next_search[1:] += done_avoid[1:]
            moving_avoid = avoid_search[:, 1:]
            next_avoid[0, :-1] += moving_avoid[0, :]
            next_avoid[:-1, :-1] += moving_avoid[1:, :]

            # A lost food target returns that population to searching.
            lost = gamma_l * grab
            still_grabbing = grab - lost
            next_search[self.ts] += float(lost.sum())
            entered_deposit = float(still_grabbing[0])
            next_deposit[self.td] += entered_deposit
            next_grab[:-1] += still_grabbing[1:]

            # Deposit and homing use timed queues.
            completed_deposit = float(deposit[0])
            next_resting += completed_deposit
            reached_home_empty = float(homing[0])
            next_resting += reached_home_empty
            next_deposit[:-1] += deposit[1:]
            next_homing[:-1] += homing[1:]

            search, avoid_search, grab, deposit, homing, resting = (
                next_search,
                next_avoid,
                next_grab,
                next_deposit,
                next_homing,
                next_resting,
            )

            # Convert the per-second food growth rate to one simulation step.
            food = max(0.0, food + self.world.p_new_per_step - entered_deposit)

            # Deposits add energy; active and resting robots consume it over dt.
            total_resting = resting
            total_active = self.world.n_robots - total_resting
            energy += (
                self.food_reward * completed_deposit
                - self.world.dt * (self.resting_cost * total_resting + self.active_cost * total_active)
            )

            if step % stride == 0 or step == steps - 1:
                rows.append(
                    {
                        "time_s": (step + 1) * self.world.dt,
                        "rest_time_s": self.rest_time_s,
                        "energy": energy,
                        "food_items": food,
                        "searching": float(search.sum()),
                        "grabbing": float(grab.sum()),
                        "deposit": float(deposit.sum()),
                        "homing": float(homing.sum()),
                        "resting": float(resting),
                        "avoidance": float(avoid_search.sum()),
                        "gamma_f": gamma_f,
                        "gamma_r": gamma_r,
                        "gamma_l": gamma_l,
                        "entered_deposit": entered_deposit,
                        "completed_deposit": completed_deposit,
                    }
                )

        return MacroResult(pd.DataFrame(rows), self.rest_time_s)

    def _totals(self, search, grab, deposit, homing, resting, avoid_search) -> dict[str, float]:
        return {
            "searching": float(search.sum()),
            "grabbing": float(grab.sum()),
            "deposit": float(deposit.sum()),
            "homing": float(homing.sum()),
            "resting": float(resting),
            "avoidance": float(avoid_search.sum()),
        }

    def _rates(self, food: float, totals: dict[str, float]) -> tuple[float, float, float]:
        active = self.world.n_robots - totals["resting"]
        gamma_f = self.world.find_probability(food)
        gamma_r = self.world.collision_probability(active)
        competing = self.world.n_robots - (totals["resting"] + totals["homing"] + totals["deposit"])
        gamma_l = self.world.loss_probability(food, competing, self.tg)

        if gamma_f + gamma_r > 0.98:
            scale = 0.98 / (gamma_f + gamma_r)
            gamma_f *= scale
            gamma_r *= scale
        if gamma_l + gamma_r > 0.98:
            scale = 0.98 / (gamma_l + gamma_r)
            gamma_l *= scale
            gamma_r *= scale
        return gamma_f, gamma_r, gamma_l
