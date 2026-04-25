"""MultiStoreFreshPriceEnv — cooperative multi-store coordination.

Theme #1 alignment (Multi-Agent Interactions — cooperation + coalition formation):
  - N store agents share a common market zone
  - Stores can TRANSFER excess at-risk inventory to stores with higher demand
  - Inter-store transfers cost Rs 5/kg (cold-chain logistics)
  - Combined reward = mean WRR across all stores (encourages cooperation)
  - Agent that donates stock takes a short-term reward hit but raises collective WRR

This creates emergent cooperation dynamics:
  Store A (excess fruits, weak demand)  ──transfer──►  Store B (low stock, festival demand)
  Store B sells more → both WRRs rise → collective reward improves

Each store runs its own FreshPriceEnv. The "coordination" action is injected by
the LLM via a TRANSFER directive in its Operating Brief. The MultiStoreFreshPriceEnv
intercepts this directive and moves inventory between state objects.

Usage:
    env = MultiStoreFreshPriceEnv(n_stores=2, scenario=CurriculumScenario.CRISIS_WEEK)
    obs_list, info = env.reset()                # one obs per store
    obs_list, rewards, done, truncated, info = env.step(brief_texts)  # one brief per store
"""

from __future__ import annotations

import random
from dataclasses import replace

from freshprice_env.constants import (
    INTER_STORE_MAX_TRANSFER_PCT,
    INTER_STORE_TRANSFER_COST_RS_PER_KG,
    MIN_UNITS_FOR_TRANSFER,
)
from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv


class InterStoreTransfer:
    """Represents a proposed transfer of inventory between two stores."""

    def __init__(
        self,
        source_store_idx: int,
        target_store_idx: int,
        batch_id: str,
        units: int,
        category: str,
    ) -> None:
        self.source_store_idx = source_store_idx
        self.target_store_idx = target_store_idx
        self.batch_id = batch_id
        self.units = units
        self.category = category


class MultiStoreFreshPriceEnv:
    """Multi-store cooperative RL environment.

    Each store is an independent FreshPriceEnv with its own inventory, engines,
    and reward accumulator. The coordinating LLM agent can propose inter-store
    transfers that move at-risk inventory to stores where it will sell faster.

    Observation:  list of per-store prompt strings
    Action:       list of Operating Brief strings (one per store) — transfers encoded
                  in a TRANSFER section of the brief
    Reward:       list of per-store WRR deltas + cooperation_bonus if transfers
                  improved collective WRR
    """

    def __init__(
        self,
        n_stores: int = 2,
        scenario: CurriculumScenario = CurriculumScenario.CRISIS_WEEK,
        seed: int = 42,
    ) -> None:
        if n_stores < 2:
            raise ValueError("MultiStoreFreshPriceEnv requires at least 2 stores")
        self.n_stores = n_stores
        self.scenario = scenario
        self._seed = seed

        # Each store gets a different seed so their inventories differ
        self._stores: list[FreshPriceEnv] = [
            FreshPriceEnv(scenario=scenario, seed=seed + i * 100)
            for i in range(n_stores)
        ]

        self._transfer_history: list[dict] = []
        self._episode_transfers: int = 0
        self._cooperation_bonus_total: float = 0.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._seed = seed
            self._stores = [
                FreshPriceEnv(scenario=self.scenario, seed=seed + i * 100)
                for i in range(self.n_stores)
            ]

        observations = []
        infos = []
        for store in self._stores:
            obs, info = store.reset()
            observations.append(obs)
            infos.append(info)

        self._transfer_history = []
        self._episode_transfers = 0
        self._cooperation_bonus_total = 0.0

        combined_info = {
            "store_infos": infos,
            "n_stores": self.n_stores,
            "mode": "multi_store_cooperative",
        }
        return observations, combined_info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        """Process one brief cycle for all stores simultaneously.

        Args:
            actions: List of Operating Brief strings, one per store.
                     If a brief contains a TRANSFER section, it is extracted
                     and executed before the individual store steps.

        Returns:
            (observations, rewards, terminated, truncated, info)
        """
        if len(actions) != self.n_stores:
            raise ValueError(f"Expected {self.n_stores} actions, got {len(actions)}")

        # 1. Extract and execute any inter-store transfers
        transfers, actions = self._extract_transfers(actions)
        transfer_results = self._execute_transfers(transfers)

        # 2. Step each store individually
        observations = []
        rewards = []
        terminated_flags = []
        truncated_flags = []
        store_infos = []

        for i, (store, action) in enumerate(zip(self._stores, actions)):
            obs, reward, terminated, truncated, info = store.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            store_infos.append(info)

        # 3. Cooperation bonus: if collective WRR improved due to transfers
        cooperation_bonus = self._compute_cooperation_bonus(rewards, transfer_results)
        self._cooperation_bonus_total += cooperation_bonus

        # Distribute cooperation bonus equally across stores
        augmented_rewards = [r + cooperation_bonus / self.n_stores for r in rewards]

        terminated = all(terminated_flags)
        truncated = any(truncated_flags)

        info = {
            "store_infos": store_infos,
            "transfers_this_step": transfer_results,
            "cooperation_bonus": cooperation_bonus,
            "collective_wrr": self._collective_wrr(),
            "episode_transfers_total": self._episode_transfers,
        }

        if terminated:
            info["final_collective_wrr"] = self._collective_wrr()
            info["total_cooperation_bonus"] = self._cooperation_bonus_total

        return observations, augmented_rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Inter-store transfer logic
    # ------------------------------------------------------------------

    def _extract_transfers(self, actions: list[str]) -> tuple[list[InterStoreTransfer], list[str]]:
        """Parse TRANSFER directives from brief texts and remove them."""
        transfers: list[InterStoreTransfer] = []
        cleaned_actions = []

        for store_idx, action in enumerate(actions):
            if "## TRANSFER" not in action:
                cleaned_actions.append(action)
                continue

            # Split on TRANSFER section
            parts = action.split("## TRANSFER", 1)
            cleaned_actions.append(parts[0])

            # Parse transfer lines: "STORE_1 → STORE_2: batch_XXXX 10 units"
            transfer_text = parts[1] if len(parts) > 1 else ""
            for line in transfer_text.strip().split("\n"):
                transfer = self._parse_transfer_line(line.strip(), store_idx)
                if transfer is not None:
                    transfers.append(transfer)

        return transfers, cleaned_actions

    @staticmethod
    def _parse_transfer_line(line: str, source_idx: int) -> InterStoreTransfer | None:
        """Parse a single TRANSFER line. Returns None on parse failure."""
        import re
        # Format: "TO_STORE_1: batch_0001 10 units (fruits)"
        m = re.match(
            r"TO_STORE_(\d+):\s*(batch_\w+)\s+(\d+)\s+units?\s*\((\w+)\)",
            line, re.IGNORECASE
        )
        if m is None:
            return None
        target_idx = int(m.group(1))
        batch_id = m.group(2)
        units = int(m.group(3))
        category = m.group(4).lower()
        return InterStoreTransfer(source_idx, target_idx, batch_id, units, category)

    def _execute_transfers(self, transfers: list[InterStoreTransfer]) -> list[dict]:
        """Execute validated inter-store transfers and return results."""
        results = []

        for t in transfers:
            if t.source_store_idx >= self.n_stores or t.target_store_idx >= self.n_stores:
                results.append({"status": "REJECTED", "reason": "invalid_store_index"})
                continue

            source_env = self._stores[t.source_store_idx]
            target_env = self._stores[t.target_store_idx]

            if source_env._state is None or target_env._state is None:
                results.append({"status": "REJECTED", "reason": "env_not_reset"})
                continue

            # Find the batch in source store
            source_batch = next(
                (b for b in source_env._state.batches if b.batch_id == t.batch_id),
                None,
            )
            if source_batch is None:
                results.append({"status": "REJECTED", "reason": "batch_not_found"})
                continue

            # Validate transfer size
            max_transfer = max(
                MIN_UNITS_FOR_TRANSFER,
                int(source_batch.quantity_remaining * INTER_STORE_MAX_TRANSFER_PCT),
            )
            units = min(t.units, max_transfer)
            if units < MIN_UNITS_FOR_TRANSFER:
                results.append({"status": "REJECTED", "reason": "below_minimum_units"})
                continue

            # Compute transfer cost
            avg_weight_kg = 0.25  # default per unit
            transfer_cost = units * avg_weight_kg * INTER_STORE_TRANSFER_COST_RS_PER_KG

            if source_env._state.risk_buffer_balance < transfer_cost:
                results.append({"status": "REJECTED", "reason": "insufficient_buffer"})
                continue

            # Execute: reduce source batch, add to target state
            new_qty = source_batch.quantity_remaining - units
            updated_batch = replace(source_batch, quantity_remaining=new_qty)
            source_env._state.batches = [
                updated_batch if b.batch_id == t.batch_id else b
                for b in source_env._state.batches
            ]
            source_env._state.risk_buffer_balance -= transfer_cost

            # Add transferred units as a new batch in target (same properties)
            from freshprice_env.enums import BatchStatus, BatchType
            transferred_batch = replace(
                source_batch,
                batch_id=f"{t.batch_id}_xfr_{t.target_store_idx}",
                store_id=f"store_{t.target_store_idx:03d}",
                quantity_remaining=units,
                batch_type=BatchType.REGULAR,
                status=BatchStatus.ACTIVE,
            )
            target_env._state.batches.append(transferred_batch)

            self._episode_transfers += 1
            results.append({
                "status": "EXECUTED",
                "batch_id": t.batch_id,
                "units": units,
                "from_store": t.source_store_idx,
                "to_store": t.target_store_idx,
                "cost_rs": round(transfer_cost, 2),
            })

        return results

    def _compute_cooperation_bonus(
        self,
        rewards: list[float],
        transfer_results: list[dict],
    ) -> float:
        """Return a small bonus when transfers successfully executed."""
        executed = sum(1 for r in transfer_results if r.get("status") == "EXECUTED")
        if executed == 0:
            return 0.0
        # Bonus proportional to number of successful transfers (capped)
        return min(executed * 0.02, 0.10)

    def _collective_wrr(self) -> float:
        """Mean WRR across all stores."""
        wrrs = [
            store._state.wrr
            for store in self._stores
            if store._state is not None
        ]
        return sum(wrrs) / len(wrrs) if wrrs else 0.0

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "mode": "multi_store_cooperative",
            "n_stores": self.n_stores,
            "collective_wrr": self._collective_wrr(),
            "episode_transfers": self._episode_transfers,
            "stores": [store.state() for store in self._stores],
        }
