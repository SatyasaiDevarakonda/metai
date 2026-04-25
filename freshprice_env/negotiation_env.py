"""NegotiationEnv — self-play farmer-store negotiation arena.

Theme #4 alignment (Self-Improvement via self-play):
  The LLM plays BOTH roles simultaneously:
    - Store Manager: wants low price, high shelf life, high viability
    - Farmer Agent:  wants high price, quick commitment, guaranteed purchase

  The negotiation is turn-based over up to MAX_ROUNDS rounds.
  The LLM alternates between roles, learning to model the other side's
  incentives — a recursive self-improvement loop.

Theme #1 alignment (Multi-Agent — negotiation and coalition formation):
  This is a bilateral negotiation with partially observable information.
  Each side has private information (reserve price, supply urgency) that
  the other side must infer through the negotiation dialogue.

Episode structure:
  Round 0: Farmer opens with offer (price, quantity, shelf life)
  Round 1: Store responds (ACCEPT / COUNTER / DECLINE)
  Round 2: Farmer responds to counter (ACCEPT_COUNTER / COUNTER_AGAIN / WALK)
  Round 3: Final store decision
  Outcome: Deal struck / No deal

Reward:
  store_reward = deal_surplus_for_store / max_possible_surplus  (0-1)
  farmer_reward = deal_surplus_for_farmer / max_possible_surplus (0-1)
  combined_reward = 0.5 * store_reward + 0.5 * farmer_reward
  cooperation_bonus = +0.15 if deal struck AND both sides >0.4 surplus

Usage:
    env = NegotiationEnv(seed=42)
    obs, info = env.reset()                         # Farmer's opening offer
    obs, reward, done, _, info = env.step(llm_response)  # Store's first response
    obs, reward, done, _, info = env.step(llm_response)  # Farmer's counter ...
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum

import gymnasium as gym

from freshprice_env.constants import FARMER_OPS_COST_PER_KG


class NegotiationRole(str, Enum):
    FARMER = "FARMER"
    STORE = "STORE"


class NegotiationOutcome(str, Enum):
    DEAL = "DEAL"
    NO_DEAL = "NO_DEAL"
    IN_PROGRESS = "IN_PROGRESS"


@dataclass
class NegotiationState:
    """Complete state of a bilateral negotiation."""
    round: int
    active_role: NegotiationRole
    outcome: NegotiationOutcome

    # Offer parameters (mutable through negotiation)
    current_price_per_kg: float
    quantity_kg: float
    shelf_life_hrs: int
    product_category: str
    product_name: str

    # Private information (each side's reserve)
    farmer_reserve_price: float    # won't sell below this
    store_max_price: float         # won't buy above this

    # Negotiation history
    history: list[dict] = field(default_factory=list)

    @property
    def deal_viable(self) -> bool:
        return self.current_price_per_kg <= self.store_max_price

    @property
    def store_surplus(self) -> float:
        if self.outcome != NegotiationOutcome.DEAL:
            return 0.0
        return max(0.0, self.store_max_price - self.current_price_per_kg)

    @property
    def farmer_surplus(self) -> float:
        if self.outcome != NegotiationOutcome.DEAL:
            return 0.0
        return max(0.0, self.current_price_per_kg - self.farmer_reserve_price)


MAX_ROUNDS: int = 4


class NegotiationEnv(gym.Env):
    """Bilateral negotiation environment for self-play training.

    The observation tells the agent:
      - Its current role (FARMER or STORE)
      - The current offer on the table
      - Its own private reserve price
      - The negotiation history

    The action is a structured negotiation response:
      ACCEPT | DECLINE | COUNTER {price: X.XX, reason: "..."}
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, seed: int = 42, render_mode: str = "none") -> None:
        super().__init__()
        self._seed = seed
        self.render_mode = render_mode
        self._rng = random.Random(seed)
        self._state: NegotiationState | None = None

        # Text action/observation spaces
        _charset = frozenset(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 \t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        )
        self.observation_space = gym.spaces.Text(min_length=0, max_length=4096, charset=_charset)
        self.action_space = gym.spaces.Text(min_length=0, max_length=1024, charset=_charset)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        # Generate a new negotiation scenario
        categories = ["fruits", "vegetables", "dairy", "mushrooms", "herbs"]
        cat = self._rng.choice(categories)
        products = {
            "fruits": "mangoes", "vegetables": "tomatoes", "dairy": "paneer",
            "mushrooms": "button mushrooms", "herbs": "coriander",
        }

        # Farmer's private reserve (won't go below cost + margin)
        cost = round(self._rng.uniform(15.0, 40.0), 2)
        farmer_reserve = round(cost * 1.10, 2)
        opening_ask = round(farmer_reserve * self._rng.uniform(1.15, 1.35), 2)

        # Store's private max price (above this, viability fails)
        store_max = round(opening_ask * self._rng.uniform(0.85, 1.10), 2)

        qty = round(self._rng.uniform(20.0, 60.0), 1)
        shelf = self._rng.randint(24, 72)

        self._state = NegotiationState(
            round=0,
            active_role=NegotiationRole.FARMER,  # Farmer opens
            outcome=NegotiationOutcome.IN_PROGRESS,
            current_price_per_kg=opening_ask,
            quantity_kg=qty,
            shelf_life_hrs=shelf,
            product_category=cat,
            product_name=products[cat],
            farmer_reserve_price=farmer_reserve,
            store_max_price=store_max,
        )

        obs = self._build_observation()
        info = {
            "round": 0,
            "active_role": NegotiationRole.FARMER.value,
            "product": f"{qty}kg {products[cat]} @ Rs{opening_ask:.2f}/kg",
        }
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: str):
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.outcome != NegotiationOutcome.IN_PROGRESS:
            raise RuntimeError("Episode already terminated")

        # Parse the LLM's response
        parsed = self._parse_action(action)
        self._state.history.append({
            "round": self._state.round,
            "role": self._state.active_role.value,
            "action": parsed,
        })

        reward = 0.0
        terminated = False

        decision = parsed.get("decision", "DECLINE").upper()

        if decision == "ACCEPT":
            self._state.outcome = NegotiationOutcome.DEAL
            terminated = True
            reward = self._compute_reward()

        elif decision == "DECLINE" or decision == "WALK":
            self._state.outcome = NegotiationOutcome.NO_DEAL
            terminated = True
            reward = -0.10  # small penalty for failed negotiation

        elif decision == "COUNTER":
            new_price = parsed.get("price", self._state.current_price_per_kg)
            self._state.current_price_per_kg = max(
                self._state.farmer_reserve_price,
                min(float(new_price), self._state.store_max_price * 1.20),
            )
            self._state.round += 1

            if self._state.round >= MAX_ROUNDS:
                # Auto-resolve at current price
                if self._state.deal_viable:
                    self._state.outcome = NegotiationOutcome.DEAL
                    reward = self._compute_reward() * 0.8  # penalty for dragging out
                else:
                    self._state.outcome = NegotiationOutcome.NO_DEAL
                    reward = -0.10
                terminated = True
            else:
                # Switch roles
                self._state.active_role = (
                    NegotiationRole.STORE
                    if self._state.active_role == NegotiationRole.FARMER
                    else NegotiationRole.FARMER
                )

        obs = self._build_observation() if not terminated else ""
        info = {
            "round": self._state.round,
            "active_role": self._state.active_role.value,
            "outcome": self._state.outcome.value,
            "current_price": self._state.current_price_per_kg,
            "store_surplus": self._state.store_surplus,
            "farmer_surplus": self._state.farmer_surplus,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # state (OpenEnv required)
    # ------------------------------------------------------------------

    def state(self) -> dict:
        if self._state is None:
            return {"status": "not_started"}
        return {
            "round": self._state.round,
            "active_role": self._state.active_role.value,
            "outcome": self._state.outcome.value,
            "current_price_per_kg": self._state.current_price_per_kg,
            "quantity_kg": self._state.quantity_kg,
            "product": f"{self._state.product_name} ({self._state.product_category})",
            "history_length": len(self._state.history),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> str:
        s = self._state
        role = s.active_role.value

        if role == "FARMER":
            private_info = (
                f"  YOUR RESERVE PRICE: Rs{s.farmer_reserve_price:.2f}/kg (do not reveal)\n"
                f"  YOUR URGENCY: Perishable — you need to sell within {s.shelf_life_hrs}hrs\n"
            )
        else:
            private_info = (
                f"  YOUR MAX PRICE: Rs{s.store_max_price:.2f}/kg (do not reveal)\n"
                f"  OPS COST: Rs{FARMER_OPS_COST_PER_KG:.2f}/kg handling\n"
            )

        history_lines = ""
        for h in s.history[-4:]:  # last 4 turns
            action_str = json.dumps(h["action"])
            history_lines += f"  [{h['role']}] {action_str}\n"

        return (
            f"## NEGOTIATION — Round {s.round + 1}/{MAX_ROUNDS}\n"
            f"  You are: {role}\n"
            f"  Product: {s.quantity_kg}kg {s.product_name} ({s.product_category})\n"
            f"  Shelf life: {s.shelf_life_hrs} hours\n"
            f"  Current offer on table: Rs{s.current_price_per_kg:.2f}/kg\n\n"
            f"## YOUR PRIVATE INFORMATION\n{private_info}\n"
            f"## NEGOTIATION HISTORY\n"
            f"{history_lines if history_lines else '  (Opening round)\n'}\n"
            f"## YOUR RESPONSE\n"
            f"Reply with JSON: "
            f"{{\"decision\": \"ACCEPT\"|\"COUNTER\"|\"DECLINE\", "
            f"\"price\": <float if COUNTER>, \"reason\": \"<brief justification>\"}}\n"
        )

    def _parse_action(self, action: str) -> dict:
        import re
        # Try JSON parse
        try:
            m = re.search(r"\{[^}]+\}", action, re.DOTALL)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback: keyword scan
        action_lower = action.lower()
        if "accept" in action_lower:
            return {"decision": "ACCEPT"}
        if "decline" in action_lower or "walk" in action_lower:
            return {"decision": "DECLINE"}
        # Find any price mentioned
        price_m = re.search(r"rs\s*(\d+(?:\.\d+)?)", action_lower)
        if price_m:
            return {"decision": "COUNTER", "price": float(price_m.group(1))}
        return {"decision": "DECLINE"}

    def _compute_reward(self) -> float:
        """Reward = normalised combined surplus from both sides."""
        s = self._state
        price_range = s.store_max_price - s.farmer_reserve_price
        if price_range <= 0:
            return 0.5  # forced deal with no surplus

        store_surplus_norm = s.store_surplus / price_range
        farmer_surplus_norm = s.farmer_surplus / price_range
        combined = 0.5 * store_surplus_norm + 0.5 * farmer_surplus_norm

        # Cooperation bonus: both sides got > 40% of their potential surplus
        if store_surplus_norm > 0.4 and farmer_surplus_norm > 0.4:
            combined += 0.15
        return min(combined, 1.0)
