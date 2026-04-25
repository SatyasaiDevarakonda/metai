"""SchemaRegistry — versioned DIRECTIVE schemas for Patronus-style schema drift.

The DIRECTIVE JSON the LLM emits has historically been one fixed shape.
The Patronus AI sub-prize wants "consumer workflows where the underlying
data schemas, API contracts, and t&cs/policies/rules change."

Mid-episode the RegulatorAgent broadcasts a policy change that swaps in
a new schema version. The validator consults the *current* schema (set
by the env). Briefs that don't adapt to the new schema fail validation,
forcing the LLM to read regulator broadcasts and update its plan.

Schemas:

  PRICING.v1      — original (price_multiplier, flash_sale, bundle_with)
  PRICING.v2      — adds mandatory `cold_chain_log` for dairy/leafy_greens
  PRICING.v3      — renames `price_multiplier` → `clearance_action`,
                    raises floor 0.25 → 0.40 (regulatory price-cap)

  FARMER.v1       — original (offer_id, decision, counter_price)
  FARMER.v2       — adds mandatory `farmer_id` field for FSSAI traceability

  TREND.v1        — original (category, decision, order_quantity_kg)
  TREND.v2        — adds mandatory `source_verification` field (cross-check)

The registry is *active*-aware: the env tells it which version to use
right now. `validate_directive(directive, engine_type)` returns
``(ok, list_of_errors)``.

This is intentionally lightweight — full JSON Schema would be overkill
and would couple us to a heavy library. The registry uses plain dict
inspection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from freshprice_env.enums import BriefEngineType


# ---------------------------------------------------------------------------
# Schema descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaSpec:
    """A single (engine, version) schema."""

    engine: BriefEngineType
    version: str
    required_top_keys: list[str]
    required_action_keys: list[str]
    forbidden_action_keys: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_PRICING_V1 = SchemaSpec(
    engine=BriefEngineType.PRICING,
    version="v1",
    required_top_keys=["engine", "actions"],
    required_action_keys=["batch_id", "price_multiplier"],
    description="Original pricing schema",
)

_PRICING_V2 = SchemaSpec(
    engine=BriefEngineType.PRICING,
    version="v2",
    required_top_keys=["engine", "actions"],
    required_action_keys=["batch_id", "price_multiplier", "cold_chain_log"],
    description="FSSAI-compliant: cold_chain_log mandatory for perishables",
)

_PRICING_V3 = SchemaSpec(
    engine=BriefEngineType.PRICING,
    version="v3",
    required_top_keys=["engine", "actions"],
    required_action_keys=["batch_id", "clearance_action"],
    forbidden_action_keys=["price_multiplier"],
    constraints=["clearance_action_min=0.40"],
    description=(
        "Regulator price-cap: floor multiplier raised to 0.40, field renamed "
        "to `clearance_action`. Briefs still using `price_multiplier` fail."
    ),
)

_FARMER_V1 = SchemaSpec(
    engine=BriefEngineType.FARMER,
    version="v1",
    required_top_keys=["engine", "actions"],
    required_action_keys=["offer_id", "decision"],
    description="Original farmer schema",
)

_FARMER_V2 = SchemaSpec(
    engine=BriefEngineType.FARMER,
    version="v2",
    required_top_keys=["engine", "actions"],
    required_action_keys=["offer_id", "decision", "farmer_id"],
    description="FSSAI traceability: farmer_id mandatory in directive",
)

_TREND_V1 = SchemaSpec(
    engine=BriefEngineType.TREND,
    version="v1",
    required_top_keys=["engine", "actions"],
    required_action_keys=["category", "decision"],
    description="Original trend schema",
)

_TREND_V2 = SchemaSpec(
    engine=BriefEngineType.TREND,
    version="v2",
    required_top_keys=["engine", "actions"],
    required_action_keys=["category", "decision", "source_verification"],
    description="Disinformation guard: source_verification mandatory",
)

ALL_SCHEMAS: dict[tuple[str, str], SchemaSpec] = {
    (s.engine.value, s.version): s
    for s in (
        _PRICING_V1, _PRICING_V2, _PRICING_V3,
        _FARMER_V1, _FARMER_V2,
        _TREND_V1, _TREND_V2,
    )
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """Holds the *current* schema version per engine.

    Mutable. Thread-safe. The RegulatorAgent updates this mid-episode.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[BriefEngineType, str] = {
            BriefEngineType.PRICING: "v1",
            BriefEngineType.FARMER: "v1",
            BriefEngineType.TREND: "v1",
        }
        self._history: list[tuple[int, BriefEngineType, str]] = []  # (tick, engine, version)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_version(
        self, engine: BriefEngineType, version: str, tick: int = 0,
    ) -> SchemaSpec:
        key = (engine.value, version)
        if key not in ALL_SCHEMAS:
            raise ValueError(f"unknown schema {engine.value} {version}")
        with self._lock:
            self._active[engine] = version
            self._history.append((tick, engine, version))
        return ALL_SCHEMAS[key]

    def reset_to_v1(self) -> None:
        with self._lock:
            self._active = {
                BriefEngineType.PRICING: "v1",
                BriefEngineType.FARMER: "v1",
                BriefEngineType.TREND: "v1",
            }
            self._history.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def active_version(self, engine: BriefEngineType) -> str:
        with self._lock:
            return self._active[engine]

    def active_spec(self, engine: BriefEngineType) -> SchemaSpec:
        v = self.active_version(engine)
        return ALL_SCHEMAS[(engine.value, v)]

    def history(self) -> list[dict]:
        with self._lock:
            return [
                {"tick": t, "engine": e.value, "version": v}
                for (t, e, v) in self._history
            ]

    def describe_active(self) -> str:
        with self._lock:
            parts = [
                f"{eng.value} {ver}: {ALL_SCHEMAS[(eng.value, ver)].description}"
                for eng, ver in self._active.items()
            ]
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, directive: dict, engine: BriefEngineType) -> tuple[bool, list[str]]:
        """Return (ok, errors) for a directive against the active schema."""
        spec = self.active_spec(engine)
        errors: list[str] = []

        if not isinstance(directive, dict):
            return False, ["directive is not a dict"]

        # Top-level keys
        for k in spec.required_top_keys:
            if k not in directive:
                errors.append(f"missing top-level key: {k}")

        actions = directive.get("actions", [])
        if not isinstance(actions, list):
            errors.append("actions must be a list")
            actions = []

        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                errors.append(f"actions[{i}] is not a dict")
                continue
            for k in spec.required_action_keys:
                if k not in action:
                    errors.append(
                        f"actions[{i}] missing required key '{k}' "
                        f"(active schema: {spec.engine.value} {spec.version})"
                    )
            for k in spec.forbidden_action_keys:
                if k in action:
                    errors.append(
                        f"actions[{i}] uses forbidden key '{k}' under "
                        f"{spec.engine.value} {spec.version}"
                    )

        # Constraint enforcement (lightweight numeric checks)
        for c in spec.constraints:
            if "=" in c:
                key, _, val = c.partition("=")
                key = key.strip()
                if key == "clearance_action_min":
                    try:
                        floor = float(val)
                    except ValueError:
                        continue
                    for i, action in enumerate(actions):
                        if not isinstance(action, dict):
                            continue
                        x = action.get("clearance_action")
                        if isinstance(x, (int, float)) and x < floor:
                            errors.append(
                                f"actions[{i}] clearance_action {x:.2f} below "
                                f"regulatory floor {floor:.2f}"
                            )
        return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_registry: SchemaRegistry | None = None
_lock = threading.Lock()


def default_registry() -> SchemaRegistry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = SchemaRegistry()
        return _registry


def reset_default_registry() -> None:
    global _registry
    with _lock:
        _registry = SchemaRegistry()
