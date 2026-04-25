"""ReputationStore — SQLite-backed memory of every store↔farmer interaction.

Farmers in QStorePrice now *remember*. A store that lowballs a farmer in
week 1 will see prices rise across the entire farmer pool by week 4.
Acceptances build trust; declines and counter-offers below reserve damage
it. Trust modulates the next offer's reserve price.

Schema:

    farmers
      farmer_id TEXT PRIMARY KEY
      farmer_name TEXT
      base_reserve_per_kg REAL
      trust_score REAL          -- 0.0 (burnt) .. 1.0 (loyal)
      total_interactions INTEGER
      total_accepted INTEGER
      total_declined INTEGER
      total_lowball_counters INTEGER
      last_seen_episode_id TEXT
      created_at_iso TEXT

    interactions
      id INTEGER PRIMARY KEY AUTOINCREMENT
      episode_id TEXT
      tick INTEGER
      farmer_id TEXT
      store_id TEXT
      offer_price_per_kg REAL
      decision TEXT             -- ACCEPT | COUNTER | DECLINE
      counter_price_per_kg REAL
      reserve_at_time REAL
      surplus_for_farmer REAL   -- price - reserve, can be negative
      created_at_iso TEXT
      FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id)

The store is theory-of-mind glue: agents that learn to model farmer
incentives across episodes will outperform agents that don't. The trust
score is *observable* to the LLM in the prompt — the agent has to choose
between short-term gain (lowball) and long-term standing (fair price).
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_REPUTATION_DECAY_PER_LOWBALL: float = 0.08
_REPUTATION_GAIN_PER_ACCEPT: float = 0.05
_REPUTATION_DECAY_PER_DECLINE: float = 0.02
_RESERVE_PRICE_TRUST_FLOOR_MULTIPLIER: float = 0.95   # loyal farmer (trust=1) floors at 95% of base
_RESERVE_PRICE_TRUST_CEIL_MULTIPLIER: float = 1.35    # burnt farmer (trust=0) charges 35% premium


@dataclass(frozen=True)
class FarmerReputation:
    """Snapshot of a farmer's standing with the wider store ecosystem."""

    farmer_id: str
    farmer_name: str
    base_reserve_per_kg: float
    trust_score: float
    total_interactions: int
    total_accepted: int
    total_declined: int
    total_lowball_counters: int
    last_seen_episode_id: str | None

    @property
    def acceptance_rate(self) -> float:
        if self.total_interactions == 0:
            return 0.0
        return self.total_accepted / self.total_interactions

    def adjusted_reserve(self) -> float:
        """Return the reserve price modulated by current trust.

        trust=1.0 → 0.95 × base (loyal farmer is willing to give a small
        discount because they expect repeat business).
        trust=0.0 → 1.35 × base (burnt farmer adds a premium).
        Linear interpolation between.
        """
        t = max(0.0, min(1.0, self.trust_score))
        mult = (
            _RESERVE_PRICE_TRUST_FLOOR_MULTIPLIER * t
            + _RESERVE_PRICE_TRUST_CEIL_MULTIPLIER * (1.0 - t)
        )
        return round(self.base_reserve_per_kg * mult, 2)

    def describe(self) -> str:
        return (
            f"{self.farmer_name} (trust={self.trust_score:.2f}, "
            f"{self.total_accepted}/{self.total_interactions} accepted)"
        )


@dataclass(frozen=True)
class InteractionRecord:
    """A single store↔farmer offer event for audit/replay."""

    episode_id: str
    tick: int
    farmer_id: str
    store_id: str
    offer_price_per_kg: float
    decision: str                  # ACCEPT | COUNTER | DECLINE
    counter_price_per_kg: float | None
    reserve_at_time: float
    surplus_for_farmer: float


class ReputationStore:
    """SQLite-backed reputation graph for farmer↔store interactions.

    Thread-safe via a single lock. Survives env.reset(). Pass ``":memory:"``
    for ephemeral test stores.
    """

    def __init__(self, db_path: str | Path = "qstoreprice_memory.db") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        # check_same_thread=False so the FastAPI server (which spawns
        # request threads) can share one store across requests.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS farmers (
                    farmer_id TEXT PRIMARY KEY,
                    farmer_name TEXT NOT NULL,
                    base_reserve_per_kg REAL NOT NULL,
                    trust_score REAL NOT NULL DEFAULT 0.6,
                    total_interactions INTEGER NOT NULL DEFAULT 0,
                    total_accepted INTEGER NOT NULL DEFAULT 0,
                    total_declined INTEGER NOT NULL DEFAULT 0,
                    total_lowball_counters INTEGER NOT NULL DEFAULT 0,
                    last_seen_episode_id TEXT,
                    created_at_iso TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    farmer_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    offer_price_per_kg REAL NOT NULL,
                    decision TEXT NOT NULL,
                    counter_price_per_kg REAL,
                    reserve_at_time REAL NOT NULL,
                    surplus_for_farmer REAL NOT NULL,
                    created_at_iso TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_inter_farmer
                  ON interactions(farmer_id);
                CREATE INDEX IF NOT EXISTS idx_inter_episode
                  ON interactions(episode_id);
                """
            )

    # ------------------------------------------------------------------
    # Farmer registration
    # ------------------------------------------------------------------

    def upsert_farmer(
        self,
        farmer_id: str,
        farmer_name: str,
        base_reserve_per_kg: float,
        initial_trust: float = 0.6,
    ) -> FarmerReputation:
        """Register a farmer or return their existing reputation."""
        now = _utc_now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO farmers
                  (farmer_id, farmer_name, base_reserve_per_kg, trust_score,
                   total_interactions, total_accepted, total_declined,
                   total_lowball_counters, last_seen_episode_id, created_at_iso)
                VALUES (?, ?, ?, ?, 0, 0, 0, 0, NULL, ?)
                """,
                (farmer_id, farmer_name, float(base_reserve_per_kg),
                 float(initial_trust), now),
            )
        rep = self.get(farmer_id)
        assert rep is not None
        return rep

    def get(self, farmer_id: str) -> FarmerReputation | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM farmers WHERE farmer_id = ?", (farmer_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_reputation(row)

    def all_farmers(self) -> list[FarmerReputation]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM farmers ORDER BY trust_score DESC"
            ).fetchall()
        return [_row_to_reputation(r) for r in rows]

    # ------------------------------------------------------------------
    # Interaction recording
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        episode_id: str,
        tick: int,
        farmer_id: str,
        store_id: str,
        offer_price_per_kg: float,
        decision: str,
        counter_price_per_kg: float | None,
        reserve_at_time: float,
    ) -> InteractionRecord:
        """Log an interaction and update the farmer's trust score in one txn.

        Returns the persisted record. Decision is one of ACCEPT/COUNTER/DECLINE.
        """
        decision = decision.upper()
        if decision not in {"ACCEPT", "COUNTER", "DECLINE"}:
            raise ValueError(f"Unknown decision: {decision}")

        # Compute surplus (positive = farmer made profit vs. reserve)
        effective_price = (
            counter_price_per_kg
            if (decision == "COUNTER" and counter_price_per_kg is not None)
            else offer_price_per_kg
        )
        surplus = effective_price - reserve_at_time

        # Trust delta
        delta = 0.0
        is_lowball_counter = (
            decision == "COUNTER"
            and counter_price_per_kg is not None
            and counter_price_per_kg < reserve_at_time
        )
        if decision == "ACCEPT":
            delta = _REPUTATION_GAIN_PER_ACCEPT
        elif decision == "DECLINE":
            delta = -_REPUTATION_DECAY_PER_DECLINE
        if is_lowball_counter:
            delta -= _REPUTATION_DECAY_PER_LOWBALL

        now = _utc_now_iso()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO interactions
                  (episode_id, tick, farmer_id, store_id, offer_price_per_kg,
                   decision, counter_price_per_kg, reserve_at_time,
                   surplus_for_farmer, created_at_iso)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id, int(tick), farmer_id, store_id,
                    float(offer_price_per_kg), decision,
                    float(counter_price_per_kg) if counter_price_per_kg is not None else None,
                    float(reserve_at_time), float(surplus), now,
                ),
            )
            self._conn.execute(
                """
                UPDATE farmers
                SET trust_score = MAX(0.0, MIN(1.0, trust_score + ?)),
                    total_interactions = total_interactions + 1,
                    total_accepted = total_accepted + ?,
                    total_declined = total_declined + ?,
                    total_lowball_counters = total_lowball_counters + ?,
                    last_seen_episode_id = ?
                WHERE farmer_id = ?
                """,
                (
                    delta,
                    1 if decision == "ACCEPT" else 0,
                    1 if decision == "DECLINE" else 0,
                    1 if is_lowball_counter else 0,
                    episode_id,
                    farmer_id,
                ),
            )

        return InteractionRecord(
            episode_id=episode_id,
            tick=tick,
            farmer_id=farmer_id,
            store_id=store_id,
            offer_price_per_kg=offer_price_per_kg,
            decision=decision,
            counter_price_per_kg=counter_price_per_kg,
            reserve_at_time=reserve_at_time,
            surplus_for_farmer=surplus,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def history_for_farmer(self, farmer_id: str, limit: int = 20) -> list[InteractionRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM interactions
                WHERE farmer_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (farmer_id, limit),
            ).fetchall()
        return [_row_to_interaction(r) for r in rows]

    def history_for_episode(self, episode_id: str) -> Iterator[InteractionRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM interactions WHERE episode_id = ? ORDER BY id ASC",
                (episode_id,),
            ).fetchall()
        for r in rows:
            yield _row_to_interaction(r)

    def pool_trust_summary(self) -> dict[str, float]:
        """Mean trust across all known farmers — drives global pricing pressure.

        Used by FarmerAgent and the dashboard. A pool-wide trust collapse
        (e.g., from repeated lowballing) raises every new farmer's reserve.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                  COUNT(*) AS n,
                  AVG(trust_score) AS mean_trust,
                  MIN(trust_score) AS min_trust,
                  MAX(trust_score) AS max_trust
                FROM farmers
                """
            ).fetchone()
        return {
            "n_farmers": int(row["n"] or 0),
            "mean_trust": float(row["mean_trust"] or 0.6),
            "min_trust": float(row["min_trust"] or 0.6),
            "max_trust": float(row["max_trust"] or 0.6),
        }

    # ------------------------------------------------------------------
    # Test / reset helpers
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """Wipe all reputation data. For tests only."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM interactions")
            self._conn.execute("DELETE FROM farmers")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton (default db lives in working dir)
# ---------------------------------------------------------------------------

_default_store: ReputationStore | None = None
_default_lock = threading.Lock()


def default_store() -> ReputationStore:
    """Return a process-wide singleton. Tests should construct their own."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = ReputationStore()
        return _default_store


def reset_default_store() -> None:
    """Replace the default singleton with an in-memory one. Used by tests."""
    global _default_store
    with _default_lock:
        if _default_store is not None:
            _default_store.close()
        _default_store = ReputationStore(db_path=":memory:")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_reputation(row: sqlite3.Row) -> FarmerReputation:
    return FarmerReputation(
        farmer_id=row["farmer_id"],
        farmer_name=row["farmer_name"],
        base_reserve_per_kg=float(row["base_reserve_per_kg"]),
        trust_score=float(row["trust_score"]),
        total_interactions=int(row["total_interactions"]),
        total_accepted=int(row["total_accepted"]),
        total_declined=int(row["total_declined"]),
        total_lowball_counters=int(row["total_lowball_counters"]),
        last_seen_episode_id=row["last_seen_episode_id"],
    )


def _row_to_interaction(row: sqlite3.Row) -> InteractionRecord:
    return InteractionRecord(
        episode_id=row["episode_id"],
        tick=int(row["tick"]),
        farmer_id=row["farmer_id"],
        store_id=row["store_id"],
        offer_price_per_kg=float(row["offer_price_per_kg"]),
        decision=row["decision"],
        counter_price_per_kg=(
            float(row["counter_price_per_kg"])
            if row["counter_price_per_kg"] is not None else None
        ),
        reserve_at_time=float(row["reserve_at_time"]),
        surplus_for_farmer=float(row["surplus_for_farmer"]),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
