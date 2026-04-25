"""Cross-episode persistence layer.

The persistence package holds state that survives env.reset() — farmer
reputation, oversight audit history, agent notebook scratchpads, and the
adaptive ScenarioComposer's failure log.

Backed by SQLite by default (no extra deps). All stores accept a
``db_path`` constructor arg; pass ``":memory:"`` for tests.
"""

from freshprice_env.persistence.reputation_store import (
    FarmerReputation,
    InteractionRecord,
    ReputationStore,
)

__all__ = ["FarmerReputation", "InteractionRecord", "ReputationStore"]
