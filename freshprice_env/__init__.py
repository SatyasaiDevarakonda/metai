"""QStorePrice — multi-agent perishable-goods ecosystem (OpenEnv-native).

Top-level façade. The original ``FreshPriceEnv`` is the single-store
gym env (still used by training/eval). The new headline envs live
alongside:

  - ``MarketCommonsEnv``: multi-agent market with hero + competitors +
    farmer pool + bus + regulator + (optional) oversight auditor
  - ``LongHorizonFreshPriceEnv``: 30-day, sparse-reward, notebook-driven
  - ``MultiAgentFreshPriceEnv``: hero + reactive consumer cohort
  - ``MultiStoreFreshPriceEnv``: cooperative N-store transfers
  - ``NegotiationEnv``: bilateral self-play arena

Tier-1 plumbing:

  - ``persistence.ReputationStore``: SQLite reputation graph
  - ``notebook.AgentNotebook``: durable scratchpad
  - ``protocol.MarketBus``: typed message log
  - ``brief_pipeline.schema_registry``: versioned DIRECTIVE schemas

Tier-2 plumbing:

  - ``scenario_composer.ScenarioComposer``: adaptive curriculum
  - ``agents.OversightAuditor``: trajectory auditor (Fleet AI)
  - ``agents.RegulatorAgent``: schema drift broadcaster (Patronus)
  - ``agents.InfluencerAgent``: trend signal emitter w/ disinfo
"""

from freshprice_env.enums import (
    BatchStatus,
    BatchType,
    BriefConfidence,
    BriefEngineType,
    CompensationPolicy,
    CurriculumScenario,
    ExpiryUrgency,
    FarmerOfferStatus,
    SellerAction,
    SignalSource,
    TrendAction,
    ViabilityOutcome,
)
from freshprice_env.freshprice_env import FreshPriceEnv


def _load_openenv_adapter():
    """Lazy import — openenv-core is an optional dependency."""
    from freshprice_env.openenv_adapter import (
        BriefAction,
        BriefObservation,
        FreshPriceOpenEnv,
        FreshPriceState,
    )
    return BriefAction, BriefObservation, FreshPriceState, FreshPriceOpenEnv


def _load_long_horizon():
    from freshprice_env.long_horizon_env import LongHorizonFreshPriceEnv
    return LongHorizonFreshPriceEnv


def _load_market_commons():
    from freshprice_env.market_commons_env import MarketCommonsEnv
    return MarketCommonsEnv


def _load_multi_agent():
    from freshprice_env.multi_agent_env import MultiAgentFreshPriceEnv
    return MultiAgentFreshPriceEnv


def _load_multi_store():
    from freshprice_env.multi_store_env import MultiStoreFreshPriceEnv
    return MultiStoreFreshPriceEnv


def _load_negotiation():
    from freshprice_env.negotiation_env import NegotiationEnv
    return NegotiationEnv


__all__ = [
    "FreshPriceEnv",
    "_load_openenv_adapter",
    "_load_long_horizon",
    "_load_market_commons",
    "_load_multi_agent",
    "_load_multi_store",
    "_load_negotiation",
    "BatchStatus",
    "BatchType",
    "BriefConfidence",
    "BriefEngineType",
    "CompensationPolicy",
    "CurriculumScenario",
    "ExpiryUrgency",
    "FarmerOfferStatus",
    "SellerAction",
    "SignalSource",
    "TrendAction",
    "ViabilityOutcome",
]
