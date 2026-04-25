"""Agents package — actors (LLM-driven and procedural) populating the ecosystem."""

from freshprice_env.agents.competitor_store_agent import (
    CompetitorAction,
    CompetitorPersona,
    CompetitorStoreAgent,
)
from freshprice_env.agents.consumer_agent import ConsumerAgent
from freshprice_env.agents.farmer_agent import (
    FarmerAgent,
    FarmerPersona,
    FarmerProfile,
    build_default_farmer_pool,
    DEFAULT_FARMER_ROSTER,
)
from freshprice_env.agents.influencer_agent import (
    InfluencerAgent,
    TaggedTrendSignal,
    TrendVeracity,
)
from freshprice_env.agents.oversight_auditor import (
    AuditableEvent,
    AuditReport,
    AuditTrajectory,
    OversightAuditor,
    trajectory_from_market_commons,
)
from freshprice_env.agents.regulator_agent import (
    PolicyChange,
    RegulatorAgent,
    RegulatoryEvent,
)

__all__ = [
    # Consumer
    "ConsumerAgent",
    # Farmer
    "FarmerAgent",
    "FarmerPersona",
    "FarmerProfile",
    "build_default_farmer_pool",
    "DEFAULT_FARMER_ROSTER",
    # Competitor
    "CompetitorAction",
    "CompetitorPersona",
    "CompetitorStoreAgent",
    # Oversight
    "AuditableEvent",
    "AuditReport",
    "AuditTrajectory",
    "OversightAuditor",
    "trajectory_from_market_commons",
    # Regulator
    "PolicyChange",
    "RegulatorAgent",
    "RegulatoryEvent",
    # Influencer
    "InfluencerAgent",
    "TaggedTrendSignal",
    "TrendVeracity",
]
