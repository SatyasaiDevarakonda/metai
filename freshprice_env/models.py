"""Pydantic v2 models for structured observation, action, and reward types.

Used by inference.py and task graders for type-safe data exchange.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FreshPriceObservation(BaseModel):
    """Structured observation returned alongside the raw prompt string."""

    tick: int
    day_of_week: int
    hour_of_day: int
    prompt: str
    engine_type: str
    scenario: str
    wrr_so_far: float
    active_batches: int
    critical_batches: int
    pending_offers: int
    active_trends: int
    risk_buffer_balance: float


class FreshPriceAction(BaseModel):
    """The action the agent takes — a raw Operating Brief string."""

    raw_brief: str


class FreshPriceReward(BaseModel):
    """Structured reward breakdown per step."""

    total: float
    wrr: float
    r1_pricing: float
    r2_farmer: float
    r3_trend: float
    brief_quality_score: float
    anti_hack_violations: int
    episode_valid: bool


class TaskGraderResult(BaseModel):
    """Result from a formal task grader."""

    task_id: str
    task_name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any]
