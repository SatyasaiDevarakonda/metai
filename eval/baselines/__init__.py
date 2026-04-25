"""Baseline agents for before/after training comparison.

Provides the comparison table that demonstrates training improvement.
"""

from eval.baselines.random_agent import RandomAgent
from eval.baselines.rule_based_agent import RuleBasedAgent

__all__ = ["RandomAgent", "RuleBasedAgent"]
