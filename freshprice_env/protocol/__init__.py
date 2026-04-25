"""Inter-agent message protocol for the QStorePrice multi-agent ecosystem.

The MarketBus is a structured, append-only log of messages between
agents. It replaces ad-hoc string parsing with a typed protocol that
the OversightAuditor can audit, the dashboard can stream live, and the
counterfactual replay tool can rewind.
"""

from freshprice_env.protocol.market_bus import (
    MarketBus,
    MarketMessage,
    MessageVerb,
    parse_messages_from_brief,
)

__all__ = [
    "MarketBus",
    "MarketMessage",
    "MessageVerb",
    "parse_messages_from_brief",
]
