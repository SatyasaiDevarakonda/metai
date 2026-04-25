"""MarketBus — typed inter-agent message log.

In MarketCommonsEnv, agents (StoreAgent, CompetitorStoreAgent,
FarmerAgent, RegulatorAgent, ConsumerCohortAgent, InfluencerAgent,
OversightAuditor) talk to each other via a small set of verbs. The bus
keeps a chronological log per episode, broadcasts to subscribers (the
dashboard's WebSocket clients), and gives the auditor an
append-only ground truth of who said what to whom.

Verbs:

  CHAT        — free-form natural-language message (no commitment)
  BID         — propose a price/quantity for a deal
  COUNTER     — counter-propose with new terms
  COMMIT      — bilateral binding agreement: both sides agree to terms
  REVEAL      — voluntarily disclose private info (e.g., "we have surplus mangoes")
  BLUFF       — flag an outgoing message as deception (auditor uses this)
  BROADCAST   — one-to-many announcement (RegulatorAgent uses this for policy)
  AUDIT       — OversightAuditor's report on a trajectory or message

Messages are extracted from agent briefs by ``parse_messages_from_brief``
which scans for a ``## MESSAGES`` section with one verb per line:

    ## MESSAGES
    CHAT @farmer.rajan: appreciate the mango offer, considering it
    BID @farmer.rajan: 38.0/kg for 50kg, 24h decision window
    REVEAL @competitor.store_2: we are over-stocked on dairy this week
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class MessageVerb(str, Enum):
    CHAT = "CHAT"
    BID = "BID"
    COUNTER = "COUNTER"
    COMMIT = "COMMIT"
    REVEAL = "REVEAL"
    BLUFF = "BLUFF"
    BROADCAST = "BROADCAST"
    AUDIT = "AUDIT"


_VERBS = {v.value for v in MessageVerb}


@dataclass(frozen=True)
class MarketMessage:
    """A single inter-agent message."""

    seq: int
    tick: int
    sender_id: str
    receiver_id: str | None         # None for BROADCAST
    verb: MessageVerb
    body: str                       # human-readable
    payload: dict = field(default_factory=dict)   # structured fields if parseable
    timestamp_iso: str = field(default_factory=lambda: _utc_now_iso())

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "tick": self.tick,
            "sender": self.sender_id,
            "receiver": self.receiver_id,
            "verb": self.verb.value,
            "body": self.body,
            "payload": self.payload,
            "timestamp": self.timestamp_iso,
        }


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class MarketBus:
    """Append-only message log with subscriber callbacks.

    Per-episode: env.reset() should call ``bus.clear()``. The dashboard
    can subscribe to a callback that fires on every ``post()`` to push
    via WebSocket.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: list[MarketMessage] = []
        self._next_seq = 0
        self._subscribers: list = []   # list[Callable[[MarketMessage], None]]

    # ------------------------------------------------------------------
    # Posting
    # ------------------------------------------------------------------

    def post(
        self,
        tick: int,
        sender_id: str,
        verb: MessageVerb | str,
        body: str,
        receiver_id: str | None = None,
        payload: dict | None = None,
    ) -> MarketMessage:
        if isinstance(verb, str):
            verb = MessageVerb(verb.upper())
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            msg = MarketMessage(
                seq=seq,
                tick=int(tick),
                sender_id=sender_id,
                receiver_id=receiver_id,
                verb=verb,
                body=str(body)[:1000],
                payload=dict(payload or {}),
            )
            self._messages.append(msg)
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(msg)
            except Exception:  # noqa: BLE001 — never let a bad subscriber stop posting
                pass
        return msg

    # ------------------------------------------------------------------
    # Subscriptions (dashboard WebSocket)
    # ------------------------------------------------------------------

    def subscribe(self, callback) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_messages(self) -> list[MarketMessage]:
        with self._lock:
            return list(self._messages)

    def messages_for(self, agent_id: str) -> list[MarketMessage]:
        with self._lock:
            return [
                m for m in self._messages
                if m.receiver_id == agent_id or m.receiver_id is None
                or m.sender_id == agent_id
            ]

    def messages_since(self, seq: int) -> list[MarketMessage]:
        with self._lock:
            return [m for m in self._messages if m.seq > seq]

    def count_by_verb(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for m in self._messages:
                out[m.verb.value] = out.get(m.verb.value, 0) + 1
        return out

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._next_seq = 0


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------

_MESSAGES_BLOCK_RE = re.compile(
    r"##\s*MESSAGES\s*\n(.*?)(?=\n##\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)
# CHAT @farmer.rajan: appreciate the offer
# BID  @farmer.rajan: 38.0/kg, 50kg, 24h
_LINE_RE = re.compile(
    r"^\s*(CHAT|BID|COUNTER|COMMIT|REVEAL|BLUFF|BROADCAST|AUDIT)"
    r"(?:\s+@?([\w.-]+))?\s*:\s*(.+)$",
    re.IGNORECASE,
)


def parse_messages_from_brief(
    brief_text: str,
    sender_id: str,
    tick: int,
) -> list[tuple[MessageVerb, str | None, str, dict]]:
    """Pull (verb, receiver_id, body, payload) tuples from a ## MESSAGES block.

    Caller does the bus.post() — keeps this module side-effect-free.
    """
    if not brief_text:
        return []
    block_match = _MESSAGES_BLOCK_RE.search(brief_text)
    body = block_match.group(1) if block_match else ""
    if not body:
        return []

    out: list[tuple[MessageVerb, str | None, str, dict]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m is None:
            continue
        verb = MessageVerb(m.group(1).upper())
        receiver = m.group(2) or None
        msg_body = m.group(3).strip()
        payload = _parse_structured_tail(msg_body, verb)
        out.append((verb, receiver, msg_body, payload))
    return out


_PRICE_RE = re.compile(r"(?:rs\s*)?(\d+(?:\.\d+)?)\s*/\s*kg", re.IGNORECASE)
_QTY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kg", re.IGNORECASE)


def _parse_structured_tail(body: str, verb: MessageVerb) -> dict:
    """Best-effort extraction of price/quantity/etc. from the body.

    BID and COUNTER routinely embed "<price>/kg" and "<qty>kg"; pull
    them out so the auditor and dashboard can show them as numbers.
    """
    payload: dict = {}
    if verb in (MessageVerb.BID, MessageVerb.COUNTER, MessageVerb.COMMIT):
        m_price = _PRICE_RE.search(body)
        if m_price:
            try:
                payload["price_per_kg"] = float(m_price.group(1))
            except ValueError:
                pass
        m_qty = _QTY_RE.search(body)
        if m_qty:
            try:
                payload["quantity_kg"] = float(m_qty.group(1))
            except ValueError:
                pass
    # Try parsing a JSON object appended at end: "...| {\"x\": 1}"
    if "|" in body:
        _, _, tail = body.rpartition("|")
        tail = tail.strip()
        if tail.startswith("{") and tail.endswith("}"):
            try:
                payload.update(json.loads(tail))
            except json.JSONDecodeError:
                pass
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
