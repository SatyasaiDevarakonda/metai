"""AgentNotebook — keyed scratchpad + commitment ledger that survives context truncation.

The notebook is the agent's externalized memory. It has three sections:

  1. **notes**: free-form key→value pairs the agent writes (and reads
     back later). Every NOTE has a `pinned` flag — pinned notes are
     always re-injected at the top of the prompt regardless of context
     budget.

  2. **commitments**: future-dated promises the agent made — e.g.,
     "by tick 800, vegetables inventory must be below 30 units." These
     drive the plan-adherence reward (r4): each commitment that comes
     due is checked against current state and scored.

  3. **plan**: a single mutable text block holding the agent's high-level
     strategy. Replaced (not appended) by UPDATE_PLAN.

Capacity limits are enforced in code so the notebook can't grow without
bound — old non-pinned entries are evicted FIFO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Capacity limits
# ---------------------------------------------------------------------------

NOTEBOOK_MAX_NOTES: int = 64
NOTEBOOK_MAX_COMMITMENTS: int = 32
NOTEBOOK_PROMPT_TAIL_LINES: int = 12   # how many recent notes the prompt shows by default
NOTEBOOK_PLAN_MAX_CHARS: int = 800     # plan text cap


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotebookEntry:
    """A single note the LLM has written."""

    key: str
    value: str
    written_at_tick: int
    pinned: bool = False


@dataclass
class Commitment:
    """A future-dated promise.

    The agent says: "by tick `due_tick`, the predicate `kind:target` must
    hold." When the env reaches `due_tick`, the predicate is evaluated.

    Supported kinds (more can be added in NotebookDirectiveExecutor):
      - `inventory_below`: the *target* is "<category>:<max_units>"
      - `inventory_above`: "<category>:<min_units>"
      - `wrr_above`: "<min_wrr_float>"
      - `accept_offer`: "<offer_id>" (must have been accepted by due_tick)
      - `decline_offer`: "<offer_id>" (must have been declined by due_tick)
      - `restock_category`: "<category>" (a TREND order placed by due_tick)
      - `custom`: free-form, never auto-resolves (manual mark only)
    """

    commitment_id: str
    kind: str
    target: str
    due_tick: int
    written_at_tick: int
    note: str = ""
    resolved: bool = False
    honored: bool | None = None         # set when resolved
    resolution_tick: int | None = None
    resolution_reason: str = ""

    def is_due(self, current_tick: int) -> bool:
        return not self.resolved and current_tick >= self.due_tick


# ---------------------------------------------------------------------------
# AgentNotebook
# ---------------------------------------------------------------------------

class AgentNotebook:
    """Per-episode externalized memory for one StoreAgent.

    Lifecycle: a fresh notebook is created at env.reset() and lives for
    the duration of the episode. The 30-day env keeps it; the 7-day env
    can use it too but it's optional there.
    """

    def __init__(self) -> None:
        self._notes: list[NotebookEntry] = []
        self._notes_index: dict[str, int] = {}      # key → position in _notes
        self._commitments: list[Commitment] = []
        self._plan: str = ""
        self._plan_updated_at_tick: int = 0
        self._next_commitment_seq: int = 0
        # Stats for reward & replay
        self._honored_count: int = 0
        self._broken_count: int = 0
        # Action log: (tick, verb, payload) — used by the dashboard replay
        self._action_log: list[tuple[int, str, str]] = []

    # ------------------------------------------------------------------
    # NOTE / RECALL
    # ------------------------------------------------------------------

    def write_note(self, key: str, value: str, tick: int, pinned: bool = False) -> NotebookEntry:
        """Write or overwrite a keyed note."""
        key = (key or "").strip()
        if not key:
            raise ValueError("note key cannot be empty")
        entry = NotebookEntry(
            key=key,
            value=str(value),
            written_at_tick=int(tick),
            pinned=bool(pinned),
        )
        if key in self._notes_index:
            self._notes[self._notes_index[key]] = entry
        else:
            self._notes.append(entry)
            self._notes_index[key] = len(self._notes) - 1
        self._evict_if_oversize()
        self._action_log.append((tick, "NOTE", f"{key} -> {value[:60]}"))
        return entry

    def recall(self, key: str) -> NotebookEntry | None:
        idx = self._notes_index.get((key or "").strip())
        if idx is None:
            return None
        return self._notes[idx]

    def all_notes(self) -> list[NotebookEntry]:
        return list(self._notes)

    def pinned_notes(self) -> list[NotebookEntry]:
        return [n for n in self._notes if n.pinned]

    def recent_notes(self, n: int = NOTEBOOK_PROMPT_TAIL_LINES) -> list[NotebookEntry]:
        """Most recently written N notes, newest first."""
        ordered = sorted(self._notes, key=lambda e: e.written_at_tick, reverse=True)
        return ordered[:n]

    # ------------------------------------------------------------------
    # COMMIT
    # ------------------------------------------------------------------

    def commit(
        self,
        kind: str,
        target: str,
        due_tick: int,
        tick: int,
        note: str = "",
    ) -> Commitment:
        """Record a future-dated commitment."""
        kind = (kind or "").strip().lower()
        if not kind:
            raise ValueError("commitment kind cannot be empty")
        cid = f"c{self._next_commitment_seq:04d}"
        self._next_commitment_seq += 1
        c = Commitment(
            commitment_id=cid,
            kind=kind,
            target=str(target),
            due_tick=int(due_tick),
            written_at_tick=int(tick),
            note=str(note),
        )
        self._commitments.append(c)
        if len(self._commitments) > NOTEBOOK_MAX_COMMITMENTS:
            # Evict oldest *resolved* commitment first; if none, oldest
            for i, existing in enumerate(self._commitments):
                if existing.resolved:
                    self._commitments.pop(i)
                    break
            else:
                self._commitments.pop(0)
        self._action_log.append((tick, "COMMIT", f"{kind}:{target}@{due_tick}"))
        return c

    def open_commitments(self) -> list[Commitment]:
        return [c for c in self._commitments if not c.resolved]

    def all_commitments(self) -> list[Commitment]:
        return list(self._commitments)

    def due_commitments(self, current_tick: int) -> list[Commitment]:
        return [c for c in self._commitments if c.is_due(current_tick)]

    def resolve_commitment(
        self,
        commitment_id: str,
        honored: bool,
        tick: int,
        reason: str = "",
    ) -> Commitment | None:
        for i, c in enumerate(self._commitments):
            if c.commitment_id == commitment_id:
                c.resolved = True
                c.honored = bool(honored)
                c.resolution_tick = int(tick)
                c.resolution_reason = str(reason)
                self._commitments[i] = c
                if honored:
                    self._honored_count += 1
                else:
                    self._broken_count += 1
                self._action_log.append(
                    (tick, "RESOLVE", f"{commitment_id} honored={honored}")
                )
                return c
        return None

    def auto_resolve_due(
        self,
        current_tick: int,
        evaluator: Callable[[Commitment], tuple[bool, str]],
    ) -> list[Commitment]:
        """Auto-resolve every commitment whose due_tick has passed.

        ``evaluator(commitment)`` must return ``(honored, reason)``. The
        long-horizon env supplies an evaluator that checks the current
        SimulatedMarketState against the commitment's predicate.
        Returns the list of commitments resolved this tick.
        """
        resolved: list[Commitment] = []
        for c in self._commitments:
            if c.is_due(current_tick):
                honored, reason = evaluator(c)
                self.resolve_commitment(
                    c.commitment_id, honored, current_tick, reason,
                )
                resolved.append(c)
        return resolved

    # ------------------------------------------------------------------
    # PLAN
    # ------------------------------------------------------------------

    def update_plan(self, plan_text: str, tick: int) -> str:
        """Replace the plan block. Truncated to NOTEBOOK_PLAN_MAX_CHARS."""
        text = (plan_text or "").strip()
        if len(text) > NOTEBOOK_PLAN_MAX_CHARS:
            text = text[:NOTEBOOK_PLAN_MAX_CHARS] + "…"
        self._plan = text
        self._plan_updated_at_tick = int(tick)
        self._action_log.append((tick, "PLAN", text[:80]))
        return text

    @property
    def plan(self) -> str:
        return self._plan

    @property
    def plan_updated_at_tick(self) -> int:
        return self._plan_updated_at_tick

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def honored_count(self) -> int:
        return self._honored_count

    @property
    def broken_count(self) -> int:
        return self._broken_count

    @property
    def open_count(self) -> int:
        return sum(1 for c in self._commitments if not c.resolved)

    def adherence_score(self) -> float:
        """Fraction of *resolved* commitments that were honored. 0..1.

        If nothing has resolved yet, returns 0.0 (neutral: no signal).
        """
        denom = self._honored_count + self._broken_count
        if denom == 0:
            return 0.0
        return self._honored_count / denom

    # ------------------------------------------------------------------
    # Prompt-side rendering
    # ------------------------------------------------------------------

    def render_prompt_block(
        self,
        current_tick: int,
        max_lines: int = NOTEBOOK_PROMPT_TAIL_LINES,
    ) -> str:
        """Build the NOTEBOOK section that gets injected into the LLM prompt.

        Always shows the plan, all pinned notes, and the most recent
        ``max_lines`` regular notes. Truncated to keep prompts small.
        """
        lines: list[str] = ["## NOTEBOOK (your durable memory across briefs)"]

        if self._plan:
            lines.append(f"  PLAN [updated tick {self._plan_updated_at_tick}]: {self._plan}")
        else:
            lines.append("  PLAN: (none — write one with UPDATE_PLAN)")

        pinned = self.pinned_notes()
        if pinned:
            lines.append("  PINNED NOTES:")
            for n in pinned:
                lines.append(f"    [t{n.written_at_tick}] {n.key}: {n.value}")

        regular = [n for n in self.recent_notes(max_lines) if not n.pinned]
        if regular:
            lines.append(f"  RECENT NOTES (last {len(regular)}):")
            for n in regular:
                lines.append(f"    [t{n.written_at_tick}] {n.key}: {n.value}")

        open_c = self.open_commitments()
        if open_c:
            lines.append(f"  OPEN COMMITMENTS ({len(open_c)}):")
            for c in open_c[:8]:  # show at most 8
                ticks_to_due = max(0, c.due_tick - current_tick)
                lines.append(
                    f"    {c.commitment_id} [{c.kind}:{c.target}] "
                    f"due in {ticks_to_due}t — {c.note[:48]}"
                )

        if self._honored_count + self._broken_count > 0:
            lines.append(
                f"  TRACK RECORD: honored {self._honored_count}, "
                f"broken {self._broken_count} "
                f"(adherence {self.adherence_score():.2f})"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Replay & debug
    # ------------------------------------------------------------------

    def to_replay(self) -> dict:
        """Snapshot for the dashboard / counterfactual replay tool."""
        return {
            "plan": self._plan,
            "plan_updated_at_tick": self._plan_updated_at_tick,
            "notes": [
                {
                    "key": n.key, "value": n.value,
                    "tick": n.written_at_tick, "pinned": n.pinned,
                }
                for n in self._notes
            ],
            "commitments": [
                {
                    "id": c.commitment_id,
                    "kind": c.kind,
                    "target": c.target,
                    "due_tick": c.due_tick,
                    "resolved": c.resolved,
                    "honored": c.honored,
                    "note": c.note,
                }
                for c in self._commitments
            ],
            "honored_count": self._honored_count,
            "broken_count": self._broken_count,
            "adherence_score": round(self.adherence_score(), 4),
            "action_log": [
                {"tick": t, "verb": v, "payload": p}
                for (t, v, p) in self._action_log[-200:]
            ],
        }

    def reset(self) -> None:
        """Wipe the notebook for a fresh episode."""
        self._notes.clear()
        self._notes_index.clear()
        self._commitments.clear()
        self._plan = ""
        self._plan_updated_at_tick = 0
        self._next_commitment_seq = 0
        self._honored_count = 0
        self._broken_count = 0
        self._action_log.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_oversize(self) -> None:
        """Evict oldest non-pinned notes if over capacity."""
        if len(self._notes) <= NOTEBOOK_MAX_NOTES:
            return
        # Find oldest non-pinned
        for i, n in enumerate(self._notes):
            if not n.pinned:
                evicted = self._notes.pop(i)
                # Rebuild index after pop
                self._notes_index = {n.key: idx for idx, n in enumerate(self._notes)}
                self._action_log.append(
                    (evicted.written_at_tick, "EVICT", evicted.key)
                )
                return
        # All pinned — evict oldest pinned anyway
        evicted = self._notes.pop(0)
        self._notes_index = {n.key: idx for idx, n in enumerate(self._notes)}
        self._action_log.append((evicted.written_at_tick, "EVICT_PINNED", evicted.key))
