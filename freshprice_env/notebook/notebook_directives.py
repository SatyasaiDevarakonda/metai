"""Notebook directives — verbs the LLM emits inside its Operating Brief.

The agent embeds notebook commands inside a ``## NOTEBOOK`` section of
its brief. This module parses them, executes them against the
AgentNotebook, and provides the long-horizon env with a way to
auto-resolve commitments.

Supported verbs:

  NOTE: <key> -> <value>          # write/overwrite a note
  NOTE_PIN: <key> -> <value>      # pinned note (always re-injected)
  RECALL: <key>                   # included for symmetry; agents can
                                  #   read RECALL output via state
  COMMIT: <kind>:<target>@<tick>  # e.g. COMMIT: inventory_below:fruits:30@800
                                  #              (units 30, due tick 800)
                                  # optional comment after | :
                                  # COMMIT: wrr_above:0.65@600 | mid-week target
  UPDATE_PLAN: <free text>        # replace plan block
  RESOLVE: <commitment_id> <ok|fail> [reason]   # manual mark

Every verb appears on its own line. The parser is lenient — leading/
trailing whitespace and case-mixed verbs are accepted. Unparseable lines
become NotebookDirectiveResult(ok=False, ...) and surface as warnings
without aborting brief execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from freshprice_env.entities import SimulatedMarketState
from freshprice_env.enums import BatchStatus, FarmerOfferStatus, TrendAction
from freshprice_env.notebook.agent_notebook import AgentNotebook, Commitment


# ---------------------------------------------------------------------------
# Directive dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotebookDirective:
    verb: str          # NOTE | NOTE_PIN | RECALL | COMMIT | UPDATE_PLAN | RESOLVE
    raw_line: str
    payload: dict


@dataclass(frozen=True)
class NotebookDirectiveResult:
    directive: NotebookDirective
    ok: bool
    detail: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NOTEBOOK_BLOCK_RE = re.compile(
    r"##\s*NOTEBOOK\s*\n(.*?)(?=\n##\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_VERB_RE = re.compile(
    r"^\s*(NOTE_PIN|NOTE|RECALL|COMMIT|UPDATE_PLAN|RESOLVE)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_COMMIT_RE = re.compile(
    r"^\s*([a-z_]+)\s*:\s*([^@|]+?)\s*@\s*(\d+)\s*(?:\|\s*(.*))?$",
    re.IGNORECASE,
)


def extract_notebook_directives(brief_text: str) -> list[NotebookDirective]:
    """Pull every notebook verb out of an Operating Brief.

    Looks for a ``## NOTEBOOK`` section first; if absent, falls back to
    scanning the entire brief line-by-line. This makes the env tolerant
    of agents that haven't yet learned the section convention.
    """
    text = brief_text or ""
    block_match = _NOTEBOOK_BLOCK_RE.search(text)
    body = block_match.group(1) if block_match else text

    directives: list[NotebookDirective] = []
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = _VERB_RE.match(line)
        if m is None:
            continue
        verb = m.group(1).upper()
        body_text = m.group(2).strip()
        payload = _parse_verb_body(verb, body_text)
        if payload is None:
            continue
        directives.append(
            NotebookDirective(verb=verb, raw_line=line, payload=payload)
        )
    return directives


def _parse_verb_body(verb: str, body: str) -> dict | None:
    if verb in {"NOTE", "NOTE_PIN"}:
        # NOTE: key -> value         (preferred)
        # NOTE: key = value          (also accepted)
        if "->" in body:
            key, _, value = body.partition("->")
        elif "=" in body:
            key, _, value = body.partition("=")
        else:
            return None
        key = key.strip()
        value = value.strip()
        if not key:
            return None
        return {"key": key, "value": value, "pinned": verb == "NOTE_PIN"}

    if verb == "RECALL":
        return {"key": body.strip()}

    if verb == "COMMIT":
        m = _COMMIT_RE.match(body)
        if m is None:
            return None
        kind = m.group(1).strip().lower()
        target = m.group(2).strip()
        due_tick = int(m.group(3))
        note = (m.group(4) or "").strip()
        return {
            "kind": kind,
            "target": target,
            "due_tick": due_tick,
            "note": note,
        }

    if verb == "UPDATE_PLAN":
        return {"plan_text": body}

    if verb == "RESOLVE":
        # RESOLVE: c0001 ok done early
        # RESOLVE: c0002 fail price moved
        parts = body.split(maxsplit=2)
        if len(parts) < 2:
            return None
        cid = parts[0].strip()
        flag = parts[1].strip().lower()
        if flag not in {"ok", "fail"}:
            return None
        reason = parts[2].strip() if len(parts) > 2 else ""
        return {
            "commitment_id": cid,
            "honored": flag == "ok",
            "reason": reason,
        }

    return None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class NotebookDirectiveExecutor:
    """Apply parsed directives to an AgentNotebook.

    Stateless — pass the notebook in for each call. Returns a list of
    per-directive results so the env can surface warnings.
    """

    @staticmethod
    def apply(
        directives: list[NotebookDirective],
        notebook: AgentNotebook,
        current_tick: int,
    ) -> list[NotebookDirectiveResult]:
        results: list[NotebookDirectiveResult] = []
        for d in directives:
            try:
                detail = NotebookDirectiveExecutor._apply_one(
                    d, notebook, current_tick,
                )
                results.append(NotebookDirectiveResult(
                    directive=d, ok=True, detail=detail,
                ))
            except Exception as exc:  # noqa: BLE001 — directive errors are user-style
                results.append(NotebookDirectiveResult(
                    directive=d, ok=False, detail=f"{type(exc).__name__}: {exc}",
                ))
        return results

    @staticmethod
    def _apply_one(
        d: NotebookDirective,
        notebook: AgentNotebook,
        current_tick: int,
    ) -> str:
        if d.verb in {"NOTE", "NOTE_PIN"}:
            entry = notebook.write_note(
                key=d.payload["key"],
                value=d.payload["value"],
                tick=current_tick,
                pinned=bool(d.payload.get("pinned", False)),
            )
            return f"wrote note {entry.key!r}"

        if d.verb == "RECALL":
            entry = notebook.recall(d.payload["key"])
            return f"recalled {entry.key!r} -> {entry.value[:40]}" if entry else "no such note"

        if d.verb == "COMMIT":
            c = notebook.commit(
                kind=d.payload["kind"],
                target=d.payload["target"],
                due_tick=d.payload["due_tick"],
                tick=current_tick,
                note=d.payload.get("note", ""),
            )
            return f"committed {c.commitment_id} ({c.kind}:{c.target}@{c.due_tick})"

        if d.verb == "UPDATE_PLAN":
            text = notebook.update_plan(d.payload["plan_text"], current_tick)
            return f"plan updated ({len(text)} chars)"

        if d.verb == "RESOLVE":
            c = notebook.resolve_commitment(
                commitment_id=d.payload["commitment_id"],
                honored=bool(d.payload["honored"]),
                tick=current_tick,
                reason=d.payload.get("reason", ""),
            )
            if c is None:
                raise ValueError(f"no commitment with id {d.payload['commitment_id']!r}")
            return f"resolved {c.commitment_id} honored={c.honored}"

        raise ValueError(f"unknown verb {d.verb}")


# ---------------------------------------------------------------------------
# Commitment auto-evaluator
# ---------------------------------------------------------------------------

def evaluate_commitment(
    commitment: Commitment,
    state: SimulatedMarketState,
) -> tuple[bool, str]:
    """Decide whether a due commitment was honored, given the current state.

    The set of supported predicates lives here so the env stays generic.
    Unknown predicates return ``(False, "unsupported predicate")`` and
    keep the commitment counted as broken — agents are penalised for
    making promises the system can't grade.

    Predicates:

    - inventory_below: ``<category>:<max_units>``
        honored iff sum of ACTIVE quantity_remaining for category ≤ max
    - inventory_above: ``<category>:<min_units>``
    - wrr_above: ``<min>``  (compares state.wrr)
    - accept_offer: ``<offer_id>`` (status == ACCEPTED by due tick)
    - decline_offer: ``<offer_id>`` (status == DECLINED by due tick)
    - restock_category: ``<category>`` (≥ 1 trend signal APPROVED for category)
    - custom: never auto-resolves (returns broken so manual RESOLVE is required)
    """
    kind = commitment.kind.lower()
    target = commitment.target.strip()

    if kind == "custom":
        return (False, "custom commitments must be RESOLVED manually")

    if kind in {"inventory_below", "inventory_above"}:
        if ":" not in target:
            return (False, "expected '<category>:<units>'")
        cat, _, units_s = target.partition(":")
        cat = cat.strip().lower()
        try:
            target_units = int(units_s.strip())
        except ValueError:
            return (False, "non-integer unit target")
        actual = sum(
            b.quantity_remaining
            for b in state.batches
            if b.status == BatchStatus.ACTIVE and b.category.lower() == cat
        )
        if kind == "inventory_below":
            ok = actual <= target_units
            return (ok, f"actual {actual} vs target ≤ {target_units}")
        ok = actual >= target_units
        return (ok, f"actual {actual} vs target ≥ {target_units}")

    if kind == "wrr_above":
        try:
            target_wrr = float(target)
        except ValueError:
            return (False, "non-float wrr target")
        actual_wrr = state.wrr
        ok = actual_wrr >= target_wrr
        return (ok, f"wrr {actual_wrr:.3f} vs target ≥ {target_wrr:.3f}")

    if kind in {"accept_offer", "decline_offer"}:
        offer_id = target
        match = next(
            (o for o in state.pending_offers if o.offer_id == offer_id),
            None,
        )
        if match is None:
            return (False, f"offer {offer_id!r} not found")
        want = (
            FarmerOfferStatus.ACCEPTED if kind == "accept_offer"
            else FarmerOfferStatus.DECLINED
        )
        ok = match.status == want
        return (ok, f"offer status {match.status.value} (wanted {want.value})")

    if kind == "restock_category":
        cat = target.lower()
        sig = state.trend_signals.get(cat)
        if sig is None:
            return (False, f"no trend signal for category {cat!r}")
        ok = sig.action_taken == TrendAction.APPROVED
        return (ok, f"trend action for {cat} is {sig.action_taken.value}")

    return (False, f"unsupported predicate {kind!r}")
