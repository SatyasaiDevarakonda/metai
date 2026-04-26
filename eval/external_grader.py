"""External LLM-judge graders (Patronus / Halluminate) for hackathon
sub-prizes (Roadmap #7).

What this is for
================

The hackathon's Patronus and Halluminate sub-prizes reward submissions
that pipe their model outputs through an external LLM-judge for
quality / hallucination scoring. This module provides a single
``ExternalGrader`` interface with three concrete adapters:

  * ``PatronusGrader``      -- POSTs to the Patronus AI evaluation API
                               (https://docs.patronus.ai). Requires
                               ``PATRONUS_API_KEY``.
  * ``HalluminateGrader``   -- POSTs to the Halluminate API. Requires
                               ``HALLUMINATE_API_KEY``.
  * ``LocalHeuristicGrader``-- the no-API-key fallback. Uses the same
                               6-section / parse-success heuristics the
                               internal grader already uses, so the
                               pipeline still produces a score for every
                               brief even offline.

How it plugs in
===============

``score_brief(prompt, brief)`` returns a ``GraderScore`` with:

  * ``overall``       float in [0, 1]
  * ``hallucination`` float in [0, 1] (1 = fully grounded, 0 = made up)
  * ``coherence``     float in [0, 1]
  * ``raw``           dict (provider response, kept for receipts)

The dashboard's "Before vs After RL" panel can show the external score
next to the internal SES so judges can see both. The
``inference_comparison.py`` CLI also accepts ``--use-external-grader``
to add an "external_score" column to ``data/comparison_results.json``.

Failure mode: if the external API is down or the key is invalid,
``score_brief`` falls back to the local heuristic (it never raises).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score type
# ---------------------------------------------------------------------------

@dataclass
class GraderScore:
    """Common shape returned by every grader adapter."""

    overall: float            # [0, 1]
    hallucination: float      # [0, 1] (1 = grounded, 0 = hallucinated)
    coherence: float          # [0, 1]
    grader: str               # "patronus" / "halluminate" / "local"
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 4),
            "hallucination": round(self.hallucination, 4),
            "coherence": round(self.coherence, 4),
            "grader": self.grader,
        }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ExternalGrader(Protocol):
    """Anything that can score a (prompt, brief) into a GraderScore."""

    name: str

    def score_brief(self, prompt: str, brief: str) -> GraderScore: ...


# ---------------------------------------------------------------------------
# Local heuristic (no API key required) — also the fallback
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = (
    "SITUATION:", "SIGNAL ANALYSIS:", "VIABILITY CHECK:",
    "RECOMMENDATION:", "DIRECTIVE:", "CONFIDENCE:",
)


class LocalHeuristicGrader:
    """No-API fallback: scores from the brief's structural quality.

    Uses the same 6-section schema the internal grader requires, plus a
    rough hallucination heuristic (penalises numbers in the brief that
    don't appear in the prompt).
    """

    name = "local"

    def score_brief(self, prompt: str, brief: str) -> GraderScore:
        # Coherence: fraction of required sections present, in order.
        present = [s for s in _REQUIRED_SECTIONS if s in brief]
        coherence = len(present) / len(_REQUIRED_SECTIONS)

        # Hallucination heuristic: any numeric token in the brief that
        # isn't in the prompt is 1 point against. Capped at 5 violations
        # so a single rogue number doesn't tank a long brief.
        prompt_nums = set(re.findall(r"\d+\.?\d*", prompt))
        brief_nums = re.findall(r"\d+\.?\d*", brief)
        violations = sum(1 for n in brief_nums if n not in prompt_nums)
        hallucination = max(0.0, 1.0 - min(violations, 5) / 5.0)

        # Parse-success bonus: a valid JSON DIRECTIVE block.
        parse_bonus = 0.0
        m = re.search(r"DIRECTIVE:\s*(\{.*?\})", brief, re.DOTALL)
        if m:
            try:
                json.loads(m.group(1))
                parse_bonus = 0.1
            except json.JSONDecodeError:
                pass

        overall = min(1.0, 0.5 * coherence + 0.4 * hallucination + parse_bonus)
        return GraderScore(
            overall=overall,
            hallucination=hallucination,
            coherence=coherence,
            grader=self.name,
            raw={"violations": violations, "sections_present": len(present)},
        )


# ---------------------------------------------------------------------------
# HTTP helper (stdlib-only so we avoid adding `requests` to deps)
# ---------------------------------------------------------------------------

def _post_json(url: str, headers: dict, payload: dict, timeout: float = 8.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", **headers,
    }, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Patronus adapter
# ---------------------------------------------------------------------------

class PatronusGrader:
    """Thin client for the Patronus evaluation endpoint.

    The exact endpoint URL and payload field names occasionally change
    upstream. The adapter is intentionally permissive: as long as the
    response carries a numeric ``overall`` (or ``score``) field, we map
    it through. Hallucination/coherence are pulled from named subscores
    when present and fall back to ``overall`` otherwise.
    """

    name = "patronus"

    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("PATRONUS_API_KEY", "")
        self.base_url = (base_url
                         or os.environ.get("PATRONUS_BASE_URL")
                         or "https://api.patronus.ai/v1/evaluate")
        self._fallback = LocalHeuristicGrader()

    def score_brief(self, prompt: str, brief: str) -> GraderScore:
        if not self.api_key:
            logger.info("PATRONUS_API_KEY not set; using local fallback")
            return self._fallback.score_brief(prompt, brief)

        payload = {
            "criteria": ["hallucination", "coherence", "task_quality"],
            "input": prompt,
            "output": brief,
            "evaluators": ["patronus-base"],
        }
        try:
            resp = _post_json(
                self.base_url,
                {"Authorization": f"Bearer {self.api_key}"},
                payload,
            )
            scores = resp.get("scores") or resp
            overall = float(
                scores.get("overall")
                or scores.get("task_quality")
                or scores.get("score")
                or 0.5
            )
            hallucination = float(scores.get("hallucination", overall))
            coherence = float(scores.get("coherence", overall))
            return GraderScore(
                overall=overall,
                hallucination=hallucination,
                coherence=coherence,
                grader=self.name,
                raw=resp,
            )
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as e:
            logger.warning("Patronus call failed (%s); falling back", e)
            return self._fallback.score_brief(prompt, brief)


# ---------------------------------------------------------------------------
# Halluminate adapter
# ---------------------------------------------------------------------------

class HalluminateGrader:
    """Thin client for Halluminate's hallucination-grading endpoint."""

    name = "halluminate"

    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("HALLUMINATE_API_KEY", "")
        self.base_url = (base_url
                         or os.environ.get("HALLUMINATE_BASE_URL")
                         or "https://api.halluminate.ai/v1/grade")
        self._fallback = LocalHeuristicGrader()

    def score_brief(self, prompt: str, brief: str) -> GraderScore:
        if not self.api_key:
            logger.info("HALLUMINATE_API_KEY not set; using local fallback")
            return self._fallback.score_brief(prompt, brief)

        payload = {"context": prompt, "completion": brief}
        try:
            resp = _post_json(
                self.base_url,
                {"Authorization": f"Bearer {self.api_key}"},
                payload,
            )
            # Halluminate returns hallucination_score where 1.0 == grounded.
            hallucination = float(resp.get("hallucination_score", 0.5))
            coherence = float(resp.get("coherence_score", hallucination))
            overall = (hallucination + coherence) / 2.0
            return GraderScore(
                overall=overall,
                hallucination=hallucination,
                coherence=coherence,
                grader=self.name,
                raw=resp,
            )
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as e:
            logger.warning("Halluminate call failed (%s); falling back", e)
            return self._fallback.score_brief(prompt, brief)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_external_grader(name: str | None = None) -> ExternalGrader:
    """Build a grader by name; default reads ``EXTERNAL_GRADER`` env var.

    Returns ``LocalHeuristicGrader`` when no remote provider is selected
    or when keys are missing -- never raises.
    """
    name = (name or os.environ.get("EXTERNAL_GRADER", "")).strip().lower()
    if name == "patronus":
        return PatronusGrader()
    if name == "halluminate":
        return HalluminateGrader()
    return LocalHeuristicGrader()
