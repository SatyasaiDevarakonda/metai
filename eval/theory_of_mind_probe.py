"""Theory-of-Mind probe — held-out task that grades multi-agent inference.

Given a partial observation from MarketCommonsEnv (bus messages, farmer
trust summary, competitor recent moves), the LLM is asked three
predictions:

  Q1 — FARMER reserve: "What is farmer X's reserve price (Rs/kg)?"
       Pass: |predicted - true| / true ≤ 0.15
  Q2 — COMPETITOR next: "Will competitor Y send a BID, PRICE_MOVE, or
       neither in the next brief cycle?"
       Pass: predicted matches the observed next action label
  Q3 — DISINFO: "Of these N trend signals, which are paid promotion?"
       Pass: F1 ≥ 0.7 against ground truth

The probe is independent of WRR. It surfaces *whether* the model is
actually doing theory-of-mind, vs. just memorising surface patterns.
Plot ToM-accuracy vs WRR over training; if ToM leads WRR, the model
is reasoning about other agents.

Usage:

    grader = ToMGrader()
    record = grader.run_one_question(
        question_kind="reserve",
        prompt="...",
        model_answer="Rs 36/kg ...",
        ground_truth=37.5,
    )
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class ToMQuestion:
    question_id: str
    kind: str              # "reserve" | "competitor_next" | "disinfo"
    prompt: str
    ground_truth: object   # float | str | list[bool]


@dataclass
class ToMResult:
    question_id: str
    kind: str
    model_answer: str
    parsed_answer: object
    ground_truth: object
    passed: bool
    score: float
    notes: str = ""


@dataclass
class ToMReport:
    n_questions: int = 0
    by_kind_pass_rate: dict = field(default_factory=dict)
    overall_pass_rate: float = 0.0
    overall_score: float = 0.0
    results: list[ToMResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [asdict(r) for r in self.results]
        return d


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

_PRICE_RE = re.compile(r"(?:rs\s*)?(\d+(?:\.\d+)?)", re.IGNORECASE)
_LABEL_RE = re.compile(r"\b(BID|PRICE_MOVE|NEITHER)\b", re.IGNORECASE)
_LIST_RE = re.compile(r"\[([^\]]+)\]")


class ToMGrader:
    """Stateless grader that computes pass/fail for one or many questions."""

    RESERVE_TOLERANCE = 0.15

    @classmethod
    def grade(
        cls,
        question: ToMQuestion,
        model_answer: str,
    ) -> ToMResult:
        if question.kind == "reserve":
            return cls._grade_reserve(question, model_answer)
        if question.kind == "competitor_next":
            return cls._grade_competitor_next(question, model_answer)
        if question.kind == "disinfo":
            return cls._grade_disinfo(question, model_answer)
        return ToMResult(
            question_id=question.question_id, kind=question.kind,
            model_answer=model_answer, parsed_answer=None,
            ground_truth=question.ground_truth,
            passed=False, score=0.0, notes=f"unknown kind {question.kind}",
        )

    # --- Reserve ---------------------------------------------------------

    @classmethod
    def _grade_reserve(
        cls, q: ToMQuestion, model_answer: str,
    ) -> ToMResult:
        truth = float(q.ground_truth)  # type: ignore[arg-type]
        m = _PRICE_RE.search(model_answer or "")
        if m is None:
            return ToMResult(
                question_id=q.question_id, kind=q.kind,
                model_answer=model_answer, parsed_answer=None,
                ground_truth=truth, passed=False, score=0.0,
                notes="no number in answer",
            )
        try:
            pred = float(m.group(1))
        except ValueError:
            return ToMResult(
                question_id=q.question_id, kind=q.kind,
                model_answer=model_answer, parsed_answer=None,
                ground_truth=truth, passed=False, score=0.0,
                notes="non-float number",
            )
        rel_err = abs(pred - truth) / max(truth, 1e-3)
        passed = rel_err <= cls.RESERVE_TOLERANCE
        # Score: 1.0 if exact, 0.0 if more than 2× tolerance off
        score = max(0.0, 1.0 - rel_err / (cls.RESERVE_TOLERANCE * 2.0))
        return ToMResult(
            question_id=q.question_id, kind=q.kind,
            model_answer=model_answer, parsed_answer=pred,
            ground_truth=truth, passed=passed, score=round(score, 3),
            notes=f"rel_err={rel_err:.3f}",
        )

    # --- Competitor next -------------------------------------------------

    @classmethod
    def _grade_competitor_next(
        cls, q: ToMQuestion, model_answer: str,
    ) -> ToMResult:
        truth = str(q.ground_truth).upper()
        m = _LABEL_RE.search(model_answer or "")
        if m is None:
            return ToMResult(
                question_id=q.question_id, kind=q.kind,
                model_answer=model_answer, parsed_answer=None,
                ground_truth=truth, passed=False, score=0.0,
                notes="no label found",
            )
        pred = m.group(1).upper()
        passed = (pred == truth)
        return ToMResult(
            question_id=q.question_id, kind=q.kind,
            model_answer=model_answer, parsed_answer=pred,
            ground_truth=truth, passed=passed, score=1.0 if passed else 0.0,
        )

    # --- Disinfo (F1) ----------------------------------------------------

    @classmethod
    def _grade_disinfo(
        cls, q: ToMQuestion, model_answer: str,
    ) -> ToMResult:
        truth_list = list(q.ground_truth)  # type: ignore[arg-type]
        # Try JSON list first
        pred_list = None
        try:
            obj = json.loads(model_answer)
            if isinstance(obj, list):
                pred_list = [bool(x) for x in obj]
        except (json.JSONDecodeError, TypeError):
            m = _LIST_RE.search(model_answer or "")
            if m:
                items = [s.strip().lower() for s in m.group(1).split(",")]
                pred_list = [
                    s in ("true", "1", "yes", "paid", "t") for s in items
                ]
        if pred_list is None or len(pred_list) != len(truth_list):
            return ToMResult(
                question_id=q.question_id, kind=q.kind,
                model_answer=model_answer, parsed_answer=pred_list,
                ground_truth=truth_list, passed=False, score=0.0,
                notes="list length mismatch or unparsable",
            )
        tp = sum(1 for p, t in zip(pred_list, truth_list) if p and t)
        fp = sum(1 for p, t in zip(pred_list, truth_list) if p and not t)
        fn = sum(1 for p, t in zip(pred_list, truth_list) if not p and t)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        passed = f1 >= 0.7
        return ToMResult(
            question_id=q.question_id, kind=q.kind,
            model_answer=model_answer, parsed_answer=pred_list,
            ground_truth=truth_list, passed=passed,
            score=round(f1, 3),
            notes=f"precision={precision:.2f}, recall={recall:.2f}",
        )

    # ------------------------------------------------------------------
    # Run a batch
    # ------------------------------------------------------------------

    @classmethod
    def run_batch(
        cls,
        questions: list[ToMQuestion],
        model_fn,    # Callable[[str], str]
    ) -> ToMReport:
        results: list[ToMResult] = []
        for q in questions:
            ans = model_fn(q.prompt)
            results.append(cls.grade(q, ans))

        by_kind: dict[str, list[ToMResult]] = {}
        for r in results:
            by_kind.setdefault(r.kind, []).append(r)
        rates = {
            kind: round(
                sum(1 for r in rs if r.passed) / len(rs), 3
            ) if rs else 0.0
            for kind, rs in by_kind.items()
        }
        overall_pass = sum(1 for r in results if r.passed) / len(results) if results else 0.0
        overall_score = sum(r.score for r in results) / len(results) if results else 0.0

        return ToMReport(
            n_questions=len(results),
            by_kind_pass_rate=rates,
            overall_pass_rate=round(overall_pass, 3),
            overall_score=round(overall_score, 3),
            results=results,
        )


# ---------------------------------------------------------------------------
# Question synthesis from MarketCommonsEnv state
# ---------------------------------------------------------------------------

def synthesize_questions_from_state(
    market_state_dict: dict,
    farmer_pool: dict,           # farmer_id -> FarmerAgent
    influencer_signals: list,    # list[TaggedTrendSignal] from InfluencerAgent
    competitor_actions: list,    # last step's competitor_actions
) -> list[ToMQuestion]:
    """Build a small, deterministic question set from an env snapshot.

    Used by eval/run_quick_eval.py and the dashboard's "ToM" panel.
    """
    questions: list[ToMQuestion] = []

    # Q1: pick first farmer with > 1 interaction → ask reserve
    for fid, farmer in farmer_pool.items():
        rep = farmer.reputation()
        if rep.total_interactions >= 1:
            adjusted = rep.adjusted_reserve()
            questions.append(ToMQuestion(
                question_id=f"q1_reserve_{fid}",
                kind="reserve",
                prompt=(
                    f"Given farmer {rep.farmer_name} (trust={rep.trust_score:.2f}, "
                    f"{rep.total_accepted}/{rep.total_interactions} accepted), "
                    f"what is their current reserve price in Rs/kg? "
                    f"Answer with a single number."
                ),
                ground_truth=adjusted,
            ))
            break

    # Q2: competitor next action label
    if competitor_actions:
        last_kind = competitor_actions[-1].get("kind", "BID")
        # Truth = same kind (predict continuation) — a simple proxy
        questions.append(ToMQuestion(
            question_id="q2_comp_next",
            kind="competitor_next",
            prompt=(
                "Recent competitor actions:\n"
                + "\n".join(f"- {a.get('kind')}: {a.get('body','')[:80]}"
                            for a in competitor_actions[-3:])
                + "\n\nWhich verb is the competitor most likely to use next? "
                  "Answer with one of: BID, PRICE_MOVE, NEITHER."
            ),
            ground_truth="BID" if last_kind == "OFFER_BID" else (
                "PRICE_MOVE" if last_kind == "PRICE_MOVE" else "NEITHER"
            ),
        ))

    # Q3: disinfo F1
    if influencer_signals:
        truth_flags = [
            getattr(s, "veracity", None) is not None
            and s.veracity.value == "PAID_PROMOTION"
            for s in influencer_signals[:6]
        ]
        signal_brief = "\n".join(
            f"- category={s.signal.category} score={s.signal.composite_score:.0f} "
            f"search_velocity={s.consumer_search_velocity:.2f} "
            f"co_purchase={s.co_purchase_uptick:.2f}"
            for s in influencer_signals[:6]
        )
        questions.append(ToMQuestion(
            question_id="q3_disinfo",
            kind="disinfo",
            prompt=(
                "For each trend signal below, decide whether it is "
                "PAID_PROMOTION (true) or GENUINE (false). Return a JSON "
                "array of booleans in the same order as the signals.\n\n"
                + signal_brief
            ),
            ground_truth=truth_flags,
        ))

    return questions
