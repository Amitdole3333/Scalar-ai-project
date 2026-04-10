"""
Deterministic Grader for Hospital ER Triage environment.

Compares the AI agent's Action against ground truth and produces a
continuous score in [0.0, 1.0] across multiple rubrics.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from .models import Action, Severity


# ---------------------------------------------------------------------------
# Scoring weights  (sum ≈ 1.0)
# ---------------------------------------------------------------------------
WEIGHT_PRIORITY = 0.30
WEIGHT_DOCTOR = 0.25
WEIGHT_CRITICAL = 0.20
WEIGHT_AMBULANCE = 0.10
WEIGHT_TRANSFER = 0.05
WEIGHT_EXPLANATION = 0.10


# ---------------------------------------------------------------------------
# Helper: Kendall-tau-like normalised distance for ranked lists
# ---------------------------------------------------------------------------

def _normalised_rank_distance(predicted: List[str], expected: List[str]) -> float:
    """Return a score in [0, 1] for how well *predicted* matches *expected*.

    Uses a positional penalty: for each item that appears in both lists,
    the penalty is |predicted_rank - expected_rank| / max_possible_distance.
    Items missing from predicted incur maximum penalty.

    1.0 = perfect match, 0.0 = worst possible.
    """
    if not expected:
        return 1.0

    n = len(expected)
    max_penalty = n * (n - 1) / 2 if n > 1 else 1
    penalty = 0.0

    pred_rank = {pid: idx for idx, pid in enumerate(predicted)}

    for exp_idx, pid in enumerate(expected):
        if pid in pred_rank:
            penalty += abs(pred_rank[pid] - exp_idx)
        else:
            penalty += n  # maximum positional penalty

    score = max(0.0, 1.0 - penalty / max(max_penalty, 1))
    return score


# ---------------------------------------------------------------------------
# Helper: set-overlap score
# ---------------------------------------------------------------------------

def _set_overlap_score(predicted: List[str], expected: List[str]) -> float:
    """Jaccard-like overlap in [0, 1]."""
    if not expected and not predicted:
        return 1.0
    if not expected:
        # predicted something when nothing was expected → mild penalty
        return max(0.0, 1.0 - 0.3 * len(predicted))
    pred_set = set(predicted)
    exp_set = set(expected)
    intersection = pred_set & exp_set
    union = pred_set | exp_set
    return len(intersection) / len(union) if union else 1.0


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class GradeResult(BaseModel):
    """Detailed grading breakdown."""
    total_score: float              # 0.0 – 1.0
    priority_score: float
    doctor_score: float
    critical_score: float
    ambulance_score: float
    transfer_score: float
    explanation_score: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": round(self.total_score, 4),
            "priority_score": round(self.priority_score, 4),
            "doctor_score": round(self.doctor_score, 4),
            "critical_handling_score": round(self.critical_score, 4),
            "ambulance_score": round(self.ambulance_score, 4),
            "transfer_score": round(self.transfer_score, 4),
            "explanation_score": round(self.explanation_score, 4),
            "details": self.details,
        }


class Grader:
    """Evaluate an agent Action against ground truth."""

    def grade(
        self,
        action: Action,
        ground_truth: Dict[str, Any],
        patients_data: Optional[List[Dict[str, Any]]] = None,
    ) -> GradeResult:
        details: Dict[str, Any] = {}

        # 1. Priority ordering ------------------------------------------------
        gt_priority = ground_truth.get("priority_order", [])
        priority_score = _normalised_rank_distance(action.priority_order, gt_priority)
        details["priority"] = {
            "predicted": action.priority_order,
            "expected": gt_priority,
        }

        # 2. Doctor assignment -------------------------------------------------
        gt_doctor = ground_truth.get("doctor_assignment", {})
        if gt_doctor:
            correct = 0
            total = len(gt_doctor)
            for pid, expected_type in gt_doctor.items():
                assigned = action.doctor_assignment.get(pid, "")
                if assigned.lower() == expected_type.lower():
                    correct += 1
                elif assigned:  # assigned but wrong type → half credit
                    correct += 0.5
            doctor_score = correct / total
        else:
            doctor_score = 1.0 if not action.doctor_assignment else 0.7
        details["doctor_assignment"] = {
            "predicted": action.doctor_assignment,
            "expected": gt_doctor,
        }

        # 3. Critical patient handling -----------------------------------------
        gt_critical = set(ground_truth.get("critical_patients", []))
        if gt_critical:
            # Check whether critical patients are in top positions of priority
            top_n = action.priority_order[: len(gt_critical)]
            identified = set(top_n) & gt_critical
            critical_score = len(identified) / len(gt_critical)
        else:
            critical_score = 1.0
        details["critical_patients"] = {
            "expected": list(gt_critical),
            "placed_in_top": list(set(action.priority_order[: len(gt_critical)]) if gt_critical else []),
        }

        # 4. Ambulance assignment ----------------------------------------------
        gt_ambulance = ground_truth.get("ambulance_assignment", [])
        ambulance_score = _set_overlap_score(action.ambulance_assignment, gt_ambulance)
        details["ambulance"] = {
            "predicted": action.ambulance_assignment,
            "expected": gt_ambulance,
        }

        # 5. Transfer decision -------------------------------------------------
        gt_transfer = ground_truth.get("transfer_decision", [])
        transfer_score = _set_overlap_score(action.transfer_decision, gt_transfer)
        details["transfer"] = {
            "predicted": action.transfer_decision,
            "expected": gt_transfer,
        }

        # 6. Explanation quality -----------------------------------------------
        explanation_score = self._score_explanation(action.explanation, ground_truth)
        details["explanation_length"] = len(action.explanation)

        # Weighted total -------------------------------------------------------
        total = (
            WEIGHT_PRIORITY * priority_score
            + WEIGHT_DOCTOR * doctor_score
            + WEIGHT_CRITICAL * critical_score
            + WEIGHT_AMBULANCE * ambulance_score
            + WEIGHT_TRANSFER * transfer_score
            + WEIGHT_EXPLANATION * explanation_score
        )

        return GradeResult(
            total_score=total,
            priority_score=priority_score,
            doctor_score=doctor_score,
            critical_score=critical_score,
            ambulance_score=ambulance_score,
            transfer_score=transfer_score,
            explanation_score=explanation_score,
            details=details,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _score_explanation(explanation: str, ground_truth: Dict[str, Any]) -> float:
        """Heuristic scoring for explanation quality.

        Rewards:
          - Non-empty explanation
          - Mentioning critical patient IDs
          - Mentioning resource constraints
          - Reasonable length (50-500 chars)
        """
        if not explanation or not explanation.strip():
            return 0.0

        score = 0.2  # base credit for providing any explanation

        # Mention critical patients
        critical = ground_truth.get("critical_patients", [])
        mentioned = sum(1 for pid in critical if pid in explanation)
        if critical:
            score += 0.3 * (mentioned / len(critical))

        # Mention key reasoning terms
        reasoning_terms = [
            "priority", "critical", "severity", "resource", "doctor",
            "senior", "junior", "ambulance", "bed", "triage", "urgent",
            "vital", "blood pressure", "heart rate", "icu",
        ]
        term_hits = sum(1 for t in reasoning_terms if t.lower() in explanation.lower())
        score += min(0.3, 0.05 * term_hits)

        # Length bonus
        length = len(explanation.strip())
        if 50 <= length <= 800:
            score += 0.2
        elif length > 20:
            score += 0.1

        return min(1.0, score)
