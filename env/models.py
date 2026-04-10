"""Data models for the Hospital ER Triage environment."""

from __future__ import annotations

import enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(enum.IntEnum):
    """Ground-truth severity (hidden from agent)."""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class Condition(str, enum.Enum):
    """Observable patient condition."""
    STABLE = "stable"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DoctorType(str, enum.Enum):
    SENIOR = "senior"
    JUNIOR = "junior"


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class Patient(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    symptoms: List[str]
    bp_systolic: int          # e.g. 120
    bp_diastolic: int         # e.g. 80
    heart_rate: int            # bpm
    condition: Condition
    waiting_time: int = 0      # time steps waited
    hidden_severity: Severity = Severity.LOW   # ground truth, hidden
    misleading_info: bool = False               # Problem 11
    escalation_rate: float = 0.0                # Problem 12 – per-step severity increase chance
    _escalated: bool = False

    # --- helpers -----------------------------------------------------------

    def observable_dict(self) -> Dict[str, Any]:
        """Return the observation visible to the agent (no ground truth)."""
        return {
            "id": self.id,
            "symptoms": self.symptoms,
            "bp": f"{self.bp_systolic}/{self.bp_diastolic}",
            "heart_rate": self.heart_rate,
            "condition": self.condition.value,
            "waiting_time": self.waiting_time,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full dict including hidden info (for grading)."""
        d = self.observable_dict()
        d["hidden_severity"] = self.hidden_severity.name
        d["misleading_info"] = self.misleading_info
        d["escalation_rate"] = self.escalation_rate
        return d


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class Resources(BaseModel):
    senior_doctors: int = 2
    junior_doctors: int = 3
    nurses: int = 5
    icu_beds: int = 3
    general_beds: int = 10
    ambulances: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_doctors": {
                "senior": self.senior_doctors,
                "junior": self.junior_doctors,
            },
            "available_nurses": self.nurses,
            "available_ICU_beds": self.icu_beds,
            "available_general_beds": self.general_beds,
            "ambulance_available": self.ambulances,
        }


# ---------------------------------------------------------------------------
# Observation  (returned to agent)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    patients: List[Dict[str, Any]]
    resources: Dict[str, Any]
    current_time_step: int
    queue_state: List[str]       # patient IDs in arrival order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patients": self.patients,
            "resources": self.resources,
            "current_time_step": self.current_time_step,
            "queue_state": self.queue_state,
        }


# ---------------------------------------------------------------------------
# Action  (expected from agent)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    priority_order: List[str]                                              # patient IDs
    doctor_assignment: Dict[str, str]                                       # patient_id -> "senior" | "junior"
    ambulance_assignment: List[str] = Field(default_factory=list)           # patient IDs needing ambulance
    transfer_decision: List[str] = Field(default_factory=list)              # patient IDs to transfer out
    explanation: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        return cls(
            priority_order=d.get("priority_order", []),
            doctor_assignment=d.get("doctor_assignment", {}),
            ambulance_assignment=d.get("ambulance_assignment", []),
            transfer_decision=d.get("transfer_decision", []),
            explanation=d.get("explanation", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority_order": self.priority_order,
            "doctor_assignment": self.doctor_assignment,
            "ambulance_assignment": self.ambulance_assignment,
            "transfer_decision": self.transfer_decision,
            "explanation": self.explanation,
        }
