"""
Pre-defined tasks (scenarios) for the Hospital ER Triage environment.

Each task is a dict consumed by HospitalEnv.reset(task=...) containing:
  - name, difficulty
  - patients: list of Patient kwargs
  - resources: Resources kwargs
  - dynamic_events: list of events that fire at specific time steps
  - ground_truth: the ideal Action the agent should produce

Covers all 12 required problems across Easy / Medium / Hard.
"""

from __future__ import annotations

from typing import Dict, Any, List

from .models import Condition, Severity, DoctorType


# ====================================================================
# EASY TASK — 2 patients, clear critical, no resource conflict
# Problems exercised: 1 (triage), 2 (queue), 3 (delay), 6 (ambulance)
# ====================================================================

EASY_TASK: Dict[str, Any] = {
    "name": "easy",
    "difficulty": "easy",
    "description": (
        "Two patients arrive. One has a clear cardiac emergency "
        "(critical), the other a minor sprain (stable). Plenty of resources."
    ),
    "patients": [
        {
            "id": "P1",
            "symptoms": ["chest_pain", "shortness_of_breath", "sweating"],
            "bp_systolic": 180,
            "bp_diastolic": 110,
            "heart_rate": 130,
            "condition": Condition.CRITICAL,
            "hidden_severity": Severity.CRITICAL,
            "misleading_info": False,
            "escalation_rate": 0.0,
        },
        {
            "id": "P2",
            "symptoms": ["ankle_pain", "swelling"],
            "bp_systolic": 120,
            "bp_diastolic": 78,
            "heart_rate": 72,
            "condition": Condition.STABLE,
            "hidden_severity": Severity.LOW,
            "misleading_info": False,
            "escalation_rate": 0.0,
        },
    ],
    "resources": {
        "senior_doctors": 2,
        "junior_doctors": 2,
        "nurses": 4,
        "icu_beds": 2,
        "general_beds": 5,
        "ambulances": 2,
    },
    "dynamic_events": [],
    "ground_truth": {
        "priority_order": ["P1", "P2"],
        "doctor_assignment": {"P1": "senior", "P2": "junior"},
        "ambulance_assignment": ["P1"],
        "transfer_decision": [],
        "critical_patients": ["P1"],
    },
    "max_steps": 3,
}


# ====================================================================
# MEDIUM TASK — 4 patients, moderate ambiguity, limited doctors
# Problems: 1-7 (Level 1) + 10 (multi-param)
# ====================================================================

MEDIUM_TASK: Dict[str, Any] = {
    "name": "medium",
    "difficulty": "medium",
    "description": (
        "Four patients with varying acuity. Only 1 senior doctor and 1 junior. "
        "Some ambiguity in condition tagging; multi-parameter reasoning needed."
    ),
    "patients": [
        {
            "id": "P1",
            "symptoms": ["severe_headache", "blurred_vision", "nausea"],
            "bp_systolic": 200,
            "bp_diastolic": 120,
            "heart_rate": 105,
            "condition": Condition.CRITICAL,
            "hidden_severity": Severity.CRITICAL,
            "misleading_info": False,
            "escalation_rate": 0.0,
        },
        {
            "id": "P2",
            "symptoms": ["abdominal_pain", "vomiting"],
            "bp_systolic": 135,
            "bp_diastolic": 88,
            "heart_rate": 96,
            "condition": Condition.UNKNOWN,
            "hidden_severity": Severity.HIGH,
            "misleading_info": False,
            "escalation_rate": 0.1,
        },
        {
            "id": "P3",
            "symptoms": ["fever", "cough", "fatigue"],
            "bp_systolic": 118,
            "bp_diastolic": 76,
            "heart_rate": 82,
            "condition": Condition.STABLE,
            "hidden_severity": Severity.MODERATE,
            "misleading_info": False,
            "escalation_rate": 0.0,
        },
        {
            "id": "P4",
            "symptoms": ["laceration", "bleeding"],
            "bp_systolic": 100,
            "bp_diastolic": 65,
            "heart_rate": 110,
            "condition": Condition.UNKNOWN,
            "hidden_severity": Severity.HIGH,
            "misleading_info": False,
            "escalation_rate": 0.05,
        },
    ],
    "resources": {
        "senior_doctors": 1,
        "junior_doctors": 1,
        "nurses": 3,
        "icu_beds": 1,
        "general_beds": 3,
        "ambulances": 1,
    },
    "dynamic_events": [
        {
            "time_step": 2,
            "type": "condition_change",
            "patient_id": "P2",
            "new_condition": Condition.CRITICAL,
            "description": "P2 develops peritonitis signs, becomes critical.",
        },
    ],
    "ground_truth": {
        "priority_order": ["P1", "P4", "P2", "P3"],
        "doctor_assignment": {"P1": "senior", "P4": "junior"},
        "ambulance_assignment": [],
        "transfer_decision": [],
        "critical_patients": ["P1", "P2"],
    },
    "max_steps": 5,
}


# ====================================================================
# HARD TASK — 6+ patients, resource conflict, dynamic changes,
#             incomplete/misleading info, escalation
# Problems: ALL 12
# ====================================================================

HARD_TASK: Dict[str, Any] = {
    "name": "hard",
    "difficulty": "hard",
    "description": (
        "Mass casualty incident: 7 patients arrive. Severe resource shortage. "
        "Some patients present misleading symptoms. Conditions change over time. "
        "Waiting patients may escalate. Transfer decisions required."
    ),
    "patients": [
        {  # Actual cardiac arrest — top priority
            "id": "P1",
            "symptoms": ["chest_pain", "collapse", "unresponsive"],
            "bp_systolic": 70,
            "bp_diastolic": 40,
            "heart_rate": 150,
            "condition": Condition.CRITICAL,
            "hidden_severity": Severity.CRITICAL,
            "misleading_info": False,
            "escalation_rate": 0.0,
        },
        {  # Looks critical but is actually panic attack (misleading)
            "id": "P2",
            "symptoms": ["chest_pain", "hyperventilation", "tingling"],
            "bp_systolic": 155,
            "bp_diastolic": 95,
            "heart_rate": 125,
            "condition": Condition.CRITICAL,
            "hidden_severity": Severity.LOW,
            "misleading_info": True,
            "escalation_rate": 0.0,
        },
        {  # Internal bleeding — unknown condition, high severity
            "id": "P3",
            "symptoms": ["abdominal_pain", "dizziness", "pallor"],
            "bp_systolic": 90,
            "bp_diastolic": 55,
            "heart_rate": 118,
            "condition": Condition.UNKNOWN,
            "hidden_severity": Severity.CRITICAL,
            "misleading_info": False,
            "escalation_rate": 0.15,
        },
        {  # Stroke symptoms — time-critical
            "id": "P4",
            "symptoms": ["facial_drooping", "arm_weakness", "slurred_speech"],
            "bp_systolic": 190,
            "bp_diastolic": 105,
            "heart_rate": 88,
            "condition": Condition.CRITICAL,
            "hidden_severity": Severity.CRITICAL,
            "misleading_info": False,
            "escalation_rate": 0.0,
        },
        {  # Fracture — painful but stable
            "id": "P5",
            "symptoms": ["leg_pain", "deformity", "swelling"],
            "bp_systolic": 130,
            "bp_diastolic": 82,
            "heart_rate": 90,
            "condition": Condition.STABLE,
            "hidden_severity": Severity.MODERATE,
            "misleading_info": False,
            "escalation_rate": 0.0,
        },
        {  # Asthma attack — could escalate
            "id": "P6",
            "symptoms": ["wheezing", "shortness_of_breath", "cyanosis"],
            "bp_systolic": 125,
            "bp_diastolic": 80,
            "heart_rate": 110,
            "condition": Condition.UNKNOWN,
            "hidden_severity": Severity.HIGH,
            "misleading_info": False,
            "escalation_rate": 0.2,
        },
        {  # Minor cut — looks worse than it is (misleading)
            "id": "P7",
            "symptoms": ["profuse_bleeding", "laceration", "anxiety"],
            "bp_systolic": 122,
            "bp_diastolic": 78,
            "heart_rate": 95,
            "condition": Condition.UNKNOWN,
            "hidden_severity": Severity.LOW,
            "misleading_info": True,
            "escalation_rate": 0.0,
        },
    ],
    "resources": {
        "senior_doctors": 1,
        "junior_doctors": 2,
        "nurses": 3,
        "icu_beds": 1,
        "general_beds": 2,
        "ambulances": 1,
    },
    "dynamic_events": [
        {
            "time_step": 1,
            "type": "condition_change",
            "patient_id": "P3",
            "new_condition": Condition.CRITICAL,
            "description": "P3 blood pressure drops further — internal bleeding confirmed.",
        },
        {
            "time_step": 2,
            "type": "condition_change",
            "patient_id": "P6",
            "new_condition": Condition.CRITICAL,
            "description": "P6 asthma attack worsens, oxygen saturation falling.",
        },
        {
            "time_step": 3,
            "type": "new_patient",
            "patient": {
                "id": "P8",
                "symptoms": ["burn", "smoke_inhalation", "confusion"],
                "bp_systolic": 95,
                "bp_diastolic": 60,
                "heart_rate": 135,
                "condition": Condition.CRITICAL,
                "hidden_severity": Severity.CRITICAL,
                "misleading_info": False,
                "escalation_rate": 0.0,
            },
            "description": "New patient P8 arrives from a building fire.",
        },
    ],
    "ground_truth": {
        "priority_order": ["P1", "P3", "P4", "P6", "P5", "P2", "P7"],
        "doctor_assignment": {"P1": "senior", "P3": "junior", "P4": "junior"},
        "ambulance_assignment": ["P1"],
        "transfer_decision": ["P5"],
        "critical_patients": ["P1", "P3", "P4", "P6"],
    },
    "max_steps": 6,
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}


def get_task(name: str) -> Dict[str, Any]:
    """Return a task config by name. Raises KeyError if not found."""
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]
