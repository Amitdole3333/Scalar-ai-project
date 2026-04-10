"""
HospitalEnv — OpenEnv-compatible RL-style environment for
Hospital Emergency Room Triage and Resource Management.

Implements all 12 required real-world challenges:
  Level 1 (Critical):
    1. Patient triage (severity detection)
    2. Priority queue management
    3. Delay penalty system
    4. Resource limitation (beds, doctors)
    5. Mass casualty scenario
    6. Ambulance allocation
    7. Doctor/nurse assignment

  Level 2 (High Complexity):
    8. Dynamic condition change
    9. Resource conflict
   10. Multi-parameter decision making

  Level 3 (Advanced):
   11. Incomplete/misleading patient information
   12. Patient escalation (waiting increases severity)
"""

from __future__ import annotations

import copy
import random
from typing import Dict, Any, List, Optional, Tuple

from .models import (
    Patient, Resources, Observation, Action,
    Condition, Severity, DoctorType,
)
from .tasks import get_task, TASKS
from .grader import Grader, GradeResult


class HospitalEnv:
    """OpenEnv-compatible Hospital ER Triage Environment."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)

        # State
        self._patients: List[Patient] = []
        self._resources: Resources = Resources()
        self._time_step: int = 0
        self._done: bool = False
        self._task: Dict[str, Any] = {}
        self._max_steps: int = 5
        self._cumulative_reward: float = 0.0
        self._history: List[Dict[str, Any]] = []
        self._grader = Grader()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "easy") -> Dict[str, Any]:
        """Initialize a new hospital scenario based on a task.

        Returns the first observation dict.
        """
        self._rng = random.Random(self._seed)
        task = get_task(task_name)
        self._task = copy.deepcopy(task)
        self._time_step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._history = []

        # Build patients
        self._patients = []
        for pdata in task["patients"]:
            self._patients.append(Patient(**pdata))

        # Build resources
        self._resources = Resources(**task["resources"])
        self._max_steps = task.get("max_steps", 5)

        return self._build_observation().to_dict()

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Process one agent decision.

        Args:
            action_dict: dict matching Action schema.

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            obs = self._build_observation().to_dict()
            return obs, 0.0, True, {"message": "Episode already finished."}

        action = Action.from_dict(action_dict)
        self._time_step += 1

        # ---- Compute reward --------------------------------------------------
        reward, reward_breakdown = self._compute_reward(action)
        self._cumulative_reward += reward

        # ---- Apply dynamic events for this time step -------------------------
        events_fired = self._apply_dynamic_events()

        # ---- Patient escalation (Problem 12) ---------------------------------
        escalations = self._apply_escalation()

        # ---- Increase waiting time (Problem 3) -------------------------------
        for p in self._patients:
            if p.id not in action.doctor_assignment:
                p.waiting_time += 1

        # ---- Resource consumption simulation ---------------------------------
        resource_notes = self._consume_resources(action)

        # ---- Check done ------------------------------------------------------
        if self._time_step >= self._max_steps:
            self._done = True

        # ---- Build info dict -------------------------------------------------
        info: Dict[str, Any] = {
            "time_step": self._time_step,
            "reward_breakdown": reward_breakdown,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "events_fired": events_fired,
            "escalations": escalations,
            "resource_notes": resource_notes,
        }

        if self._done:
            grade = self._grader.grade(
                action,
                self._task["ground_truth"],
                [p.to_dict() for p in self._patients],
            )
            info["final_grade"] = grade.to_dict()

        self._history.append({
            "step": self._time_step,
            "action": action.to_dict(),
            "reward": reward,
            "info": info,
        })

        obs = self._build_observation().to_dict()
        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return full current state (for debugging / grading)."""
        return {
            "time_step": self._time_step,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "patients": [p.to_dict() for p in self._patients],
            "resources": self._resources.to_dict(),
            "queue": [p.id for p in self._patients],
            "history": self._history,
            "task_name": self._task.get("name", ""),
        }

    def close(self):
        """Clean up environment resources.

        Required by OpenEnv interface. No-op for this environment since
        all state is in-memory.
        """
        pass

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        return Observation(
            patients=[p.observable_dict() for p in self._patients],
            resources=self._resources.to_dict(),
            current_time_step=self._time_step,
            queue_state=[p.id for p in self._patients],
        )

    # ------------------------------------------------------------------
    # Reward (Problem 3 – delay, and general reward shaping)
    # ------------------------------------------------------------------

    def _compute_reward(self, action: Action) -> Tuple[float, Dict[str, float]]:
        """Return (total_reward, breakdown_dict).

        Reward components:
          +1.0  correct priority for a critical patient
          +0.5  partial correct priority
           0.0  wrong priority
          -0.5  delay penalty per critical patient not prioritized
          -1.0  incorrect handling of critical patient
          +0.2  good explanation
          -0.3  resource misuse
        """
        breakdown: Dict[str, float] = {}
        total = 0.0

        gt = self._task["ground_truth"]
        gt_priority = gt.get("priority_order", [])
        gt_critical = set(gt.get("critical_patients", []))

        # --- Priority correctness ---
        priority_reward = 0.0
        for i, pid in enumerate(action.priority_order):
            if i < len(gt_priority) and gt_priority[i] == pid:
                priority_reward += 1.0
            elif pid in gt_priority:
                priority_reward += 0.5
        if gt_priority:
            priority_reward /= len(gt_priority)
        breakdown["priority"] = round(priority_reward, 4)
        total += priority_reward

        # --- Critical handling ---
        critical_reward = 0.0
        if gt_critical:
            top_slots = action.priority_order[: len(gt_critical)]
            correct_critical = len(set(top_slots) & gt_critical)
            missed = len(gt_critical) - correct_critical
            critical_reward = correct_critical * 1.0 - missed * 1.0
            critical_reward /= len(gt_critical)  # normalise
        breakdown["critical_handling"] = round(critical_reward, 4)
        total += critical_reward

        # --- Delay penalty ---
        delay_penalty = 0.0
        for p in self._patients:
            if p.hidden_severity >= Severity.HIGH and p.id not in action.doctor_assignment:
                delay_penalty -= 0.5
        breakdown["delay_penalty"] = round(delay_penalty, 4)
        total += delay_penalty

        # --- Resource misuse ---
        resource_penalty = 0.0
        # Assigning senior doctor to low-severity patient
        for pid, dtype in action.doctor_assignment.items():
            patient = self._get_patient(pid)
            if patient and dtype.lower() == "senior" and patient.hidden_severity <= Severity.LOW:
                resource_penalty -= 0.3
        # Requesting more ambulances than available
        if len(action.ambulance_assignment) > self._resources.ambulances:
            resource_penalty -= 0.3
        breakdown["resource_misuse"] = round(resource_penalty, 4)
        total += resource_penalty

        # --- Explanation bonus ---
        explanation_bonus = 0.0
        if action.explanation and len(action.explanation.strip()) > 30:
            explanation_bonus = 0.2
        breakdown["explanation_bonus"] = round(explanation_bonus, 4)
        total += explanation_bonus

        return round(total, 4), breakdown

    # ------------------------------------------------------------------
    # Dynamic events (Problem 8)
    # ------------------------------------------------------------------

    def _apply_dynamic_events(self) -> List[str]:
        events_fired: List[str] = []
        for event in self._task.get("dynamic_events", []):
            if event["time_step"] == self._time_step:
                if event["type"] == "condition_change":
                    patient = self._get_patient(event["patient_id"])
                    if patient:
                        patient.condition = event["new_condition"]
                        events_fired.append(event["description"])
                elif event["type"] == "new_patient":
                    new_p = Patient(**event["patient"])
                    self._patients.append(new_p)
                    events_fired.append(event["description"])
        return events_fired

    # ------------------------------------------------------------------
    # Escalation (Problem 12)
    # ------------------------------------------------------------------

    def _apply_escalation(self) -> List[str]:
        escalations: List[str] = []
        for p in self._patients:
            if p.escalation_rate > 0 and not p._escalated:
                if self._rng.random() < p.escalation_rate * p.waiting_time:
                    # Escalate severity
                    if p.hidden_severity < Severity.CRITICAL:
                        old = p.hidden_severity
                        p.hidden_severity = Severity(min(p.hidden_severity + 1, Severity.CRITICAL))
                        p.condition = Condition.CRITICAL
                        p._escalated = True
                        escalations.append(
                            f"{p.id} escalated from {old.name} to {p.hidden_severity.name} "
                            f"after waiting {p.waiting_time} steps."
                        )
        return escalations

    # ------------------------------------------------------------------
    # Resource consumption (Problems 4, 6, 7, 9)
    # ------------------------------------------------------------------

    def _consume_resources(self, action: Action) -> List[str]:
        notes: List[str] = []

        # Doctor assignment
        senior_used = sum(1 for d in action.doctor_assignment.values() if d.lower() == "senior")
        junior_used = sum(1 for d in action.doctor_assignment.values() if d.lower() == "junior")

        if senior_used > self._resources.senior_doctors:
            notes.append(
                f"WARNING: Requested {senior_used} senior doctors but only "
                f"{self._resources.senior_doctors} available. Excess ignored."
            )
            senior_used = self._resources.senior_doctors
        if junior_used > self._resources.junior_doctors:
            notes.append(
                f"WARNING: Requested {junior_used} junior doctors but only "
                f"{self._resources.junior_doctors} available. Excess ignored."
            )
            junior_used = self._resources.junior_doctors

        # We don't permanently decrement — doctors free up next step
        # (simplification for environment loop)

        # Ambulance
        amb_used = len(action.ambulance_assignment)
        if amb_used > self._resources.ambulances:
            notes.append(
                f"WARNING: Requested {amb_used} ambulances but only "
                f"{self._resources.ambulances} available."
            )

        # Beds – check ICU capacity for critical patients assigned
        critical_needing_bed = sum(
            1 for pid in action.doctor_assignment
            if self._get_patient(pid) and self._get_patient(pid).hidden_severity >= Severity.CRITICAL
        )
        if critical_needing_bed > self._resources.icu_beds:
            notes.append(
                f"WARNING: {critical_needing_bed} critical patients need ICU beds, "
                f"only {self._resources.icu_beds} available. Consider transfers."
            )

        return notes

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_patient(self, pid: str) -> Optional[Patient]:
        for p in self._patients:
            if p.id == pid:
                return p
        return None

    def get_task_names(self) -> List[str]:
        return list(TASKS.keys())
