"""
Inference Script — Hospital ER Triage OpenEnv Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

Usage:
    # Offline (no API key needed)
    python inference.py --offline
    python inference.py --offline --task hard

    # With HF_TOKEN (mandatory for submission)
    set HF_TOKEN=hf_xxxxx
    python inference.py

    # Override model/endpoint via CLI
    python inference.py --api-base https://api.openai.com/v1 --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from typing import Dict, Any, List, Optional

from env import HospitalEnv, Grader


# ═══════════════════════════════════════════════════════════════════════════
# MANDATORY ENVIRONMENT VARIABLES (Hackathon Guidelines §3)
# ═══════════════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "hospital-er-triage"


# ═══════════════════════════════════════════════════════════════════════════
# STDOUT LOGGING HELPERS (Hackathon Guidelines §4)
# ═══════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line to stdout."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line to stdout."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# OFFLINE RULE-BASED AGENT
# ═══════════════════════════════════════════════════════════════════════════

# Symptom severity weights for the rule-based engine
CRITICAL_SYMPTOMS = {
    "chest_pain": 8, "collapse": 10, "unresponsive": 10,
    "cardiac_arrest": 10, "seizure": 9, "severe_bleeding": 8,
    "shortness_of_breath": 6, "facial_drooping": 9,
    "arm_weakness": 8, "slurred_speech": 8,
    "smoke_inhalation": 7, "burn": 6,
    "abdominal_pain": 4, "profuse_bleeding": 5,
    "cyanosis": 7, "confusion": 6,
}

MODERATE_SYMPTOMS = {
    "sweating": 3, "dizziness": 4, "pallor": 4,
    "nausea": 2, "vomiting": 3, "blurred_vision": 4,
    "severe_headache": 5, "wheezing": 4,
    "hyperventilation": 3, "tingling": 2,
    "bleeding": 3, "laceration": 2,
}

LOW_SYMPTOMS = {
    "fever": 1, "cough": 1, "fatigue": 1,
    "ankle_pain": 1, "swelling": 1, "leg_pain": 2,
    "deformity": 2, "anxiety": 1,
}


def _compute_acuity_score(patient: Dict[str, Any]) -> float:
    """Compute a numeric acuity score from observable patient data.

    Higher score = more critical. Considers:
      - Symptoms (weighted lookup)
      - Blood pressure deviation from normal (120/80)
      - Heart rate deviation from normal (60–100)
      - Condition tag
      - Waiting time (urgency increases)
    """
    score = 0.0

    # --- Symptom scoring ---
    for s in patient.get("symptoms", []):
        s_lower = s.lower()
        if s_lower in CRITICAL_SYMPTOMS:
            score += CRITICAL_SYMPTOMS[s_lower]
        elif s_lower in MODERATE_SYMPTOMS:
            score += MODERATE_SYMPTOMS[s_lower]
        elif s_lower in LOW_SYMPTOMS:
            score += LOW_SYMPTOMS[s_lower]
        else:
            score += 2  # unknown symptom gets moderate weight

    # --- Blood pressure scoring ---
    bp_str = patient.get("bp", "120/80")
    try:
        sys_bp, dia_bp = map(int, bp_str.split("/"))
    except (ValueError, AttributeError):
        sys_bp, dia_bp = 120, 80

    # Hypertensive crisis (>180/120) or hypotension (<90/60)
    if sys_bp >= 180 or dia_bp >= 120:
        score += 8
    elif sys_bp >= 160 or dia_bp >= 100:
        score += 5
    elif sys_bp <= 80 or dia_bp <= 45:
        score += 9  # severe hypotension — shock
    elif sys_bp <= 90 or dia_bp <= 55:
        score += 7  # hypotension

    # --- Heart rate scoring ---
    hr = patient.get("heart_rate", 75)
    if hr >= 140:
        score += 8
    elif hr >= 120:
        score += 6
    elif hr >= 100:
        score += 3
    elif hr <= 45:
        score += 8  # bradycardia
    elif hr <= 55:
        score += 4

    # --- Condition tag ---
    condition = patient.get("condition", "unknown")
    if condition == "critical":
        score += 6
    elif condition == "unknown":
        score += 3  # unknown is suspicious

    # --- Waiting time urgency ---
    waiting = patient.get("waiting_time", 0)
    score += waiting * 1.5

    return score


def rule_based_agent(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic rule-based triage agent.

    Strategy:
      1. Score each patient by acuity
      2. Sort by descending score → priority order
      3. Assign senior doctors to top patients, junior to next
      4. Ambulance for patients with very high acuity + transport symptoms
      5. Transfer stable patients when beds are scarce
    """
    patients = observation["patients"]
    resources = observation["resources"]

    # Score and sort
    scored = []
    for p in patients:
        acuity = _compute_acuity_score(p)
        scored.append((p["id"], acuity, p))
    scored.sort(key=lambda x: x[1], reverse=True)

    priority_order = [pid for pid, _, _ in scored]

    # Doctor assignment
    senior_avail = resources["available_doctors"]["senior"]
    junior_avail = resources["available_doctors"]["junior"]
    doctor_assignment: Dict[str, str] = {}

    for pid, acuity, p in scored:
        if senior_avail > 0 and acuity >= 15:
            doctor_assignment[pid] = "senior"
            senior_avail -= 1
        elif junior_avail > 0:
            doctor_assignment[pid] = "junior"
            junior_avail -= 1
        elif senior_avail > 0:
            doctor_assignment[pid] = "senior"
            senior_avail -= 1

    # Ambulance — assign to highest-acuity patients needing emergency intervention
    ambulance_avail = resources["ambulance_available"]
    ambulance_assignment: List[str] = []
    transport_symptoms = {"collapse", "unresponsive", "cardiac_arrest", "seizure",
                          "smoke_inhalation", "burn", "severe_bleeding"}
    for pid, acuity, p in scored:
        if ambulance_avail <= 0:
            break
        patient_symptoms = set(s.lower() for s in p.get("symptoms", []))
        if acuity >= 20 and patient_symptoms & transport_symptoms:
            ambulance_assignment.append(pid)
            ambulance_avail -= 1

    # Transfer — if ICU beds are very limited and we have stable patients
    icu_beds = resources["available_ICU_beds"]
    general_beds = resources["available_general_beds"]
    transfer_decision: List[str] = []

    # Count how many critical patients need ICU
    critical_count = sum(1 for _, acuity, _ in scored if acuity >= 15)
    if critical_count > icu_beds:
        # Transfer stable patients to free resources
        for pid, acuity, p in reversed(scored):
            if p.get("condition") == "stable" and acuity < 10:
                transfer_decision.append(pid)
                if len(transfer_decision) >= (critical_count - icu_beds):
                    break

    # Build explanation
    explanation_parts = ["Rule-based triage analysis:"]
    for pid, acuity, p in scored[:3]:
        explanation_parts.append(
            f"  {pid}: acuity={acuity:.1f}, condition={p['condition']}, "
            f"BP={p['bp']}, HR={p['heart_rate']}, symptoms={p['symptoms']}"
        )
    if len(scored) > 3:
        explanation_parts.append(f"  ... and {len(scored)-3} more patients.")

    top_critical = [pid for pid, acuity, _ in scored if acuity >= 15]
    if top_critical:
        explanation_parts.append(
            f"  Critical patients identified: {top_critical}. "
            f"Senior doctors assigned to highest severity cases. "
            f"Priority based on combined vital signs, symptoms, and condition assessment."
        )
    if transfer_decision:
        explanation_parts.append(
            f"  Resource conflict: {critical_count} critical patients but only "
            f"{icu_beds} ICU beds. Transferring stable patients: {transfer_decision}."
        )

    return {
        "priority_order": priority_order,
        "doctor_assignment": doctor_assignment,
        "ambulance_assignment": ambulance_assignment,
        "transfer_decision": transfer_decision,
        "explanation": "\n".join(explanation_parts),
    }


# ═══════════════════════════════════════════════════════════════════════════
# OPENAI LLM AGENT  (enhanced prompt engineering for 0.9+ scores)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert emergency triage AI operating in a high-pressure hospital ER.
Your goal is to MAXIMIZE patient survival under resource constraints.
Your performance is graded on a 0-1 scale across: priority ordering, doctor
assignment, critical patient handling, ambulance usage, transfers, and explanation.

# ════════════════════════════════════════════════════════════════
# 🔥 CORE PRIORITY RULES — STRICT MEDICAL SEVERITY HIERARCHY
# ════════════════════════════════════════════════════════════════
Rank patients using ONLY objective clinical indicators. ALWAYS prioritize
based on immediate life threat (airway, breathing, circulation), rapidly
deteriorating conditions, internal bleeding, oxygen drops, and shock.

  TIER 1 — IMMEDIATELY LIFE-THREATENING (highest priority):
    - Cardiac arrest, collapse, unresponsive, seizure
    - Systolic BP ≤ 80 OR diastolic BP ≤ 45 (severe shock / hypotension)
    - Active internal bleeding with hemodynamic instability
      (low BP + high HR + abdominal pain + pallor/dizziness = TIER 1)
    - Stroke signs: facial_drooping + arm_weakness + slurred_speech

  TIER 2 — EMERGENT:
    - Systolic BP ≤ 90 or ≥ 180 (hypertensive crisis / hypotension)
    - Heart rate ≥ 130 bpm
    - Severe burns + smoke inhalation
    - Respiratory failure: cyanosis + wheezing + shortness_of_breath
    - Condition worsening/escalating from previous step

  TIER 3 — URGENT:
    - Significant bleeding WITH low BP (lacerations + hemodynamic changes)
    - Abdominal pain with hemodynamic changes
    - Heart rate 100-129 with concerning symptoms

  TIER 4 — LESS URGENT / MISLEADING:
    - Stable vitals + minor injuries (sprains, minor cuts, fractures)
    - Fever + cough without respiratory distress
    - Panic attacks (see below)
    - Profuse bleeding with NORMAL vitals = looks scary but stable

# ════════════════════════════════════════════════════════════════
# ⚠️ CRITICAL: PANIC ATTACKS vs REAL EMERGENCIES
# ════════════════════════════════════════════════════════════════
This is the MOST IMPORTANT distinction. Getting this wrong tanks your score.

PANIC ATTACK (TIER 4 — LOW priority):
  - hyperventilation + tingling + chest_pain
  - BP > 100 systolic, patient is conscious
  - NO collapse, NO unresponsiveness
  - Even if labeled "critical" — IGNORE the label, use vitals

REAL CARDIAC EMERGENCY (TIER 1 — HIGHEST priority):
  - collapse + unresponsive + abnormal BP (< 90 or > 180)
  - chest_pain with hemodynamic instability

MISLEADING BLEEDING (TIER 4 — LOW priority):
  - profuse_bleeding + laceration + anxiety
  - BUT normal vitals (BP 120/78, HR 95) = superficial wound, NOT urgent

# ════════════════════════════════════════════════════════════════
# 👨‍⚕️ DOCTOR ASSIGNMENT RULES
# ════════════════════════════════════════════════════════════════
1. Senior doctors → TIER 1 patients (most critical, unstable vitals)
2. Junior doctors → TIER 2 and TIER 3 patients
3. NEVER assign a senior doctor to a low-severity patient if a critical
   patient exists
4. ASSIGN ALL AVAILABLE DOCTORS. If you have 1 senior + 2 junior doctors,
   assign exactly 3 doctors to the top 3 priority patients.
   Do NOT leave doctors idle when patients are waiting.
5. Assignment format: top priority patient gets senior, next patients
   get junior doctors, in priority order.

# ════════════════════════════════════════════════════════════════
# 🚑 AMBULANCE & TRANSFER RULES
# ════════════════════════════════════════════════════════════════
AMBULANCE assignment is MANDATORY for:
  - ANY patient who is collapsed or unresponsive
  - Cardiac arrest patients
  - Patients needing emergency transport to surgery/ICU
  → If a patient has "collapse" or "unresponsive" in symptoms, they MUST
    get an ambulance. This includes cardiac emergencies.

TRANSFER decisions:
  - When critical patients EXCEED available ICU beds → TRANSFER the most
    STABLE patients (TIER 3/4) to free resources
  - Transfer patients with stable vitals and non-life-threatening conditions
    (e.g., fractures, minor injuries)
  - Do NOT delay transfers — initiate early when resource warnings appear

# ════════════════════════════════════════════════════════════════
# 🏥 ICU & RESOURCE MANAGEMENT
# ════════════════════════════════════════════════════════════════
1. ICU beds → patients with highest survival probability AND urgent need
2. If multiple critical patients and limited ICU → transfer stable patients
3. React to resource warnings IMMEDIATELY
4. "unknown" condition is SUSPICIOUS — analyze vitals carefully:
   - Low BP + high HR + abdominal_pain + unknown = likely INTERNAL EMERGENCY
   - Treat as TIER 1 or TIER 2 based on vital severity

# ════════════════════════════════════════════════════════════════
# ⚡ DYNAMIC EVENT HANDLING — MOST CRITICAL RULE
# ════════════════════════════════════════════════════════════════
CONTINUOUSLY RE-EVALUATE at every time step. The situation is DYNAMIC.
Your decisions must be ADAPTIVE, not static.

1. If ANY patient's condition worsens (condition changes to "critical"):
   → IMMEDIATELY move them UP in priority
   → Re-rank ALL patients from scratch
   → Reassign doctors if needed

2. If a NEW patient arrives:
   → Evaluate their severity against ALL existing patients
   → Insert them at the correct priority position
   → Re-evaluate transfers and doctor assignments

3. NEVER keep the same priority order if the situation has changed.
   Every step is a fresh evaluation based on current patient states.

# ════════════════════════════════════════════════════════════════
# 🎯 OBJECTIVE — What you are graded on
# ════════════════════════════════════════════════════════════════
MAXIMIZE: Survival rate, correct prioritization, efficient resource usage,
          timely transfers
MINIMIZE: Delays in critical care, misallocation of doctors, ignoring
          dynamic changes

Scoring weights:
  Priority ordering:    30% (positional accuracy vs ground truth)
  Doctor assignment:    25% (correct type to correct patient)
  Critical handling:    20% (critical patients in top priority slots)
  Ambulance:            10% (correct ambulance assignments)
  Transfer:              5% (correct transfer decisions)
  Explanation:          10% (clinical reasoning quality)

# ════════════════════════════════════════════════════════════════
# REQUIRED REASONING PROCESS (follow in order)
# ════════════════════════════════════════════════════════════════
Step 1: Parse each patient's vitals (BP, HR) and symptoms
Step 2: Identify MISLEADING cases (panic attacks, superficial bleeding)
Step 3: Classify each patient into TIER 1/2/3/4
Step 4: Sort by tier (1 first), then by vital severity within tier
Step 5: Assign senior doctor to #1 priority, junior to #2 and #3
Step 6: Check for collapsed/unresponsive → assign ambulance
Step 7: Check if critical count > ICU beds → add stable patients to transfers
Step 8: Write explanation mentioning tiers, vitals, and key reasoning

# ════════════════════════════════════════════════════════════════
# OUTPUT FORMAT (respond with ONLY this JSON, nothing else)
# ════════════════════════════════════════════════════════════════
{
    "priority_order": ["most_critical_first", ...],
    "doctor_assignment": {"patient_id": "senior_or_junior", ...},
    "ambulance_assignment": ["patient_ids_needing_transport"],
    "transfer_decision": ["stable_patient_ids_to_transfer"],
    "explanation": "Step-by-step clinical reasoning..."
}
""")

# -- Few-shot examples for consistent output format --
FEW_SHOT_EXAMPLE = textwrap.dedent("""\
=== EMERGENCY ROOM STATUS (Time Step 0) ===

## PATIENTS IN QUEUE
- **P1**: symptoms=['chest_pain', 'collapse', 'unresponsive'], BP=70/40, HR=150 bpm, condition=critical, waiting=0 steps  ALERTS: *** SEVERE HYPOTENSION/SHOCK ***, *** SEVERE TACHYCARDIA ***, !!! UNCONSCIOUS/COLLAPSED !!!
- **P2**: symptoms=['chest_pain', 'hyperventilation', 'tingling'], BP=155/95, HR=125 bpm, condition=critical, waiting=0 steps  ALERTS: (possible panic attack presentation)
- **P3**: symptoms=['abdominal_pain', 'dizziness', 'pallor'], BP=90/55, HR=118 bpm, condition=unknown, waiting=0 steps  ALERTS: ** HYPOTENSION **, ** TACHYCARDIA **

## AVAILABLE RESOURCES
- Senior Doctors: 1
- Junior Doctors: 1
- ICU Beds: 1
- General Beds: 3
- Ambulances: 1

## CONSTRAINTS
- WARNING: Only 1 ICU bed available. Transfer stable patients if needed.
""")

FEW_SHOT_RESPONSE = json.dumps({
    "priority_order": ["P1", "P3", "P2"],
    "doctor_assignment": {"P1": "senior", "P3": "junior"},
    "ambulance_assignment": ["P1"],
    "transfer_decision": [],
    "explanation": (
        "TIER CLASSIFICATION: "
        "P1 = TIER 1 (cardiac arrest: collapse + unresponsive + BP 70/40 = severe shock, HR 150 = severe tachycardia). "
        "P3 = TIER 1/2 (unknown condition but BP 90/55 = hypotension + HR 118 = tachycardia + pallor + dizziness = likely internal bleeding, hemodynamically unstable). "
        "P2 = TIER 4 (despite 'critical' label: hyperventilation + tingling + chest_pain with BP 155/95 = classic panic attack presentation, NOT true cardiac emergency. BP is elevated but not in shock range, patient is conscious and responsive). "
        "DOCTORS: Senior to P1 (TIER 1, most critical), Junior to P3 (TIER 2, internal bleeding). P2 is stable and can wait. "
        "AMBULANCE: P1 MUST have ambulance — collapsed and unresponsive, needs emergency transport. "
        "TRANSFER: Not needed yet — 1 ICU bed for P1, P3 can use general bed initially. "
        "KEY: Condition labels can be misleading. P2 is labeled 'critical' but vitals and symptom pattern indicate panic attack. P3 is labeled 'unknown' but vitals indicate serious internal emergency."
    ),
}, indent=2)


def build_user_prompt(observation: Dict[str, Any]) -> str:
    """Convert environment observation into a detailed clinical prompt.

    Pre-computes vital sign flags and resource warnings to help the LLM
    make better decisions.
    """
    parts = [f"=== EMERGENCY ROOM STATUS (Time Step {observation['current_time_step']}) ===\n"]

    # -- Patient details with vital sign analysis --
    parts.append("## PATIENTS IN QUEUE\n")
    for p in observation["patients"]:
        # Parse BP
        try:
            sys_bp, dia_bp = map(int, p["bp"].split("/"))
        except (ValueError, AttributeError):
            sys_bp, dia_bp = 120, 80

        # Compute vital sign flags
        flags = []
        if sys_bp <= 80 or dia_bp <= 45:
            flags.append("*** SEVERE HYPOTENSION/SHOCK ***")
        elif sys_bp <= 90 or dia_bp <= 55:
            flags.append("** HYPOTENSION **")
        elif sys_bp >= 180 or dia_bp >= 120:
            flags.append("** HYPERTENSIVE CRISIS **")

        hr = p["heart_rate"]
        if hr >= 140:
            flags.append("*** SEVERE TACHYCARDIA ***")
        elif hr >= 120:
            flags.append("** TACHYCARDIA **")
        elif hr <= 50:
            flags.append("** BRADYCARDIA **")

        # Check for misleading patterns
        symptom_set = set(s.lower() for s in p.get("symptoms", []))
        if "hyperventilation" in symptom_set and "tingling" in symptom_set:
            if sys_bp > 100:
                flags.append("(possible panic attack presentation)")
        if "collapse" in symptom_set or "unresponsive" in symptom_set:
            flags.append("!!! UNCONSCIOUS/COLLAPSED !!!")

        flag_str = f"  ALERTS: {', '.join(flags)}" if flags else ""

        parts.append(
            f"- **{p['id']}**: symptoms={p['symptoms']}, "
            f"BP={p['bp']}, HR={hr} bpm, "
            f"condition={p['condition']}, waiting={p['waiting_time']} steps"
            f"{flag_str}"
        )

    # -- Resources with constraint warnings --
    res = observation["resources"]
    parts.append("\n## AVAILABLE RESOURCES")
    parts.append(f"- Senior Doctors: {res['available_doctors']['senior']}")
    parts.append(f"- Junior Doctors: {res['available_doctors']['junior']}")
    parts.append(f"- Nurses: {res['available_nurses']}")
    parts.append(f"- ICU Beds: {res['available_ICU_beds']}")
    parts.append(f"- General Beds: {res['available_general_beds']}")
    parts.append(f"- Ambulances: {res['ambulance_available']}")

    # Resource constraint warnings
    total_doctors = res['available_doctors']['senior'] + res['available_doctors']['junior']
    num_patients = len(observation["patients"])
    constraints = []

    if res['available_ICU_beds'] <= 1:
        constraints.append(
            f"WARNING: Only {res['available_ICU_beds']} ICU bed(s) available. "
            "You MUST include transfer_decision for stable patients if multiple "
            "critical patients need ICU care."
        )
    if total_doctors < num_patients:
        constraints.append(
            f"WARNING: Only {total_doctors} doctors for {num_patients} patients. "
            "Assign to highest-priority patients first."
        )
    if res['ambulance_available'] <= 1:
        constraints.append(
            f"WARNING: Only {res['ambulance_available']} ambulance(s). "
            "Reserve for collapsed/unresponsive patients."
        )

    if constraints:
        parts.append("\n## CONSTRAINTS")
        for c in constraints:
            parts.append(f"- {c}")

    parts.append(f"\n## QUEUE ORDER: {observation['queue_state']}")
    parts.append(
        "\nAnalyze each patient using the STRICT MEDICAL SEVERITY HIERARCHY. "
        "Classify into TIER 1/2/3/4. Sort by tier. "
        f"Assign doctors (max {res['available_doctors']['senior']} senior + "
        f"{res['available_doctors']['junior']} junior). "
        "Include transfer_decision if critical patients exceed ICU beds. "
        "Respond with ONLY the JSON object."
    )

    return "\n".join(parts)


def query_llm(client, observation: Dict[str, Any], model: str = "gpt-4o") -> Dict[str, Any]:
    """Send observation to OpenAI-compatible API and parse structured action.

    Uses few-shot prompting and falls back to rule-based agent on failure.
    """
    user_prompt = build_user_prompt(observation)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Few-shot example
        {"role": "user", "content": FEW_SHOT_EXAMPLE},
        {"role": "assistant", "content": FEW_SHOT_RESPONSE},
        # Actual observation
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DEBUG] LLM API call failed: {e}", file=sys.stderr, flush=True)
        print("[DEBUG] Using rule-based agent for this step.", file=sys.stderr, flush=True)
        return rule_based_agent(observation)

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL)
    if json_match:
        raw_text = json_match.group(1).strip()

    action_dict = None

    # Attempt 1: direct parse
    try:
        action_dict = json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first { ... } block
    if action_dict is None:
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                action_dict = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

    # Attempt 3: fallback to rule-based agent
    if action_dict is None or not isinstance(action_dict, dict):
        print(f"[DEBUG] Could not parse LLM response. Falling back to rule-based agent.", file=sys.stderr, flush=True)
        return rule_based_agent(observation)

    # Validate and fill defaults
    action_dict.setdefault("priority_order", [])
    action_dict.setdefault("doctor_assignment", {})
    action_dict.setdefault("ambulance_assignment", [])
    # Handle both key names (transfer_decision vs transfer)
    if "transfer" in action_dict and "transfer_decision" not in action_dict:
        action_dict["transfer_decision"] = action_dict.pop("transfer")
    action_dict.setdefault("transfer_decision", [])
    action_dict.setdefault("explanation", "")

    # Validate priority_order contains actual patient IDs
    patient_ids = set(p["id"] for p in observation["patients"])
    valid_priority = [pid for pid in action_dict["priority_order"] if pid in patient_ids]
    # Add any missing patients at the end
    missing = [pid for pid in observation["queue_state"] if pid not in valid_priority]
    action_dict["priority_order"] = valid_priority + missing

    return action_dict


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP  (with mandatory [START]/[STEP]/[END] stdout format)
# ═══════════════════════════════════════════════════════════════════════════

def run_task(
    env: HospitalEnv,
    task_name: str,
    *,
    offline: bool = False,
    client=None,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Run a single task and return results.

    Emits [START]/[STEP]/[END] lines to stdout.
    All verbose debugging output goes to stderr.
    """
    mode_label = "OFFLINE (rule-based)" if offline else f"ONLINE ({model})"
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  TASK: {task_name.upper()}  [{mode_label}]", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    obs = env.reset(task_name=task_name)

    # ── [START] line (mandatory stdout) ──
    log_start(task=task_name, env=BENCHMARK, model=model)

    total_reward = 0.0
    final_grade = None
    step_count = 0
    step_rewards: List[float] = []
    success = False

    try:
        while True:
            print(f"\n--- Step {obs['current_time_step']} ---", file=sys.stderr)
            print(f"  Patients: {len(obs['patients'])}", file=sys.stderr)
            print(f"  Queue: {obs['queue_state']}", file=sys.stderr)

            # Get AI decision
            if offline:
                action_dict = rule_based_agent(obs)
            else:
                action_dict = query_llm(client, obs, model=model)

            print(f"  AI Priority: {action_dict['priority_order']}", file=sys.stderr)
            print(f"  AI Doctors:  {action_dict['doctor_assignment']}", file=sys.stderr)
            print(f"  AI Ambulance: {action_dict['ambulance_assignment']}", file=sys.stderr)
            if action_dict["transfer_decision"]:
                print(f"  AI Transfer: {action_dict['transfer_decision']}", file=sys.stderr)

            # Step environment
            obs, reward, done, info = env.step(action_dict)
            total_reward += reward
            step_count += 1
            step_rewards.append(reward)

            # ── [STEP] line (mandatory stdout) ──
            action_str = json.dumps(action_dict, separators=(',', ':'))
            error_str = info.get("last_action_error", None)
            log_step(
                step=step_count,
                action=action_str,
                reward=reward,
                done=done,
                error=error_str,
            )

            print(f"  Reward: {reward:+.4f}  (cumulative: {info['cumulative_reward']:+.4f})", file=sys.stderr)

            if info.get("events_fired"):
                for e in info["events_fired"]:
                    print(f"  [EVENT] {e}", file=sys.stderr)
            if info.get("escalations"):
                for e in info["escalations"]:
                    print(f"  [ESCALATION] {e}", file=sys.stderr)
            if info.get("resource_notes"):
                for n in info["resource_notes"]:
                    print(f"  [RESOURCE] {n}", file=sys.stderr)

            if done:
                final_grade = info.get("final_grade")
                break

        # Compute score in [0, 1]
        score = final_grade["total_score"] if final_grade else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as e:
        print(f"[ERROR] Task {task_name} failed: {e}", file=sys.stderr, flush=True)
        score = 0.0
        success = False

    finally:
        # ── [END] line (mandatory stdout — ALWAYS emitted) ──
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=step_count, score=score, rewards=step_rewards)

    # Verbose summary to stderr
    print(f"\n{'-'*40}", file=sys.stderr)
    print(f"  Task Complete: {task_name}", file=sys.stderr)
    print(f"  Steps: {step_count}", file=sys.stderr)
    print(f"  Cumulative Reward: {total_reward:+.4f}", file=sys.stderr)
    print(f"  Score: {score:.4f}", file=sys.stderr)

    if final_grade:
        print(f"  Final Grade: {final_grade['total_score']:.4f} / 1.0", file=sys.stderr)
        print(f"    Priority:    {final_grade['priority_score']:.4f}", file=sys.stderr)
        print(f"    Doctor:      {final_grade['doctor_score']:.4f}", file=sys.stderr)
        print(f"    Critical:    {final_grade['critical_handling_score']:.4f}", file=sys.stderr)
        print(f"    Ambulance:   {final_grade['ambulance_score']:.4f}", file=sys.stderr)
        print(f"    Transfer:    {final_grade['transfer_score']:.4f}", file=sys.stderr)
        print(f"    Explanation: {final_grade['explanation_score']:.4f}", file=sys.stderr)

    return {
        "task": task_name,
        "steps": step_count,
        "cumulative_reward": round(total_reward, 4),
        "final_grade": final_grade,
        "score": score,
    }


# -----------------------------------------------------------------------
# Provider presets  (base_url, default_model)
# -----------------------------------------------------------------------
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "openai":    {"base_url": "https://api.openai.com/v1",                        "model": "gpt-4o"},
    "github":    {"base_url": "https://models.inference.ai.azure.com",             "model": "gpt-4o"},
    "gemini":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.0-flash"},
    "groq":      {"base_url": "https://api.groq.com/openai/v1",                   "model": "llama-3.3-70b-versatile"},
    "together":  {"base_url": "https://api.together.xyz/v1",                      "model": "meta-llama/Llama-3-70b-chat-hf"},
    "deepseek":  {"base_url": "https://api.deepseek.com/v1",                      "model": "deepseek-chat"},
    "nvidia":    {"base_url": "https://integrate.api.nvidia.com/v1",               "model": "meta/llama-3.3-70b-instruct"},
    "ollama":    {"base_url": "http://localhost:11434/v1",                         "model": "llama3"},
    "lmstudio":  {"base_url": "http://localhost:1234/v1",                          "model": "local-model"},
}


def main():
    parser = argparse.ArgumentParser(
        description="Hospital ER Triage - OpenEnv Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Mandatory env vars (hackathon guidelines):
          API_BASE_URL   LLM endpoint   (default: https://router.huggingface.co/v1)
          MODEL_NAME     Model ID       (default: Qwen/Qwen2.5-72B-Instruct)
          HF_TOKEN       API key        (required)

        Provider shortcuts (--provider):
          openai, github, gemini, groq, together, deepseek, nvidia, ollama, lmstudio
        """),
    )
    parser.add_argument("--task", type=str, default=None,
                        help="Run a specific task (easy/medium/hard). Default: all tasks.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--offline", action="store_true",
                        help="Use rule-based agent (no API key needed).")

    # LLM provider options (overrides for env vars)
    llm_group = parser.add_argument_group("LLM provider options (override env vars)")
    llm_group.add_argument("--provider", type=str, default=None,
                           choices=list(PROVIDER_PRESETS.keys()),
                           help="Provider shortcut (sets base URL & default model).")
    llm_group.add_argument("--api-key", type=str, default=None,
                           help="API key. Overrides HF_TOKEN env var.")
    llm_group.add_argument("--api-base", type=str, default=None,
                           help="Base URL. Overrides API_BASE_URL env var.")
    llm_group.add_argument("--model", type=str, default=None,
                           help="Model name. Overrides MODEL_NAME env var.")
    args = parser.parse_args()

    # ── Resolve credentials ──────────────────────────────────────────
    # Priority: CLI args > mandatory env vars > provider presets
    client = None
    active_model = MODEL_NAME  # default from env var

    if not args.offline:
        preset = PROVIDER_PRESETS.get(args.provider or "", {})

        # API key: --api-key > HF_TOKEN env var
        api_key = args.api_key or HF_TOKEN

        # Base URL: --api-base > API_BASE_URL env var > provider preset
        api_base = args.api_base or API_BASE_URL or preset.get("base_url")

        # Model: --model > MODEL_NAME env var > provider preset
        active_model = args.model or MODEL_NAME or preset.get("model", "Qwen/Qwen2.5-72B-Instruct")

        # For local providers (ollama, lmstudio), no API key is needed
        is_local = (args.provider in ("ollama", "lmstudio")) or (
            api_base and ("localhost" in api_base or "127.0.0.1" in api_base)
        )

        # HF_TOKEN is mandatory (hackathon requirement) unless offline or local
        if not api_key and not is_local:
            raise ValueError(
                "HF_TOKEN environment variable is required.\n"
                "Set it via:  export HF_TOKEN=hf_your_token\n"
                "Or run in offline mode:  python inference.py --offline"
            )

        from openai import OpenAI
        client = OpenAI(
            base_url=api_base,
            api_key=api_key if api_key else "not-needed",
        )

        print(f"Using provider: {args.provider or 'env-config'}", file=sys.stderr)
        print(f"  Base URL: {api_base}", file=sys.stderr)
        print(f"  Model:    {active_model}\n", file=sys.stderr)

    # ── Run tasks ─────────────────────────────────────────────────────
    env = HospitalEnv(seed=args.seed)
    tasks_to_run = [args.task] if args.task else env.get_task_names()
    results = []

    for task_name in tasks_to_run:
        result = run_task(
            env, task_name,
            offline=args.offline,
            client=client,
            model=active_model,
        )
        results.append(result)

    # ── Final summary (stderr) ────────────────────────────────────────
    print(f"\n{'='*60}", file=sys.stderr)
    print("  FINAL RESULTS SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    total_score = 0.0
    for r in results:
        score = r.get("score", 0.0)
        total_score += score
        print(f"  {r['task']:10s}  Score: {score:.4f}  Reward: {r['cumulative_reward']:+.4f}", file=sys.stderr)

    avg_score = total_score / len(results) if results else 0.0
    print(f"\n  Average Score: {avg_score:.4f} / 1.0", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Write results to file for reproducibility
    output_path = "results.json"
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_path}", file=sys.stderr)
    except PermissionError:
        # Docker containers may have read-only app directories
        fallback = os.path.join("/tmp", "results.json")
        try:
            with open(fallback, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to {fallback}", file=sys.stderr)
        except Exception:
            print("\nCould not save results.json (read-only filesystem)", file=sys.stderr)


if __name__ == "__main__":
    main()
