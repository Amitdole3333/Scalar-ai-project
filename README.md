---
title: Hospital ER Triage
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🏥 Hospital Emergency Room Triage & Resource Management

> An OpenEnv-compatible AI training environment that simulates real-world hospital emergency decision-making under pressure.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [12 Real-World Challenges](#-12-real-world-challenges)
- [Observation & Action Format](#-observation--action-format)
- [Reward System](#-reward-system)
- [Grading Rubric](#-grading-rubric)
- [Tasks](#-tasks)
- [Quick Start](#-quick-start)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)

---

## 🎯 Problem Statement

In a hospital emergency room, every second counts. Patients arrive with varying severity, resources are finite, and conditions change dynamically. This environment challenges an AI agent to:

- **Triage patients** by analysing symptoms, vitals, and conditions
- **Prioritize queues** so the most critical patients are treated first
- **Allocate resources** (senior/junior doctors, ICU beds, ambulances)
- **Handle uncertainty** — some patients present misleading symptoms
- **Adapt to change** — conditions evolve, new patients arrive mid-scenario
- **Explain decisions** — every triage call must be justified

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    inference.py                      │
│    ┌───────────┐    ┌───────────┐    ┌───────────┐  │
│    │  Observe  │───▶│  OpenAI   │───▶│   Parse   │  │
│    │    Env    │    │    LLM    │    │  Action   │  │
│    └───────────┘    └───────────┘    └───────────┘  │
│         ▲                                  │         │
│         │           ┌───────────┐          │         │
│         └───────────│  Env.step │◀─────────┘         │
│                     └───────────┘                    │
│                          │                           │
│                     ┌────▼────┐                      │
│                     │ Grader  │                      │
│                     └─────────┘                      │
└─────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **HospitalEnv** | `env/environment.py` | RL-style env with `reset()`, `step()`, `state()` |
| **Models** | `env/models.py` | Pydantic models: `Patient`, `Resources`, `Action`, `Observation` |
| **Tasks** | `env/tasks.py` | Pre-defined scenarios (Easy / Medium / Hard) |
| **Grader** | `env/grader.py` | Deterministic scoring [0.0–1.0] |
| **Inference** | `inference.py` | OpenAI API loop with prompt engineering |

---

## 🔥 12 Real-World Challenges

### Level 1 — Critical Fundamentals

| # | Challenge | Description |
|---|-----------|-------------|
| 1 | **Patient Triage** | Detect severity from symptoms + vitals |
| 2 | **Priority Queue** | Order patients by acuity, not arrival time |
| 3 | **Delay Penalty** | Waiting critical patients incur negative reward |
| 4 | **Resource Limitation** | Finite beds, doctors, and nurses |
| 5 | **Mass Casualty** | 7+ simultaneous patients in the hard task |
| 6 | **Ambulance Allocation** | Assign limited ambulances to those needing transport |
| 7 | **Doctor Assignment** | Match senior doctors to critical cases, juniors to stable |

### Level 2 — High Complexity

| # | Challenge | Description |
|---|-----------|-------------|
| 8 | **Dynamic Condition Change** | Patient status evolves over time steps |
| 9 | **Resource Conflict** | Multiple critical patients, insufficient resources |
| 10 | **Multi-Parameter Reasoning** | Decisions require weighing BP + HR + symptoms together |

### Level 3 — Advanced / Winning Edge

| # | Challenge | Description |
|---|-----------|-------------|
| 11 | **Misleading Information** | Some patients present symptoms that don't match true severity |
| 12 | **Patient Escalation** | Waiting too long increases a patient's hidden severity |

---

## 📊 Observation & Action Format

### Observation (returned by `env.reset()` and `env.step()`)

```json
{
  "patients": [
    {
      "id": "P1",
      "symptoms": ["chest_pain", "shortness_of_breath"],
      "bp": "180/110",
      "heart_rate": 130,
      "condition": "critical",
      "waiting_time": 0
    }
  ],
  "resources": {
    "available_doctors": {"senior": 2, "junior": 3},
    "available_nurses": 5,
    "available_ICU_beds": 3,
    "available_general_beds": 10,
    "ambulance_available": 2
  },
  "current_time_step": 0,
  "queue_state": ["P1", "P2"]
}
```

### Action (expected from agent)

```json
{
  "priority_order": ["P1", "P2"],
  "doctor_assignment": {"P1": "senior", "P2": "junior"},
  "ambulance_assignment": ["P1"],
  "transfer_decision": [],
  "explanation": "P1 shows signs of acute MI with dangerously elevated BP..."
}
```

---

## 🧪 Reward System

| Event | Reward |
|-------|--------|
| Correct priority for critical patient | **+1.0** |
| Partial correct priority | **+0.5** |
| Wrong priority | **0.0** |
| Delay penalty (critical left untreated) | **-0.5** |
| Incorrect critical handling | **-1.0** |
| Good explanation (>30 chars) | **+0.2** |
| Resource misuse (senior on minor case) | **-0.3** |

Rewards are **continuous** — partial progress is always recognised.

---

## 🧠 Grading Rubric

The `Grader` produces a deterministic score in **[0.0, 1.0]** across 6 weighted categories:

| Category | Weight | Metric |
|----------|--------|--------|
| Priority Ordering | 30% | Normalised rank distance vs ground truth |
| Doctor Assignment | 25% | Exact match per patient (half credit for wrong type) |
| Critical Handling | 20% | Critical patients placed in top priority slots |
| Ambulance Allocation | 10% | Set overlap (Jaccard) |
| Transfer Decisions | 5% | Set overlap (Jaccard) |
| Explanation Quality | 10% | Heuristic: mentions critical IDs, reasoning terms, length |

---

## 📝 Tasks

### Easy (2 patients)
- Clear cardiac emergency vs. minor sprain
- Plenty of resources — no conflict
- Tests: triage, queue management, ambulance allocation

### Medium (4 patients)
- Hypertensive crisis, unknown abdominal pain, fever, laceration
- Only 1 senior + 1 junior doctor
- Dynamic event: patient condition worsens at step 2
- Tests: multi-parameter reasoning, resource limitation

### Hard (7+ patients)
- Mass casualty: cardiac arrest, panic attack (misleading), internal bleeding, stroke, fracture, asthma, superficial laceration (misleading)
- Severe resource shortage (1 senior doctor, 1 ICU bed)
- Dynamic events: condition changes, new patient arrives
- Patient escalation: asthma and bleeding patients worsen if untreated
- Tests: **all 12 challenges**

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd hospital-er-triage

# Install dependencies
pip install -r requirements.txt
```

### Set Environment Variables (Mandatory)

```bash
# Linux/macOS
export HF_TOKEN=your-token-here                          # Required
export API_BASE_URL=https://router.huggingface.co/v1      # Optional (has default)
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct              # Optional (has default)

# Windows (PowerShell)
$env:HF_TOKEN = "your-token-here"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
```

> **Note:** `HF_TOKEN` is **mandatory** for online mode. The script will raise `ValueError` if not set. Use `--offline` for rule-based mode without an API key.

### Run Inference

```bash
# Run all tasks (easy, medium, hard)
python inference.py

# Run a specific task
python inference.py --task easy

# Offline mode (no API key needed)
python inference.py --offline

# Override model/endpoint via CLI
python inference.py --api-base https://api.openai.com/v1 --model gpt-4o

# Set random seed
python inference.py --seed 123
```

### Expected Output (stdout)

The script emits structured `[START]`/`[STEP]`/`[END]` lines to stdout:

```
[START] task=easy env=hospital-er-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"priority_order":["P1","P2"],...} reward=2.20 done=false error=null
[STEP] step=2 action={...} reward=2.20 done=false error=null
[STEP] step=3 action={...} reward=2.20 done=true error=null
[END] success=true steps=3 score=0.900 rewards=2.20,2.20,2.20
```

Verbose debug output goes to `stderr`.

### Baseline Scores (rule-based offline agent)

| Task | Score | Steps |
|------|-------|-------|
| Easy | 0.900 | 3 |
| Medium | 0.885 | 5 |
| Hard | 0.767 | 6 |
| **Average** | **0.851** | — |

### Use as a Library

```python
from env import HospitalEnv, Grader, Action

env = HospitalEnv(seed=42)
obs = env.reset(task_name="easy")

action = {
    "priority_order": ["P1", "P2"],
    "doctor_assignment": {"P1": "senior", "P2": "junior"},
    "ambulance_assignment": ["P1"],
    "transfer_decision": [],
    "explanation": "P1 has critical cardiac symptoms requiring immediate senior attention."
}

obs, reward, done, info = env.step(action)
print(f"Reward: {reward}, Done: {done}")
print(f"Grade: {info.get('final_grade', {}).get('total_score', 'N/A')}")
```

---

## 🐳 Docker Deployment

### Build Locally

```bash
docker build -t hospital-er-triage .
```

### Run Locally

```bash
# With HF_TOKEN (mandatory)
docker run -e HF_TOKEN=your-token-here hospital-er-triage

# Override model/endpoint
docker run -e HF_TOKEN=your-token -e API_BASE_URL=https://api.openai.com/v1 -e MODEL_NAME=gpt-4o hospital-er-triage

# Run a specific task
docker run -e HF_TOKEN=your-token-here hospital-er-triage \
    python inference.py --task hard

# Offline mode (no API key needed)
docker run hospital-er-triage python inference.py --offline
```

---

## 🤗 Hugging Face Spaces Deployment

This environment is designed to run as a **containerized Hugging Face Space** tagged with `openenv`.

### Option 1: Deploy via OpenEnv CLI

```bash
# Install openenv CLI
pip install openenv

# Push to Hugging Face Spaces
openenv push --repo-id your-username/hospital-er-triage
```

### Option 2: Deploy Manually via Git

```bash
# 1. Create a new Space on huggingface.co/new-space
#    - SDK: Docker
#    - Hardware: CPU Basic

# 2. Clone the Space repo
git clone https://huggingface.co/spaces/your-username/hospital-er-triage
cd hospital-er-triage

# 3. Copy your project files into the Space repo
#    (the Dockerfile and README.md front matter are already configured)

# 4. Push to trigger build
git add .
git commit -m "Deploy Hospital ER Triage environment"
git push
```

### Setting API Credentials on HF Spaces

1. Go to your Space's **Settings** tab
2. Under **Variables and secrets**, click **New secret**
3. Add `HF_TOKEN` with your API key as the value
4. The Space will automatically restart with the secret available

> **Note:** The inference script reads three mandatory env vars: `HF_TOKEN` (API key, required), `API_BASE_URL` (defaults to `https://router.huggingface.co/v1`), and `MODEL_NAME` (defaults to `Qwen/Qwen2.5-72B-Instruct`).

---

## 📁 Project Structure

```
hospital-er-triage/
├── env/
│   ├── __init__.py          # Package exports
│   ├── environment.py       # HospitalEnv class (reset/step/state)
│   ├── models.py            # Patient, Resources, Action, Observation
│   ├── tasks.py             # Easy/Medium/Hard task definitions
│   └── grader.py            # Deterministic grading system
├── inference.py             # OpenAI inference loop
├── openenv.yaml             # OpenEnv specification
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container deployment
└── README.md                # This file
```

---

## 🔑 Design Decisions

1. **Deterministic grading**: The `Grader` uses normalised rank distance and set overlap — no randomness in scoring.
2. **Continuous rewards**: Every action gets meaningful feedback, enabling gradient-based learning.
3. **Hidden ground truth**: The agent only sees observable patient data; true severity is hidden — forcing genuine reasoning.
4. **Misleading presentations**: Panic attacks mimicking cardiac events (P2 in hard task) test robustness.
5. **Escalation mechanics**: Untreated patients probabilistically worsen, rewarding proactive resource allocation.
6. **Explanation scoring**: Encourages interpretable AI decisions — critical for medical applications.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
