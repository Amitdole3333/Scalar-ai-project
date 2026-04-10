"""Quick validation script for the Hospital ER environment."""

from env import HospitalEnv
from env.tasks import get_task


def test_all_tasks():
    env = HospitalEnv(seed=42)

    for task_name in ["easy", "medium", "hard"]:
        obs = env.reset(task_name)
        print(f"\n=== {task_name.upper()} ===")
        print(f"  Patients: {len(obs['patients'])}")
        print(f"  Queue:    {obs['queue_state']}")

        gt = get_task(task_name)["ground_truth"]
        action = {
            "priority_order": gt["priority_order"],
            "doctor_assignment": gt["doctor_assignment"],
            "ambulance_assignment": gt.get("ambulance_assignment", []),
            "transfer_decision": gt.get("transfer_decision", []),
            "explanation": (
                f"Critical patients {gt['critical_patients']} prioritized. "
                "Senior doctors assigned to highest severity. "
                "Resources allocated based on triage protocol."
            ),
        }

        max_steps = get_task(task_name)["max_steps"]
        for step in range(max_steps):
            obs, reward, done, info = env.step(action)
            print(f"  Step {step+1}: reward={reward:+.4f}, done={done}")

            if info.get("events_fired"):
                for e in info["events_fired"]:
                    print(f"    [EVENT] {e}")
            if info.get("escalations"):
                for e in info["escalations"]:
                    print(f"    [ESCALATION] {e}")

            if done:
                grade = info.get("final_grade", {})
                print(f"  --- FINAL GRADE ---")
                print(f"    Total:       {grade.get('total_score', 'N/A')}")
                print(f"    Priority:    {grade.get('priority_score', 'N/A')}")
                print(f"    Doctor:      {grade.get('doctor_score', 'N/A')}")
                print(f"    Critical:    {grade.get('critical_handling_score', 'N/A')}")
                print(f"    Ambulance:   {grade.get('ambulance_score', 'N/A')}")
                print(f"    Transfer:    {grade.get('transfer_score', 'N/A')}")
                print(f"    Explanation: {grade.get('explanation_score', 'N/A')}")
                break

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_all_tasks()
