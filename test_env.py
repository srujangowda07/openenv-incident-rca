from environment.env import IncidentRCAEnv, ActionModel
from graders.grader import IncidentRCAGrader
import json

def test_medium_path():
    env = IncidentRCAEnv(task_id="medium_001", seed=42)
    obs = env.reset()
    grader = IncidentRCAGrader()

    actions = [
        ActionModel(action_type="query_dependencies", parameters={"service": "product-search"}),
        ActionModel(action_type="query_metrics", parameters={"service": "inventory-service", "metric_name": "latency"}),
        ActionModel(action_type="query_dependencies", parameters={"service": "inventory-service"}),
        ActionModel(action_type="query_metrics", parameters={"service": "mysql-primary", "metric_name": "cpu_usage"}),
        ActionModel(action_type="grep_logs", parameters={"service": "mysql-primary", "keyword": "slow"}),
        ActionModel(action_type="submit_diagnosis", parameters={"root_cause_service": "mysql-primary", "cause_type": "missing index slow query full table scan"})
    ]

    for a in actions:
        obs, reward, done, info = env.step(a)
        print(f"Action: {a.action_type}")
        print(f"Reward: {reward.total}")
        
    episode_log = {
        "task_id": env.task_id,
        "scenario": env._scenario,
        "actions_taken": [(a.model_dump(), {}) for a in actions],
        "final_state": env.state(),
        "info": info.model_dump(),
    }
    
    result = grader.grade(episode_log)
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump({
            "score": result.score,
            "breakdown": result.breakdown,
            "feedback": result.feedback
        }, f, indent=2)

if __name__ == "__main__":
    test_medium_path()
