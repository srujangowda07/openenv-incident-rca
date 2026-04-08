import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment.env import IncidentRCAEnv, ActionModel
from graders.grader import IncidentRCAGrader
from tasks.task_definitions import TASKS, list_tasks

app = FastAPI(
    title="Incident RCA Environment",
    description="Deterministic RL environment for debugging production incidents using logs, metrics, and traces",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, IncidentRCAEnv] = {}
_episodes: dict[str, dict] = {}


class ResetRequest(BaseModel):
    task_id: str = "easy_001"
    seed: int | None = None


class StepRequest(BaseModel):
    session_id: str
    action: dict


class GradeRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}

@app.get("/")
def root():
    return {
        "project": "Incident RCA Environment",
        "status": "running",
        "description": "Deterministic RL environment for debugging production incidents (logs, metrics, traces)",
        "how_it_works": [
            "1. POST /reset → start new investigation",
            "2. POST /step → interact with environment",
            "3. POST /grade → evaluate final diagnosis"
        ],
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "reset": "/reset",
            "step": "/step",
            "state": "/state/{session_id}",
            "grade": "/grade",
            "close": "/session/{session_id}",
            "docs": "/docs"
        }
    }

@app.get("/tasks")
def get_tasks(difficulty: str | None = None):
    return {"tasks": list_tasks(difficulty)}


@app.post("/reset")
def reset(req: ResetRequest):
    session_id = str(uuid.uuid4())
    
    # Cap active sessions to prevent production memory leaks
    if len(_sessions) > 1000:
        oldest_session = list(_sessions.keys())[0]
        _sessions.pop(oldest_session, None)
        _episodes.pop(oldest_session, None)

    env = IncidentRCAEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env
    _episodes[session_id] = {
        "task_id": req.task_id,
        "actions_taken": [],
        "max_steps": env.max_steps,
    }
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="session not found — call /reset first")
    env = _sessions[req.session_id]
    try:
        action = ActionModel(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"invalid action format: {e}")

    obs, reward, done, info = env.step(action)
    ep = _episodes[req.session_id]
    ep["actions_taken"].append((action.model_dump(), reward.model_dump()))

    if done:
        ep["final_state"] = env.state()
        ep["info"] = info.model_dump()

    return {
        "session_id": req.session_id,
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump(),
    }


@app.get("/state/{session_id}")
def state(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="session not found")
    return _sessions[session_id].state()


@app.post("/grade")
def grade(req: GradeRequest):
    if req.session_id not in _episodes:
        raise HTTPException(status_code=404, detail="session not found")
    ep = _episodes[req.session_id]
    if "final_state" not in ep:
        raise HTTPException(status_code=400, detail="episode not complete yet")
    grader = IncidentRCAGrader()
    result = grader.grade(ep)
    return {
        "session_id": req.session_id,
        "score": result.score,
        "passed": result.passed,
        "breakdown": result.breakdown,
        "feedback": result.feedback,
    }


@app.delete("/session/{session_id}")
def close(session_id: str):
    _sessions.pop(session_id, None)
    _episodes.pop(session_id, None)
    return {"status": "closed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
