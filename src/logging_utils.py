"""Plain structured JSONL logging -- no dashboard, no secrets logged."""
from __future__ import annotations

import json
import datetime


def log_turn(log_path: str | None, *, user_message: str, retrieved: list[dict],
             tool_calls: list[dict], response: dict) -> None:
    if not log_path:
        return
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user_message": user_message,
        "retrieved_passages": [
            {"source": r["source"], "status": r["status"], "score": r["score"]}
            for r in retrieved
        ],
        "tool_calls": tool_calls,
        "final_response": {
            "answer": response["answer"],
            "sources": response["sources"],
            "handoff": response["handoff"],
        },
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
