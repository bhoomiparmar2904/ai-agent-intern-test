"""Minimal CLI for the Aster & Row support agent.

Run:  python src/cli.py
Debug mode (prints retrieval + tool trace for every turn):  python src/cli.py --debug
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from agent import Agent, Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Print retrieval/tool trace")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent = Agent(
        kb_dir=os.path.join(base, "knowledge-base"),
        orders_path=os.path.join(base, "data", "orders.json"),
        log_path=os.path.join(base, "logs", "session.jsonl"),
    )
    session = Session()

    print("Aster & Row Support Agent (type 'exit' to quit)\n")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in ("exit", "quit"):
            break
        if not user_text:
            continue

        result = agent.handle_message(session, user_text)

        if args.debug:
            print("\n--- retrieved ---")
            for r in result["retrieved"]:
                print(f"  [{r['score']}] {r['source']} ({r['status']})")
            print("--- tool calls ---")
            for t in result["tool_calls"]:
                print(f"  {t['tool']}({t['arguments']}) -> {t['result']}")
            print("-----------------\n")

        print(f"\nagent> {result['answer']}")
        if result["sources"]:
            print(f"sources: {', '.join(result['sources'])}")
        if result["handoff"]:
            print("[recommending human support]")
        print()


if __name__ == "__main__":
    sys.exit(main())
