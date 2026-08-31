import os
import sys
from dotenv import load_dotenv

load_dotenv(".env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent import Agent, Session

agent = Agent(
    kb_dir="knowledge-base",
    orders_path="data/orders.json"
)

session = Session()

print("\n=== Aster & Row AI Support Agent Demo ===")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    result = agent.handle_message(session, user_input)

    print("\nAgent:", result["answer"])

    if result.get("sources"):
        print("Sources:", result["sources"])

    print()