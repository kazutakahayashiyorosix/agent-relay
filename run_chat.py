"""
2つの AI に会話させる。
  python run_chat.py
  python run_chat.py --turns 10 --mock
"""
import argparse
import os
import time

from relay.chat import run_chat
from relay.llm import ClaudeLLM, MockLLM
from run_relay import load_dotenv


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="2つの AI エージェントに会話させる")
    ap.add_argument("--model", default=os.environ.get("RELAY_MODEL", "claude-haiku-4-5-20251001"))
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    A = {"name": "依頼者", "persona": "飼い猫が昨日から帰ってこず困っている会社員。近所づきあいは少ない。"}
    B = {"name": "近所の大学生", "persona": "サークルの友人が多く、地域の掲示板もよく見ている。親切だが忙しい。"}
    opening = "すみません、うちの三毛猫が昨日から戻ってこなくて…。この辺りで見かけませんでしたか？"
    llm = MockLLM(max_calls=a.turns) if a.mock else ClaudeLLM(a.model, max_calls=a.turns)
    os.makedirs("runs", exist_ok=True)
    run_chat(A, B, opening, llm, turns=a.turns,
             log_path=os.path.join("runs", f"chat_{time.strftime('%Y%m%d_%H%M%S')}.json"))


if __name__ == "__main__":
    main()
