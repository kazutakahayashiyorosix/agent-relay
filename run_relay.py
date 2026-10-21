"""
使い方:
  python run_relay.py                       # examples/lost_cat.json を Claude で実行
  python run_relay.py --world examples/lost_cat.json --model claude-haiku-4-5-20251001
  python run_relay.py --mock                # API を使わずに動作確認（お金はかかりません）
"""
import argparse
import json
import os
import time

from relay.llm import ClaudeLLM, MockLLM
from relay.sim import load_world, run


def load_dotenv(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="AI エージェント同士で、頼みが手渡しで広がる様子を再現する")
    ap.add_argument("--world", default="examples/lost_cat.json")
    ap.add_argument("--model", default=os.environ.get("RELAY_MODEL", "claude-haiku-4-5-20251001"))
    ap.add_argument("--max-hops", type=int, default=4, help="何人先まで回すか")
    ap.add_argument("--max-receipts", type=int, default=30, help="受け取る人数の上限")
    ap.add_argument("--max-calls", type=int, default=40, help="API 呼び出し回数の上限（料金の上限）")
    ap.add_argument("--mock", action="store_true", help="API を使わずにダミーの判断で動かす")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    world = load_world(a.world)
    llm = MockLLM(seed=a.seed, max_calls=a.max_calls) if a.mock else ClaudeLLM(a.model, max_calls=a.max_calls)
    os.makedirs("runs", exist_ok=True)
    log_path = os.path.join("runs", f"relay_{time.strftime('%Y%m%d_%H%M%S')}{'_mock' if a.mock else ''}.jsonl")
    print(f"頼み：{world['request']['title']}（依頼者：{world['request']['from']}）\n")
    _, s = run(world, llm, max_hops=a.max_hops, max_receipts=a.max_receipts, log_path=log_path)
    print("\n--- まとめ ---")
    print(json.dumps(s, ensure_ascii=False, indent=1))
    print(f"\nログ：{log_path}")


if __name__ == "__main__":
    main()
