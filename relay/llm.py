"""LLM backends. `ClaudeLLM` calls the Anthropic API; `MockLLM` is an offline stand-in for testing."""
import json
import os
import random
import re
import time


class BudgetExceeded(RuntimeError):
    pass


class ClaudeLLM:
    def __init__(self, model, max_calls=60, max_tokens=400):
        try:
            import anthropic
        except ImportError as e:
            raise SystemExit("anthropic が入っていません。 pip install -r requirements.txt を実行してください。") from e
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY が設定されていません。.env.example を .env にコピーして、自分のキーを入れてください。")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def __call__(self, system, messages):
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"呼び出し回数の上限（{self.max_calls}回）に達しました")
        self.calls += 1
        for attempt in range(3):
            try:
                r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                                system=system, messages=messages)
                break
            except Exception as e:  # rate limit / transient errors
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        u = getattr(r, "usage", None)
        if u is not None:
            self.input_tokens += getattr(u, "input_tokens", 0) or 0
            self.output_tokens += getattr(u, "output_tokens", 0) or 0
        return "".join(getattr(b, "text", "") for b in r.content)


class MockLLM:
    """API を使わずに動きを確かめるための代役。決まった確率で返事・転送・見送りを返す。"""

    def __init__(self, seed=0, max_calls=10_000):
        self.rng = random.Random(seed)
        self.calls = 0
        self.max_calls = max_calls
        self.input_tokens = self.output_tokens = 0

    def __call__(self, system, messages):
        if self.calls >= self.max_calls:
            raise BudgetExceeded("mock budget")
        self.calls += 1
        last = messages[-1]["content"]
        if '"action"' in system or "action" in last:
            contacts = re.findall(r"- (\w+)（", system)
            w = {"REPLY": 0.25, "FORWARD": 0.35, "REPLY_AND_FORWARD": 0.15, "DROP": 0.25}
            # 教師データの割合が見えていたら、それに寄せる（モックで効果を確かめるため）
            lab = {"何もしない": "DROP", "返事をする": "REPLY", "誰かに回す": "FORWARD", "返事をして回す": "REPLY_AND_FORWARD"}
            for ja, pct in re.findall(r"- (何もしない|返事をして回す|返事をする|誰かに回す)：(\d+)%", system):
                w[lab[ja]] = 0.3 * w[lab[ja]] + 0.7 * int(pct) / 100
            act = self.rng.choices(list(w), list(w.values()))[0]
            fw = self.rng.sample(contacts, k=min(len(contacts), self.rng.randint(1, 2))) if contacts and "FORWARD" in act else []
            return json.dumps({"action": act, "forward_to": fw,
                               "message": "（モック）近くの公園で似た猫を見ました。" if "REPLY" in act else "",
                               "forward_note": "（モック）詳しそうな人に回します。" if fw else "",
                               "reason": "mock"}, ensure_ascii=False)
        return f"（モック発言 {self.calls}）"


def parse_json(text):
    """モデルの返事から JSON を取り出す（```json で囲まれていても、前後に文があってもよい）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON が見つかりません")
    return json.loads(m.group(0))
