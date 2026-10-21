"""
協力の教師データ（実際の行動の記録）を読み込んで、エージェントに見せる。

教師データは JSONL（1行1件）で、次の項目を持つ：
  request_title  頼みのタイトル
  request_body   頼みの本文（個人情報は消しておくこと）
  action         実際の行動: none / reply / share / reply_share
                 （cbpipe の none/reply/share と同じ語彙。両方した場合は reply_share）
  hop            何人目で受け取ったか（任意）
  tie            依頼者・送り手との関係: strong / weak / unknown（任意）
  reply_text     返事の文（任意・個人情報は消すこと）
  forward_note   回すときに添えた一言（任意）

使い方は2通り：
  examples … 実際の例を何件か、そのまま見せる（few-shot）
  stats    … 実際に各行動をとった割合だけを見せる（ベースレートの調整）
"""
import json
import random
from collections import Counter

ACTION_MAP = {"none": "DROP", "reply": "REPLY", "share": "FORWARD", "reply_share": "REPLY_AND_FORWARD"}
LABEL_JA = {"DROP": "何もしない", "REPLY": "返事をする", "FORWARD": "誰かに回す", "REPLY_AND_FORWARD": "返事をして回す"}


def load_teacher(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            a = str(r.get("action", "")).lower()
            if a not in ACTION_MAP:
                raise ValueError(f"{path}:{i} action が不正です: {a}")
            r["agent_action"] = ACTION_MAP[a]
            rows.append(r)
    if not rows:
        raise ValueError("教師データが空です")
    return rows


def distribution(rows):
    c = Counter(r["agent_action"] for r in rows)
    n = sum(c.values())
    return {a: c.get(a, 0) / n for a in LABEL_JA}


class TeacherPrompt:
    def __init__(self, rows, mode="both", k=6, seed=0):
        assert mode in ("examples", "stats", "both")
        self.rows, self.mode, self.k = rows, mode, k
        self.rng = random.Random(seed)
        self.dist = distribution(rows)

    def block(self):
        parts = []
        if self.mode in ("stats", "both"):
            lines = "\n".join(f"- {LABEL_JA[a]}：{p*100:.0f}%" for a, p in self.dist.items())
            parts.append(f"【参考：実際に同じような頼みを受け取った人の行動の割合（{len(self.rows)}件）】\n{lines}\n"
                         "実際の人は、このくらいの割合で行動しています。あなたの判断の目安にしてください。")
        if self.mode in ("examples", "both"):
            ex = self.rng.sample(self.rows, k=min(self.k, len(self.rows)))
            lines = []
            for r in ex:
                s = f"- 頼み「{r.get('request_title', '')}」"
                if r.get("tie"):
                    s += f"（関係：{r['tie']}）"
                s += f" → {LABEL_JA[r['agent_action']]}"
                if r.get("reply_text"):
                    s += f"／返事「{r['reply_text'][:60]}」"
                if r.get("forward_note"):
                    s += f"／添えた一言「{r['forward_note'][:40]}」"
                lines.append(s)
            parts.append("【参考：実際の人の行動の例】\n" + "\n".join(lines))
        return "\n\n".join(parts)
