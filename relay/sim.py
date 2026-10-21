"""
協力の連鎖（リレー）シミュレーション。
1枚の「頼み」が、依頼者から知り合いへ、その知り合いからさらに次の人へと手渡しで進む様子を、
AI エージェント同士のやりとりで再現する。各エージェントは頼みを受け取るたびに
  REPLY（自分で手伝う）／FORWARD（誰かに回す）／REPLY_AND_FORWARD（両方）／DROP（何もしない）
のどれかを選ぶ。
"""
import json
import time
from collections import deque

from .llm import BudgetExceeded, parse_json

ACTIONS = {"REPLY", "FORWARD", "REPLY_AND_FORWARD", "DROP"}

AGENT_SYSTEM = """あなたは「{name}」という人物として振る舞います。
人物像：{persona}
あなたは知り合いから、困りごとの頼み（デジタルポスター）を受け取りました。
あなたの知り合い（回せる相手）は次の人たちです：
{contacts}

この頼みに対して、あなたが実際にとりそうな行動を1つ選んでください。
- REPLY：自分が手伝える・情報を持っているので返事をする
- FORWARD：自分では手伝えないが、役に立ちそうな知り合いに回す
- REPLY_AND_FORWARD：返事もして、さらに知り合いにも回す
- DROP：見たが何もしない（忙しい、関係ない、など。現実ではよくあることです）

人物像に合った、現実的な判断をしてください。全員が親切である必要はありません。
次の JSON だけを出力してください（説明文は不要）：
{{"action": "REPLY|FORWARD|REPLY_AND_FORWARD|DROP",
  "forward_to": ["回す相手の名前（FORWARD系のときだけ、上の知り合いの中から）"],
  "message": "依頼者への返事（REPLY系のときだけ。2文以内）",
  "forward_note": "回すときに添える一言（FORWARD系のときだけ。1文）",
  "reason": "その行動を選んだ理由（1文）"}}"""

AGENT_USER = """【頼み】
タイトル：{title}
内容：{body}

【届いた経路】{path}
{note_line}
あなたの行動を JSON で答えてください。"""


def load_world(path):
    with open(path, encoding="utf-8") as f:
        w = json.load(f)
    names = {p["name"] for p in w["people"]}
    for p in w["people"]:
        for c in p.get("contacts", []):
            if c not in names:
                raise ValueError(f"{p['name']} の知り合い {c} が people にいません")
    if w["request"]["from"] not in names:
        raise ValueError("request.from が people にいません")
    return w


def run(world, llm, max_hops=4, max_receipts=30, log_path=None, verbose=True, teacher=None):
    people = {p["name"]: p for p in world["people"]}
    req = world["request"]
    origin = req["from"]
    events = []
    t0 = time.time()

    def log(ev):
        ev["t"] = round(time.time() - t0, 2)
        events.append(ev)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    seen = {origin}
    queue = deque()
    for c in req.get("send_to", people[origin].get("contacts", [])):
        queue.append((c, origin, [origin], req.get("note", "")))
        log({"type": "SEND", "from": origin, "to": c, "hop": 1})
    receipts = 0
    stopped = None
    while queue:
        name, sender, path, note = queue.popleft()
        hop = len(path)
        if name in seen:
            log({"type": "DUPLICATE", "from": sender, "to": name, "hop": hop})
            continue
        if receipts >= max_receipts:
            stopped = f"受け取りの上限（{max_receipts}件）"
            break
        seen.add(name)
        receipts += 1
        p = people[name]
        contacts = [c for c in p.get("contacts", []) if c not in path]
        contact_lines = "\n".join(f"- {c}（{people[c]['persona'][:40]}）" for c in contacts) or "（回せる知り合いはいません）"
        system = AGENT_SYSTEM.format(name=name, persona=p["persona"], contacts=contact_lines)
        if teacher is not None:
            system += "\n\n" + teacher.block()
        user = AGENT_USER.format(title=req["title"], body=req["body"], path=" → ".join(path + [name]),
                                 note_line=f"回してくれた人の一言：「{note}」" if note else "")
        try:
            raw = llm(system, [{"role": "user", "content": user}])
        except BudgetExceeded as e:
            stopped = str(e)
            break
        try:
            d = parse_json(raw)
            act = str(d.get("action", "")).upper()
            if act not in ACTIONS:
                raise ValueError(act)
        except Exception:
            log({"type": "PARSE_ERROR", "agent": name, "hop": hop, "raw": raw[:500]})
            continue
        fw = [c for c in d.get("forward_to", []) or [] if c in contacts] if "FORWARD" in act else []
        if "FORWARD" in act and not fw:
            act = "REPLY" if act == "REPLY_AND_FORWARD" else "DROP"
        ev = {"type": "DECISION", "agent": name, "from": sender, "hop": hop, "action": act,
              "forward_to": fw, "message": d.get("message", "") if "REPLY" in act else "",
              "reason": d.get("reason", ""), "path": path + [name]}
        log(ev)
        if verbose:
            arrow = f" → {', '.join(fw)}" if fw else ""
            msg = f"「{ev['message']}」" if ev["message"] else ""
            print(f"[hop {hop}] {name}: {act}{arrow} {msg}")
        if hop >= max_hops:
            continue
        for c in fw:
            queue.append((c, name, path + [name], d.get("forward_note", "")))
            log({"type": "SEND", "from": name, "to": c, "hop": hop + 1})
    summary = summarize(events, origin)
    summary["stopped_by"] = stopped
    summary["llm_calls"] = llm.calls
    summary["input_tokens"] = llm.input_tokens
    summary["output_tokens"] = llm.output_tokens
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "SUMMARY", **summary}, ensure_ascii=False) + "\n")
    return events, summary


def summarize(events, origin):
    dec = [e for e in events if e["type"] == "DECISION"]
    count = {a: sum(1 for e in dec if e["action"] == a) for a in sorted(ACTIONS)}
    replies = [e for e in dec if "REPLY" in e["action"]]
    return {
        "reached": len(dec),
        "actions": count,
        "replies": len(replies),
        "max_hop": max((e["hop"] for e in dec), default=0),
        "first_reply_hop": min((e["hop"] for e in replies), default=None),
        "forwards": sum(len(e["forward_to"]) for e in dec),
        "duplicates": sum(1 for e in events if e["type"] == "DUPLICATE"),
        "parse_errors": sum(1 for e in events if e["type"] == "PARSE_ERROR"),
    }
