"""2つの AI エージェントに、決まった回数だけ会話させる。"""
import json
import time

from .llm import BudgetExceeded


def run_chat(a, b, opening, llm, turns=6, log_path=None, verbose=True):
    """a, b: {"name":..., "persona":...}。opening: a の最初の発言。"""
    history = {a["name"]: [], b["name"]: []}
    sys_tpl = ("あなたは「{me}」です。人物像：{persona}\n相手は「{other}」です。"
               "自然な会話として、毎回3文以内で返してください。話が終わったと思ったら最後に [END] と書いてください。")
    speakers = [a, b]
    transcript = [{"speaker": a["name"], "text": opening}]
    history[b["name"]].append({"role": "user", "content": opening})
    history[a["name"]].append({"role": "assistant", "content": opening})
    if verbose:
        print(f"{a['name']}: {opening}")
    t0 = time.time()
    for i in range(1, turns):
        me, other = speakers[i % 2], speakers[(i + 1) % 2]
        system = sys_tpl.format(me=me["name"], persona=me["persona"], other=other["name"])
        try:
            text = llm(system, history[me["name"]]).strip()
        except BudgetExceeded:
            break
        transcript.append({"speaker": me["name"], "text": text, "t": round(time.time() - t0, 2)})
        history[me["name"]].append({"role": "assistant", "content": text})
        history[other["name"]].append({"role": "user", "content": text})
        if verbose:
            print(f"{me['name']}: {text}")
        if "[END]" in text:
            break
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=1)
    return transcript
