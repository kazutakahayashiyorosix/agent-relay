"""
実際の行動データを見せると、AI の「人は助けるか」の見積もりが現実に近づくか（公開データでの検証）

データ：Random Acts of Pizza（Althoff, Danescu-Niculescu-Mizil & Jurafsky, ICWSM 2014）
  Reddit の r/Random_Acts_Of_Pizza に投稿された 5,671 件の「ピザをください」というお願いと、
  実際に誰かがピザを贈ったかどうか（requester_received_pizza）。実際に助けてもらえた割合は約 25%。
  データは同梱していません。配布元から入手して --data で場所を指定してください。

比べる3つの条件（どれも同じ評価用のお願いに答える）:
  baseline  何も見せない
  stats     過去のお願いで、実際に助けてもらえた割合だけ見せる
  examples  過去のお願いの例（本文の冒頭と、助けてもらえたか）を k 件見せる

各お願いについて、AI に「このお願いは誰かに助けてもらえるか」を 0〜100 の確率で答えさせ、
  ・AI が予想した平均の成功率 と 実際の成功率 の差（=見積もりのずれ）
  ・Brier スコア（確率の当たり具合。小さいほど良い）
  ・AUC（助けてもらえるお願いを見分けられているか。0.5 が当てずっぽう）
を条件ごとに比べる。過去（参考に見せる側）と評価する側は、重ならないように分けてある（--split time で「古い投稿を参考に新しい投稿を評価」も可）。

使い方:
  python experiments/raop_calibration.py --data path/to/pizza_request_dataset.json --n 60
  python experiments/raop_calibration.py --data ... --mock          # API なしで動作確認
"""
import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relay.llm import BudgetExceeded, ClaudeLLM, parse_json  # noqa: E402
from run_relay import load_dotenv  # noqa: E402

SYSTEM = """You are a long-time member of the Reddit community r/Random_Acts_Of_Pizza, where people ask strangers to buy them a pizza and other users sometimes do.
You will read one request. Estimate how likely it is that someone in the community actually gives this person a pizza.
Be realistic: many requests on this forum never receive a pizza.
{context}
Answer with JSON only: {{"probability": <integer 0-100>, "reason": "<one short sentence>"}}"""

USER = "Title: {title}\n\nRequest:\n{body}"


def clip(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n].rstrip() + "…"


def load(path, n_eval, n_pool, seed, split="random"):
    d = json.load(open(path, encoding="utf-8"))
    d = [r for r in d if r.get("request_text_edit_aware", "").strip()]
    rng = random.Random(seed)
    if split == "time":
        d.sort(key=lambda r: r["unix_timestamp_of_request_utc"])   # 過去を参考に見せ、未来で評価する
    else:
        rng.shuffle(d)                                             # 参考用と評価用を重ならないように分ける
    cut = int(len(d) * 0.7)
    past, future = d[:cut], d[cut:]
    ev = rng.sample(future, k=min(n_eval, len(future)))
    pool = rng.sample(past, k=min(n_pool, len(past)))
    base_rate = sum(r["requester_received_pizza"] for r in past) / len(past)
    return ev, pool, base_rate, len(past)


def context_for(cond, pool, base_rate, n_past, k, rng):
    if cond == "baseline":
        return ""
    if cond == "stats":
        return (f"\nReference from this community's history: of {n_past} earlier requests, "
                f"{base_rate*100:.0f}% actually received a pizza.\n")
    ex = rng.sample(pool, k=k)
    lines = [f'- "{clip(r["request_title"], 80)}" / "{clip(r["request_text_edit_aware"], 160)}" → '
             f'{"RECEIVED a pizza" if r["requester_received_pizza"] else "did NOT receive a pizza"}' for r in ex]
    return "\nHere are some earlier requests from this community and what actually happened:\n" + "\n".join(lines) + "\n"


def auc(y, p):
    pos = [pi for yi, pi in zip(y, p) if yi]
    neg = [pi for yi, pi in zip(y, p) if not yi]
    if not pos or not neg:
        return None
    s = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


class MockRAOP:
    """API を使わない代役：何も見ないと楽観的、参考を見ると現実寄りになる（流れの確認用）。"""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.calls = self.input_tokens = self.output_tokens = 0

    def __call__(self, system, messages):
        self.calls += 1
        if "received a pizza" in system.lower() or "actually received" in system:
            p = self.rng.gauss(30, 15)
        else:
            p = self.rng.gauss(65, 15)
        return json.dumps({"probability": int(max(0, min(100, p))), "reason": "mock"})


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="実データを見せると AI の見積もりが現実に近づくかを確かめる（RAOP）")
    ap.add_argument("--data", required=True, help="pizza_request_dataset.json の場所")
    ap.add_argument("--n", type=int, default=100, help="評価に使うお願いの数（条件ごと）")
    ap.add_argument("--k", type=int, default=8, help="examples で見せる例の数")
    ap.add_argument("--conditions", default="baseline,stats,examples")
    ap.add_argument("--model", default=os.environ.get("RELAY_MODEL", "claude-haiku-4-5-20251001"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["random", "time"], default="random",
                    help="random：参考用と評価用をランダムに分ける／time：古い投稿を参考に、新しい投稿で評価（時期による変化も含む）")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    ev, pool, base_rate, n_past = load(a.data, a.n, 400, a.seed, a.split)
    actual = sum(r["requester_received_pizza"] for r in ev) / len(ev)
    llm = MockRAOP(a.seed) if a.mock else ClaudeLLM(a.model, max_calls=len(ev) * len(conds) + 5, max_tokens=150)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("runs", exist_ok=True)
    rows = []
    print(f"評価：{len(ev)}件（実際に助けてもらえた割合 {actual*100:.0f}%）　過去の割合 {base_rate*100:.0f}%（{n_past}件）\n")
    for cond in conds:
        rng = random.Random(a.seed + 1)
        print(f"{cond} を実行中…", flush=True)
        for i, r in enumerate(ev):
            sysmsg = SYSTEM.format(context=context_for(cond, pool, base_rate, n_past, a.k, rng))
            user = USER.format(title=clip(r["request_title"], 200), body=clip(r["request_text_edit_aware"], 1500))
            try:
                raw = llm(sysmsg, [{"role": "user", "content": user}])
                prob = float(parse_json(raw)["probability"]) / 100
                prob = min(1.0, max(0.0, prob))
            except BudgetExceeded:
                break
            except Exception:
                prob = None
            rows.append({"condition": cond, "request_id": r["request_id"], "actual": int(r["requester_received_pizza"]), "prob": prob})

    out_csv = os.path.join("runs", f"raop_{stamp}{'_mock' if a.mock else ''}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "request_id", "actual", "prob"])
        w.writeheader()
        w.writerows(rows)

    summary = []
    print(f"\n{'条件':<10}{'AIの予想':>10}{'実際':>8}{'ずれ':>8}{'Brier':>8}{'AUC':>7}{'無効':>6}")
    for cond in conds:
        rs = [x for x in rows if x["condition"] == cond and x["prob"] is not None]
        bad = sum(1 for x in rows if x["condition"] == cond and x["prob"] is None)
        if not rs:
            continue
        y = [x["actual"] for x in rs]
        p = [x["prob"] for x in rs]
        mean_p = sum(p) / len(p)
        act = sum(y) / len(y)
        brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p)
        a_ = auc(y, p)
        summary.append({"condition": cond, "predicted": mean_p, "actual": act, "gap": mean_p - act, "brier": brier, "auc": a_, "n": len(rs)})
        print(f"{cond:<10}{mean_p*100:>9.0f}%{act*100:>7.0f}%{(mean_p-act)*100:>+7.0f}pt{brier:>8.3f}{(a_ if a_ is not None else float('nan')):>7.2f}{bad:>6}")
    ref_brier = actual * (1 - actual)
    print(f"\n参考：常に実際の割合（{actual*100:.0f}%）を答えた場合の Brier = {ref_brier:.3f}")
    print(f"API 呼び出し {llm.calls} 回　入力 {llm.input_tokens} / 出力 {llm.output_tokens} トークン")
    print(f"結果：{out_csv}")

    with open(out_csv.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "mock" if a.mock else a.model, "n_eval": len(ev), "k": a.k, "past_rate": base_rate,
                   "results": summary}, f, ensure_ascii=False, indent=1)
    try:
        chart(summary, actual, out_csv.replace(".csv", ".png"), "mock" if a.mock else a.model)
        print(f"グラフ：{out_csv.replace('.csv', '.png')}")
    except Exception as e:
        print(f"（グラフは作れませんでした：{e}）")
    if len(ev) < 100:
        print("\n※ 評価の件数が少ないので、条件の差は数ポイントなら偶然の範囲です。")
    if not a.mock:
        print("※ このデータは公開されているため、モデルが学習時に見ている可能性があります。")


def chart(summary, actual, path, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"baseline": "No data", "stats": "Base rate shown", "examples": "Real examples shown"}
    names = [labels.get(s["condition"], s["condition"]) for s in summary]
    vals = [s["predicted"] * 100 for s in summary]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    colors = ["#b0a49c"] + ["#e08a1e"] * (len(vals) - 1)
    bars = ax.bar(names, vals, color=colors[:len(vals)], width=0.55)
    ax.axhline(actual * 100, color="#2a1d18", lw=2, ls="--")
    ax.text(-0.45, actual * 100 + 1.5, f"Actual: {actual*100:.0f}%", ha="left", fontsize=10, color="#2a1d18", fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals + [actual * 100]) * 1.25 + 5)
    ax.set_ylabel("Predicted share of requests that get help (%)")
    ax.set_title("Does real behavioural data calibrate an AI's view of cooperation?\n(Random Acts of Pizza, held-out requests)", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.99, 0.01, f"model: {model}", ha="right", fontsize=7, color="#888")
    if model == "mock":
        fig.text(0.5, 0.5, "MOCK — NOT REAL RESULTS", ha="center", va="center", fontsize=26, color="#c00", alpha=0.25, rotation=20)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main()
