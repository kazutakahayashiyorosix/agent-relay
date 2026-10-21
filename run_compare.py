"""
教師データあり／なしで、エージェントの協力のしかたがどう変わるかを比べる。

  python run_compare.py --teacher teacher/example_synthetic.jsonl --reps 3
  python run_compare.py --teacher teacher/my_real_data.jsonl --mode stats --reps 5
  python run_compare.py --teacher teacher/example_synthetic.jsonl --mock --reps 20   # API なしで動作確認

同じ「世界」を、教師データなし（baseline）と教師データあり（teacher）で reps 回ずつ動かし、
・行動の割合（何もしない／返事／回す／両方）
・届いた人数、返事の数、何人先まで届いたか
・教師データの行動割合とのずれ（全変動距離 TV：0＝完全一致、1＝まったく違う）
を表にして、runs/compare_日時.csv にも保存する。
"""
import argparse
import csv
import os
import statistics as st
import time

from relay.llm import ClaudeLLM, MockLLM
from relay.sim import load_world, run
from relay.teacher import LABEL_JA, TeacherPrompt, distribution, load_teacher
from run_relay import load_dotenv

ACTS = list(LABEL_JA)


def tv(p, q):
    return 0.5 * sum(abs(p.get(a, 0) - q.get(a, 0)) for a in ACTS)


def one_condition(world, make_llm, teacher, reps, args, tag):
    runs, decisions, calls = [], [], 0
    for r in range(reps):
        llm = make_llm(r)
        t = TeacherPrompt(teacher, mode=args.mode, k=args.k, seed=r) if teacher else None
        log = os.path.join("runs", f"{tag}_{args.stamp}_r{r}.jsonl")
        events, s = run(world, llm, max_hops=args.max_hops, max_receipts=args.max_receipts,
                        log_path=log, verbose=False, teacher=t)
        runs.append(s)
        decisions += [e["action"] for e in events if e["type"] == "DECISION"]
        calls += llm.calls
        if s.get("stopped_by"):
            print(f"  [{tag} r{r}] 途中で止まりました：{s['stopped_by']}")
    n = max(1, len(decisions))
    dist = {a: decisions.count(a) / n for a in ACTS}
    return runs, dist, len(decisions), calls


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{st.mean(xs):.2f}" + (f" ± {st.stdev(xs):.2f}" if len(xs) > 1 else "")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="教師データあり／なしで協力のしかたを比べる")
    ap.add_argument("--world", default="examples/lost_cat.json")
    ap.add_argument("--teacher", required=True, help="教師データ（JSONL）")
    ap.add_argument("--mode", choices=["examples", "stats", "both"], default="both")
    ap.add_argument("--k", type=int, default=6, help="examples で見せる例の数")
    ap.add_argument("--reps", type=int, default=3, help="条件ごとの繰り返し回数")
    ap.add_argument("--model", default=os.environ.get("RELAY_MODEL", "claude-haiku-4-5-20251001"))
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--max-receipts", type=int, default=30)
    ap.add_argument("--max-calls", type=int, default=300, help="全体の API 呼び出し回数の上限")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    args.stamp = time.strftime("%Y%m%d_%H%M%S")

    world = load_world(args.world)
    teacher = load_teacher(args.teacher)
    tdist = distribution(teacher)
    per_run = max(1, args.max_calls // (2 * args.reps))
    if args.mock:
        make = lambda r: MockLLM(seed=1000 + r, max_calls=per_run)
    else:
        make = lambda r: ClaudeLLM(args.model, max_calls=per_run)
    os.makedirs("runs", exist_ok=True)
    print(f"教師データ：{len(teacher)}件　モード：{args.mode}　繰り返し：{args.reps}回／条件　1回あたりの呼び出し上限：{per_run}\n")

    res = {}
    for tag, tch in [("baseline", None), ("teacher", teacher)]:
        print(f"{tag} を実行中…")
        res[tag] = one_condition(world, make, tch, args.reps, args, tag)

    print("\n=== 行動の割合 ===")
    print(f"{'':16}{'教師データ':>10}{'baseline':>10}{'teacher':>10}")
    for a in ACTS:
        print(f"{LABEL_JA[a]:<14}{tdist[a]*100:>9.0f}%{res['baseline'][1][a]*100:>9.0f}%{res['teacher'][1][a]*100:>9.0f}%")
    print(f"{'判断の件数':<14}{len(teacher):>10}{res['baseline'][2]:>10}{res['teacher'][2]:>10}")
    print("\n=== 教師データとのずれ（TV、小さいほど実際の行動に近い） ===")
    for tag in res:
        print(f"{tag:<10}{tv(tdist, res[tag][1]):.3f}")
    print("\n=== 広がり方（平均 ± 標準偏差） ===")
    for key, lab in [("reached", "届いた人数"), ("replies", "返事の数"), ("max_hop", "何人先まで"), ("forwards", "回した回数")]:
        print(f"{lab:<10}baseline {mean_sd([r[key] for r in res['baseline'][0]]):<14} teacher {mean_sd([r[key] for r in res['teacher'][0]])}")
    print(f"\nAPI 呼び出し：baseline {res['baseline'][3]}回 / teacher {res['teacher'][3]}回")

    out = os.path.join("runs", f"compare_{args.stamp}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["condition", "rep", "reached", "replies", "max_hop", "forwards", "duplicates", "parse_errors"] + ACTS)
        for tag in res:
            for i, r in enumerate(res[tag][0]):
                w.writerow([tag, i, r["reached"], r["replies"], r["max_hop"], r["forwards"], r["duplicates"], r["parse_errors"]]
                           + [r["actions"].get(a, 0) for a in ACTS])
    print(f"結果：{out}")
    if args.reps < 5:
        print("\n※ 繰り返しが少ないので、差は偶然の範囲かもしれません。結論を出すなら reps を増やしてください。")


if __name__ == "__main__":
    main()
