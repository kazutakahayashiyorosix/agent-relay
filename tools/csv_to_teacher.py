"""
YOROSIX などから書き出した行動データ（CSV）を、教師データ（JSONL）に変換する。

  python tools/csv_to_teacher.py 入力.csv teacher/my_real_data.jsonl

CSV の列（1行＝ある人が頼みを受け取って、どう行動したか）:
  request_title, request_body, action(none/reply/share/reply_share), hop, tie, reply_text, forward_note
  ※ request_title と action は必須。ほかは空でもよい。

変換のときに、メールアドレス・電話番号・URL を自動で伏せ字にします。
ただし名前や住所までは見分けられないので、必ず目で確認してから使ってください。
"""
import csv
import json
import re
import sys

ACTIONS = {"none", "reply", "share", "reply_share"}
PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "［メール］"),
    (re.compile(r"https?://\S+"), "［URL］"),
    (re.compile(r"0\d{1,4}-?\d{1,4}-?\d{3,4}"), "［電話］"),
]


def scrub(s):
    s = (s or "").strip()
    for p, rep in PATTERNS:
        s = p.sub(rep, s)
    return s


def main(src, dst):
    n = skipped = 0
    with open(src, encoding="utf-8-sig") as f, open(dst, "w", encoding="utf-8") as out:
        for row in csv.DictReader(f):
            a = (row.get("action") or "").strip().lower()
            if a not in ACTIONS or not row.get("request_title"):
                skipped += 1
                continue
            r = {"request_title": scrub(row["request_title"]), "request_body": scrub(row.get("request_body")), "action": a}
            if row.get("hop"):
                try:
                    r["hop"] = int(row["hop"])
                except ValueError:
                    pass
            for k in ("tie", "reply_text", "forward_note"):
                if row.get(k):
                    r[k] = scrub(row[k])
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} 件を書き出しました（読み飛ばし {skipped} 件）→ {dst}")
    if n < 30:
        print("※ 件数が少ないです。割合（stats）は不安定になるので、examples モードの方が向いています。")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
