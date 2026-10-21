# agent-relay

AI エージェント同士のやりとりで、**1枚の「頼み」が人から人へ手渡しで広がっていく様子**を再現する小さなシミュレーターです。

*A small simulator in which AI agents pass a help request from person to person. Each agent decides to reply, forward, do both, or ignore it, and every step is logged.*

```
[hop 1] たかし: FORWARD → あや
[hop 1] みさき: REPLY_AND_FORWARD → えり 「SNS で拡散しておくね」
[hop 2] えり: REPLY 「病院に保護の連絡が来ていないか確認します」
```

## できること

- **リレー（`run_relay.py`）**：依頼者が知り合いに頼みを送り、受け取った各エージェントが人物像に沿って
  - `REPLY`（自分で手伝う）
  - `FORWARD`（知り合いに回す）
  - `REPLY_AND_FORWARD`（両方）
  - `DROP`（何もしない）

  を選びます。回された相手がまた判断し、頼みが広がっていきます。全ての判断と経路を JSONL で記録します。
- **会話（`run_chat.py`）**：2つのエージェントに決まった回数だけ会話させます。

## 使い方

```bash
pip install -r requirements.txt
cp .env.example .env          # .env を開いて ANTHROPIC_API_KEY に自分のキーを入れる
python run_relay.py           # examples/lost_cat.json で実行
python run_chat.py            # 2人の会話
```

API を使わずに動きだけ確かめるときは `--mock` を付けてください（お金はかかりません）。

```bash
python run_relay.py --mock
```

### 主なオプション

| オプション | 意味 | 既定値 |
|---|---|---|
| `--world` | 人と関係と頼みを書いたファイル | `examples/lost_cat.json` |
| `--model` | 使うモデル | `claude-haiku-4-5-20251001`（`.env` の `RELAY_MODEL` でも指定可） |
| `--max-hops` | 何人先まで回すか | 4 |
| `--max-receipts` | 受け取る人数の上限 | 30 |
| `--max-calls` | API を呼ぶ回数の上限（＝料金の上限） | 40 |

## 自分の「世界」を作る

`examples/lost_cat.json` をコピーして書き換えます。

- `people`：登場人物。`name`、`persona`（人物像）、`contacts`（回せる知り合い）
- `request`：頼み。`from`（依頼者）、`title`、`body`、`send_to`（最初に送る相手）

人物像の書き方で結果が大きく変わります。全員を親切にすると現実離れするので、忙しい人・関心のない人も混ぜてください。

## 教師データを入れて比べる（`run_compare.py`）

実際の行動の記録（協力の教師データ）をエージェントに見せると、協力のしかたがどう変わるかを比べます。
同じ「世界」を **教師データなし（baseline）** と **あり（teacher）** で何回か動かし、次を表にします。

- 行動の割合（何もしない／返事／回す／両方）
- 届いた人数、返事の数、何人先まで届いたか、回した回数
- 教師データの行動割合とのずれ（全変動距離。0 なら実際の行動と同じ割合）

```bash
python run_compare.py --teacher teacher/example_synthetic.jsonl --reps 3          # Claude で
python run_compare.py --teacher teacher/example_synthetic.jsonl --mock --reps 20  # API なしで動作確認
```

| オプション | 意味 |
|---|---|
| `--teacher` | 教師データ（JSONL）。形式は `relay/teacher.py` を参照 |
| `--mode` | `examples`（実際の例を見せる）／`stats`（行動の割合だけ見せる）／`both`（両方・既定） |
| `--k` | examples で見せる例の数（既定 6） |
| `--reps` | 条件ごとの繰り返し回数（既定 3。結論を出すなら 10 以上） |
| `--max-calls` | 全体の API 呼び出し回数の上限（既定 300） |

自分のデータを使うときは、CSV を `tools/csv_to_teacher.py` で変換して `teacher/` に置いてください。
`teacher/` の実データは `.gitignore` で GitHub に上がらないようになっています（合成の例だけ上がります）。

**読み方の注意**
- ここでの「教師データ」は、モデルを学習し直すのではなく、**判断のときに参考として見せる**（few-shot）使い方です。
- 「教師データに近づいた」は「より協力的になった」とは限りません。実際の人の多くが何もしないなら、近づくほど協力は減ります。何を目標にするか（実際の行動の再現か、協力を増やす方法の検証か）を先に決めてください。
- AI の判断にはばらつきがあります。繰り返しが少ないと、差は偶然の範囲かもしれません。

## 公開データでの検証：実データを見せると、AI の見積もりは現実に近づくか（`experiments/raop_calibration.py`）

Random Acts of Pizza（Althoff ほか, ICWSM 2014）の 5,671 件の「ピザをください」というお願いと、実際に誰かが贈ったかどうか（成功率は約 25%）を使います。
各お願いについて AI に「このお願いは助けてもらえるか」を確率で答えさせ、次の3条件を比べます。

| 条件 | AI に見せるもの |
|---|---|
| `baseline` | 何も見せない |
| `stats` | 過去のお願いで実際に助けてもらえた割合 |
| `examples` | 過去のお願いの例と、助けてもらえたかどうか（`--k` 件） |

参考に見せる過去のお願いと、評価するお願いは重なりません。結果は表・CSV・グラフ（PNG）で出ます。

```bash
python experiments/raop_calibration.py --data path/to/pizza_request_dataset.json --n 100
python experiments/raop_calibration.py --data path/to/pizza_request_dataset.json --mock   # API なしで流れだけ確認
```

- データは同梱していません。配布元から入手してください。
- `--split time` にすると、古い投稿を参考に新しい投稿で評価します（時期によって成功率が変わるため、ずれは大きくなります）。
- 見るべき数字：**AI の予想した平均成功率と実際との差**（見積もりのずれ）、Brier スコア（小さいほど良い）、AUC（0.5 は当てずっぽう）。
- `--mock` の結果は作り物です（グラフに「MOCK」と出ます）。発表などには使わないでください。
- このデータは公開されているので、モデルが学習時に見ている可能性があります。

## ログ

`runs/relay_日時.jsonl` に1行1件で記録されます。

| type | 内容 |
|---|---|
| `SEND` | 誰から誰へ送ったか |
| `DECISION` | 受け取った人の判断（action、回した相手、返事、理由、届いた経路） |
| `DUPLICATE` | すでに受け取った人に、別の経路からも届いた |
| `PARSE_ERROR` | モデルの返事が読めなかった |
| `SUMMARY` | 届いた人数、返事の数、最大ホップ数、API の使用量など |

## 注意

- **料金**：API の呼び出しごとに料金がかかります。`--max-calls` で上限を決めてから実行してください。まずは安いモデルと `--mock` で試すのがおすすめです。
- **API キー**：`.env` は `.gitignore` に入っていて、GitHub には上がりません。キーをコードに直接書かないでください。
- **現実との差**：AI の判断は実際の人の行動と同じではありません。研究に使う場合は、実際の行動ログと比べて、どれだけ似ているかを確かめてください。
- エージェントには外部に連絡する道具（メール送信など）を持たせていません。会話と判断だけを行います。

## ライセンス

MIT
