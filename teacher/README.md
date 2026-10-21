# teacher/

協力の教師データ（実際の行動の記録）を置く場所です。

- `example_synthetic.jsonl` … **合成データ**です。実在の人や実際のログではありません。動作確認用です。
- 実データ（例：`my_real_data.jsonl`）はここに置いてください。`.gitignore` により **GitHub には上がりません**。

形式は `relay/teacher.py` の説明を、CSV からの変換は `tools/csv_to_teacher.py` を見てください。
個人が分かる情報（名前・住所・連絡先）は、必ず消してから使ってください。
