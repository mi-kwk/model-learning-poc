# サンプルデータ

`demos/` は、`tools/make_synthetic_demos.py` で機械の手本役が作った **合成の手本 5 回分** です（人の手本ではありません）。データの形式を見るためのものです。

- 1 ファイル = 手本 1 回。`frames` に 1 コマごとの `state`（7 つの数値）と `action`（0〜9）が入っています。
- `result`（success / failure）、`excluded`（除外の印）、`note`（メモ）、`source`（synthetic = 合成）も入っています。

このデータで学習を試す場合:

```bash
.venv\Scripts\python train.py --data-dir samples/demos --model-dir models_synthetic
```
