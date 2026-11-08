# 模倣学習 PoC（段階 1）

画面の中で、人がマウスで箱を置き場へ運ぶ手本を記録し、その「状態 → 行動」を小さなニューラルネットに学習させて、未学習の配置で自動実行・評価する個人 PoC です。

- 説明は [docs/説明メモ.md](docs/説明メモ.md)、判断の記録は [docs/判断ログ.md](docs/判断ログ.md) にあります。
- 環境：Windows 11、Python 3.12、PyTorch（CPU 版）、pygame、matplotlib、numpy

## セットアップ（済んでいれば不要）

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

## 操作手順

### 1. 手本を記録する

```bash
.venv\Scripts\python app.py
```

- 起動すると手本モードになります。赤い箱にマウスを合わせ、左ボタンを押したまま緑の置き場まで運んで離してください。
- 青い丸が「記録されるカーソル」です。マウス（灰色の十字）を 1 コマ 1 歩ずつ、8 方向のどれかで追いかけます。速く動かすと青い丸が遅れて付いてくるので、ゆっくりめに動かすのがコツです。
- 置き場に入れて離すと、自動で「成功」として保存され、次の配置になります。
- うまくいかなかったときは `F` で「失敗」として保存（学習には使われません）、`Space` で保存せずに配置をやり直します。
- 直前の手本が雑だったときは `X` で除外の印を付けます（もう一度押すと外れます）。`P` で直前の手本を再生して確認できます。
- 比較のため、成功した手本を **50 回** を目安に記録してください（1 回 5〜10 秒ほど）。

### 2. 学習する

画面で `T` を押すと、学習に使える手本（成功・除外なし）すべてで学習し、`models/model.pt` に保存して誤差のグラフを表示します。グラフは `G` で再表示できます。

### 3. 自動実行と評価

- `2`：自動実行モード。モデルが毎コマ状態を見て行動を選びます。右側に入力の 7 つの数値と、10 個の行動の確率が表示されます。`+` `-` で速さを変えられます。
- `3`：ランダム行動モード（比較用）。
- `S`：行動の選び方を切り替えます。既定は「確率に従ってくじ引き」で、もう一方は「確率が最大の行動を必ず選ぶ」です。同じモデルでも成功率が変わります（[docs/説明メモ.md](docs/説明メモ.md) の 7）。
- `E`：学習に使っていない配置 20 回を早送りで実行し、成功率と平均手数を表示します。
- `L`：`model.pt` を読み込み直します。

### 4. 手本の数で比べる（コマンド）

```bash
.venv\Scripts\python train.py --compare 5 20 50
```

```bash
.venv\Scripts\python evaluate.py --compare
```

`evaluate.py --compare --greedy` にすると「確率が最大の行動を選ぶ」方式で比べます。`train.py` に `--early-stop` を付けると、検証誤差が最小だったエポックの重みで保存します（比較用。この課題では成功率が下がりました）。

先に記録した 5 回・20 回・50 回分で学習した `models/model_n5.pt` などを作り、ランダム行動と並べて同じ 20 配置で評価します。結果は `runs/compare.png` に、各学習の誤差グラフは `runs/<日時>_model_n*/loss.png` に保存されます。

## ファイル構成

| パス | 内容 |
| --- | --- |
| `app.py` | pygame の画面 |
| `train.py` / `evaluate.py` | コマンドでの学習・評価 |
| `poc/env.py` | 箱運びの環境（ルール・状態・行動） |
| `poc/demos.py` | 手本の保存・読み込み |
| `poc/policy.py` | モデル・学習・保存 |
| `poc/runner.py` | 自動実行（閉ループ）・評価 |
| `tools/make_synthetic_demos.py` | 動作確認専用の合成手本 |
| `data/demos/` | 記録した手本（1 回 1 ファイルの JSON） |
| `models/` | 学習したモデル |
| `runs/` | 誤差グラフ・評価グラフ |

`data/` `models/` `runs/` は git の管理対象外にしています。データ形式の見本として、合成の手本 5 回分だけを `samples/demos/` に入れています（[samples/README.md](samples/README.md)）。手本を残したい場合は `.gitignore` から `data/demos/` を外してください。

## 動作確認用の合成手本（任意）

人の手本の代わりに機械が作る手本で、仕組みだけを先に確かめられます。結果は人の手本と混ざらないよう別フォルダに出します。

```bash
.venv\Scripts\python tools\make_synthetic_demos.py --n 50
```

```bash
.venv\Scripts\python train.py --data-dir data/synthetic --compare 5 20 50 --model-dir models_synthetic
```

```bash
.venv\Scripts\python evaluate.py --compare --model-dir models_synthetic
```
