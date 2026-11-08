"""手本から学習してモデルを保存する。

使い方:
  python train.py                      # 使える手本をすべて使い models/model.pt を作る
  python train.py --compare 5 20 50    # 先に記録した 5 回・20 回・50 回分でそれぞれ学習
  python train.py --limit 20           # 先に記録した 20 回分だけで models/model.pt を作る
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from poc.demos import DEFAULT_DEMO_DIR, usable_demos
from poc.policy import DEFAULT_MODEL, MODEL_DIR, RUNS_DIR, plot_history, save_model, train_policy


def train_one(demos, out_path: Path, title: str, epochs: int) -> dict:
    print(f"\n=== {title}: 手本 {len(demos)} 回 ===")

    def on_epoch(ep, tr, va):
        if ep == 1 or ep % 25 == 0 or ep == epochs:
            va_text = f"{va:.4f}" if va is not None else "---"
            print(f"  エポック {ep:4d}  学習誤差 {tr:.4f}  検証誤差 {va_text}")

    model, info = train_policy(demos, epochs=epochs, on_epoch=on_epoch)
    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{out_path.stem}"
    plot_history(info["history"], run_dir / "loss.png", f"{title}（手本 {len(demos)} 回）")
    meta = {k: v for k, v in info.items() if k != "history"}
    meta["loss_plot"] = str(run_dir / "loss.png")
    save_model(model, out_path, meta)
    (run_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  重みの数: {info['num_parameters']}")
    print(f"  学習 {info['num_train_demos']} 回（{info['num_train_frames']} コマ） / "
          f"検証 {info['num_val_demos']} 回（{info['num_val_frames']} コマ）")
    if info["final_val_acc"] is not None:
        print(f"  検証データで手本と同じ行動を選べた割合: {info['final_val_acc'] * 100:.1f}%")
    print(f"  モデル: {out_path}")
    print(f"  誤差グラフ: {run_dir / 'loss.png'}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="手本から方策ネットを学習する")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DEMO_DIR, help="手本の保存先")
    ap.add_argument("--compare", type=int, nargs="*", help="比較用に、先頭から N 回分ずつで学習する")
    ap.add_argument("--limit", type=int, help="先頭から N 回分だけ使う")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="モデルの保存先")
    args = ap.parse_args()

    all_demos = usable_demos(args.data_dir)
    print(f"学習に使える手本（成功・除外なし）: {len(all_demos)} 回  [{args.data_dir}]")
    if not all_demos:
        print("手本がありません。python app.py で手本を記録してください。")
        return

    if args.compare:
        for n in args.compare:
            if len(all_demos) < n:
                print(f"\n手本が {len(all_demos)} 回しかないため、{n} 回のモデルは作りません。")
                continue
            train_one(all_demos[:n], args.model_dir / f"model_n{n}.pt", f"比較用モデル n={n}", args.epochs)
    else:
        demos = all_demos[:args.limit] if args.limit else all_demos
        train_one(demos, args.model_dir / DEFAULT_MODEL.name, "学習", args.epochs)


if __name__ == "__main__":
    main()
