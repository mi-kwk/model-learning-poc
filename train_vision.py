"""段階 2：見るモデルを学習する（画像 → 7 つの数値）。

使い方:
  python train_vision.py                 # 画像 8000 枚を自動で作って学習し models/vision.pt に保存
  python train_vision.py --n 2000        # 画像の枚数を変える（少ないと位置のずれが大きくなる）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from poc.render import IMG_SIZE
from poc.vision import ROOT, VISION_MODEL, make_dataset, save_vision, to_tensor, train_vision

RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data" / "vision"


def jp_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ["Yu Gothic", "Meiryo", "MS Gothic"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    return plt


def plot_vision_history(history: dict, path: Path, title: str) -> None:
    plt = jp_matplotlib()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=100)
    ep = range(1, len(history["train_loss"]) + 1)
    a1.plot(ep, history["train_loss"], label="学習データの誤差")
    a1.plot(ep, history["val_loss"], label="検証データの誤差")
    a1.set_yscale("log")
    a1.set_xlabel("エポック")
    a1.set_ylabel("誤差（対数目盛）")
    a1.set_title(title)
    a1.legend()
    a1.grid(alpha=0.3)
    for key, name in [("cursor_px", "カーソル"), ("box_px", "箱"), ("target_px", "置き場")]:
        a2.plot(ep, [e[key] for e in history["val_err"]], label=name)
    a2.set_xlabel("エポック")
    a2.set_ylabel("位置のずれ（ピクセル、600×600 の画面で）")
    a2.set_title("検証画像での位置のずれ")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_examples(model, images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """検証画像 8 枚に、正解（○）と見るモデルの推定（×）を重ねて保存する。"""
    plt = jp_matplotlib()
    with torch.no_grad():
        out = model(to_tensor(images[:8])).numpy()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4), dpi=100)
    to_px = lambda v: (v + 1) / 2 * IMG_SIZE  # noqa: E731
    for ax, img, y, o in zip(axes.flat, images[:8], labels[:8], out):
        ax.imshow(img)
        for k, c in enumerate(["#2850c8", "#d04020", "#20a060"]):
            ax.scatter(to_px(y[2 * k]), to_px(y[2 * k + 1]), s=80, facecolors="none", edgecolors=c, linewidths=1.5)
            ax.scatter(to_px(o[2 * k]), to_px(o[2 * k + 1]), s=40, marker="x", c="black", linewidths=1.5)
        ax.set_title(f"つかみ 正解 {int(y[6])} / 推定 {int(o[6] > 0)}", fontsize=9)
        ax.axis("off")
    fig.suptitle("○ = 正解（青:カーソル 赤:箱 緑:置き場）  × = 見るモデルの推定")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="見るモデルを学習する")
    ap.add_argument("--n", type=int, default=8000, help="作る画像の枚数")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=Path, default=VISION_MODEL)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cache = DATA_DIR / f"auto_v2_{args.n}.npz"  # v2: 画面の端の場面を追加した版
    t0 = time.time()
    if cache.exists():
        d = np.load(cache)
        images, labels = d["images"], d["labels"]
        print(f"画像 {len(images)} 枚を読み込みました（{cache}）")
    else:
        print(f"画像 {args.n} 枚と正解ラベルを自動で作ります…")
        images, labels = make_dataset(args.n, on_progress=lambda i: print(f"  {i} 枚", flush=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, images=images, labels=labels)
        print(f"  作成 {time.time() - t0:.0f} 秒 → {cache}")

    print(f"\n学習します（{args.epochs} エポック、CPU）")

    def on_epoch(ep, tr, va, err):
        print(f"  エポック {ep:3d}  学習誤差 {tr:.5f}  検証誤差 {va:.5f}  "
              f"ずれ: カーソル {err['cursor_px']:.1f}px 箱 {err['box_px']:.1f}px 置き場 {err['target_px']:.1f}px  "
              f"つかみ正解率 {err['hold_acc'] * 100:.1f}%  （{time.time() - t1:.0f} 秒）", flush=True)

    t1 = time.time()
    model, info = train_vision(images, labels, epochs=args.epochs, on_epoch=on_epoch)

    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_vision"
    plot_vision_history(info["history"], run_dir / "loss.png", f"見るモデル（画像 {len(images)} 枚）")
    n_val = info["num_val_images"]
    plot_examples(model, images[:n_val], labels[:n_val], run_dir / "examples.png")
    meta = {k: v for k, v in info.items() if k != "history"}
    meta["loss_plot"] = str(run_dir / "loss.png")
    save_vision(model, args.out, meta)
    (run_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    e = info["final_val_err"]
    print(f"\n重みの数: {info['num_parameters']:,}（段階 1 の動くモデルは 5,322）")
    print(f"学習 {info['num_train_images']} 枚 / 検証 {n_val} 枚")
    print(f"検証画像での位置のずれ（平均）: カーソル {e['cursor_px']:.1f}px / 箱 {e['box_px']:.1f}px / "
          f"置き場 {e['target_px']:.1f}px、つかみ正解率 {e['hold_acc'] * 100:.1f}%")
    print("  参考: 箱は 32px、つかめる範囲は箱の中心から ±20px、置き場に入る許容は ±24px")
    print(f"モデル: {args.out}")
    print(f"誤差グラフ: {run_dir / 'loss.png'}")
    print(f"推定の例: {run_dir / 'examples.png'}")


if __name__ == "__main__":
    main()
