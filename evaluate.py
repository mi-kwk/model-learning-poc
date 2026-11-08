"""学習に使っていない配置で 20 回ずつ実行し、成功率と手数を比べる。

使い方:
  python evaluate.py                    # models/model.pt とランダム行動を比べる
  python evaluate.py --compare          # models/model_n*.pt（手本 5/20/50 回など）とランダム行動を比べる
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from poc.policy import DEFAULT_MODEL, MODEL_DIR, RUNS_DIR, load_model
from poc.runner import EVAL_EPISODES, evaluate, model_policy, random_policy


def plot_summaries(summaries, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for name in ["Yu Gothic", "Meiryo", "MS Gothic"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break

    names = [s.name for s in summaries]
    rates = [s.success_rate * 100 for s in summaries]
    steps = [s.mean_steps_success or 0 for s in summaries]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=100)
    a1.bar(names, rates, color="#3b7dd8")
    a1.set_ylim(0, 105)
    a1.set_ylabel("成功率（%）")
    a1.set_title(f"未学習の配置 {EVAL_EPISODES} 回での成功率")
    for i, r in enumerate(rates):
        a1.text(i, r + 2, f"{r:.0f}%", ha="center")
    a2.bar(names, steps, color="#e08a2c")
    a2.set_ylabel("手数")
    a2.set_title("成功したときの平均手数（少ないほど無駄が少ない）")
    for i, s in enumerate(steps):
        a2.text(i, s + 2, f"{s:.0f}" if s else "成功なし", ha="center")
    for ax in (a1, a2):
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="モデルを未学習の配置で評価する")
    ap.add_argument("--compare", action="store_true", help="models/model_n*.pt をまとめて比べる")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="--compare で探すフォルダ")
    ap.add_argument("--greedy", action="store_true",
                    help="くじ引きではなく、確率が最大の行動を必ず選ぶ（比較用）")
    args = ap.parse_args()

    if args.compare:
        paths = sorted(args.model_dir.glob("model_n*.pt"),
                       key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        if not paths:
            print("models/model_n*.pt がありません。先に python train.py --compare 5 20 50 を実行してください。")
            return
    else:
        if not args.model.exists():
            print(f"{args.model} がありません。先に python train.py を実行してください。")
            return
        paths = [args.model]

    summaries = [evaluate("ランダム行動", random_policy(seed=0))]
    for p in paths:
        model, meta = load_model(p)  # 保存したファイルから読み込み直して使う
        n = meta.get("num_demos", "?")
        summaries.append(evaluate(f"手本{n}回", model_policy(model, sample=not args.greedy)))

    how = "確率が最大の行動を選ぶ" if args.greedy else "確率に従ってくじ引き（乱数の種は固定）"
    print(f"\n評価: 学習に使っていない配置 {EVAL_EPISODES} 回（どのモデルも同じ配置）")
    print(f"行動の選び方: {how}")
    for s in summaries:
        print("  " + s.line())

    name = ("compare" if args.compare else "evaluate") + ("_greedy" if args.greedy else "")
    out = RUNS_DIR / f"{name}.png"
    plot_summaries(summaries, out)
    print(f"\nグラフ: {out}")


if __name__ == "__main__":
    main()
