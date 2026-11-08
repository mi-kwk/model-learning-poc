"""【動作確認専用】機械の手本役で、人の手本の代わりのデータを作る。

人の手本は data/demos/ に、この合成データは data/synthetic/ に分けて置く。
成果として見せる結果には、人が記録した data/demos/ の手本を使うこと。

使い方:
  python tools/make_synthetic_demos.py --n 50
  python train.py --data-dir data/synthetic --compare 5 20 50
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poc.demos import ROOT, Demo, save_demo  # noqa: E402
from poc.env import BoxEnv  # noqa: E402
from poc.runner import scripted_expert  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--noise", type=float, default=0.15, help="人のぶれを真似て、隣の方向へずらす確率")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    rng = random.Random(123)
    for i in range(1, args.n + 1):
        env = BoxEnv()
        seed = rng.randrange(0, 1_000_000)
        env.reset(seed)
        demo = Demo(demo_id=f"demo_{i:04d}", seed=seed, source="synthetic")
        while not env.done and env.steps < 1000:
            a = scripted_expert(env)
            if a < 8 and rng.random() < args.noise:
                a = (a + rng.choice([-1, 1])) % 8
            demo.frames.append({"state": env.state(), "action": a})
            env.step(a)
        demo.result = "success" if env.success else "failure"
        save_demo(demo, args.out)
    print(f"{args.n} 回分の合成手本を {args.out} に作りました。")


if __name__ == "__main__":
    main()
