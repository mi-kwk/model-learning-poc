"""自動実行と評価（ロボットで言う「適用」と「現場での評価」）。

自動実行は「状態 → 行動の予測 → 世界が 1 コマ進む → 新しい状態」の繰り返し。
記録した操作を再生しているのではなく、毎コマその場の状態からモデルが行動を選ぶ。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .env import GRAB, MAX_STEPS, NUM_ACTIONS, RELEASE, BoxEnv, nearest_direction
from .policy import PolicyNet, predict_probs

Policy = Callable[[list[float]], int]

# 評価用の配置。手本の配置（0〜999,999 の乱数シード）と重ならない番号を使うので、
# 評価は「学習に使っていない配置」で行われる。
EVAL_SEED_BASE = 9_000_000
EVAL_EPISODES = 20


def eval_seeds(n: int = EVAL_EPISODES) -> list[int]:
    return [EVAL_SEED_BASE + i for i in range(n)]


EVAL_RNG_SEED = 0  # くじ引きの乱数の種。固定しておくと、評価は何度やっても同じ結果になる


def choose_action(probs: np.ndarray, rng: np.random.Generator, sample: bool) -> int:
    """モデルが出した 10 個の確率から、実際に行う行動を 1 つ決める。

    sample=True : 確率に従ってくじ引きで選ぶ（右上 70% なら 10 回に 7 回くらい右上）
    sample=False: いちばん確率が高い行動を必ず選ぶ

    必ず最大を選ぶと、同じ状態では同じ行動しか出ないため、
    「何も起きない行動」や「行ったり来たり」に一度はまると抜け出せない。
    くじ引きなら、たまに別の行動が出て抜け出せる。
    """
    if sample:
        p = probs.astype(np.float64)
        return int(rng.choice(len(p), p=p / p.sum()))
    return int(np.argmax(probs))


def model_policy(model: PolicyNet, sample: bool = True, seed: int = EVAL_RNG_SEED) -> Policy:
    """学習したモデルで行動を選ぶ方策（既定はくじ引き）。"""
    rng = np.random.default_rng(seed)

    def act(state: list[float]) -> int:
        return choose_action(predict_probs(model, state), rng, sample)
    return act


def random_policy(seed: int = 0) -> Policy:
    """比較用：10 個の行動から毎回でたらめに選ぶ方策。"""
    rng = random.Random(seed)

    def act(state: list[float]) -> int:
        return rng.randrange(NUM_ACTIONS)
    return act


def scripted_expert(env: BoxEnv) -> int:
    """動作確認用の「機械の手本役」。人の手本の代わりにテストで使うだけで、
    自動実行の方式そのものではない（学習したモデルはこの関数を知らない）。"""
    if not env.holding:
        if env.can_grab() and abs(env.cursor[0] - env.box[0]) < 6 and abs(env.cursor[1] - env.box[1]) < 6:
            return GRAB
        goal = env.box
    else:
        if abs(env.box[0] - env.target[0]) < 5 and abs(env.box[1] - env.target[1]) < 5:
            return RELEASE
        goal = env.target
    return nearest_direction(goal[0] - env.cursor[0], goal[1] - env.cursor[1])


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    steps: int


def run_episode(policy: Policy, seed: int, max_steps: int = MAX_STEPS,
                on_step: Callable[[BoxEnv, int], bool] | None = None) -> EpisodeResult:
    """1 回分を最後まで動かす。on_step を渡すと 1 コマごとに呼ばれる（画面描画用）。"""
    env = BoxEnv()
    state = env.reset(seed)
    while env.steps < max_steps and not env.done:
        action = policy(state)          # 状態 → 行動の予測
        env.step(action)                # 世界が 1 コマ進む
        state = env.state()             # 新しい状態
        if on_step and on_step(env, action) is False:
            break
    return EpisodeResult(seed, env.success, env.steps)


@dataclass
class EvalSummary:
    name: str
    results: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        return sum(r.success for r in self.results) / len(self.results)

    @property
    def mean_steps_success(self) -> float | None:
        s = [r.steps for r in self.results if r.success]
        return sum(s) / len(s) if s else None

    def line(self) -> str:
        ms = self.mean_steps_success
        ms_text = f"{ms:6.1f}" if ms is not None else "   ---"
        return (f"{self.name:<16} 成功 {sum(r.success for r in self.results):2d}/{len(self.results)}"
                f" （成功率 {self.success_rate * 100:5.1f}%） 成功時の平均手数 {ms_text}")


def evaluate(name: str, policy: Policy, n: int = EVAL_EPISODES) -> EvalSummary:
    return EvalSummary(name, [run_episode(policy, s) for s in eval_seeds(n)])
