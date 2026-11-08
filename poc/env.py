"""箱運びの環境（ロボットで言う「作業台と箱」）。

pygame に依存しない純粋な計算だけを置く。
画面表示・手本記録・自動実行・評価のすべてが、この同じルールで世界を進める。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

# ---- 世界の大きさとルール -------------------------------------------------
FIELD_W = 600          # 作業領域の幅（ピクセル）
FIELD_H = 600          # 作業領域の高さ
STEP = 6.0             # 1 コマの移動量（ピクセル）
BOX_SIZE = 32          # 箱の一辺
TARGET_SIZE = 80       # 置き場の一辺
GRAB_MARGIN = 4        # 箱の外周から何ピクセルまでならつかめるか
MARGIN = 50            # 配置するときの画面端からの余白
MIN_BOX_TARGET_DIST = 150  # 箱と置き場を最低これだけ離して置く
MAX_STEPS = 400        # 自動実行で打ち切る手数

# ---- 行動（出力の 10 クラス） ---------------------------------------------
# 0〜7 は 8 方向の移動。画面座標は y が下向きなので「上」は dy = -1。
DIRECTIONS = [
    (1, 0),    # 0: 右
    (1, -1),   # 1: 右上
    (0, -1),   # 2: 上
    (-1, -1),  # 3: 左上
    (-1, 0),   # 4: 左
    (-1, 1),   # 5: 左下
    (0, 1),    # 6: 下
    (1, 1),    # 7: 右下
]
GRAB = 8
RELEASE = 9
NUM_ACTIONS = 10
ACTION_NAMES = ["右", "右上", "上", "左上", "左", "左下", "下", "右下", "つかむ", "離す"]

STATE_DIM = 7
STATE_NAMES = ["カーソルx", "カーソルy", "箱x", "箱y", "置き場x", "置き場y", "つかみ"]


def nearest_direction(dx: float, dy: float) -> int:
    """マウスの動き (dx, dy) を、最も近い 8 方向の番号に置き換える。"""
    angle = math.atan2(-dy, dx)  # 画面座標の y を反転して数学の角度にする
    return int(round(angle / (math.pi / 4))) % 8


@dataclass
class StepResult:
    done: bool
    success: bool


class BoxEnv:
    """箱 1 個・置き場 1 個の環境。"""

    def __init__(self) -> None:
        self.cursor = [0.0, 0.0]
        self.box = [0.0, 0.0]
        self.target = [0.0, 0.0]
        self.holding = False
        self.steps = 0
        self.done = False
        self.success = False
        self.seed = None

    # ---- 配置 --------------------------------------------------------------
    def reset(self, seed: int | None = None) -> list[float]:
        """配置をランダムに決め直す。seed を決めれば同じ配置を再現できる。"""
        self.seed = seed
        rng = random.Random(seed)

        def rand_point():
            return [rng.uniform(MARGIN, FIELD_W - MARGIN), rng.uniform(MARGIN, FIELD_H - MARGIN)]

        while True:
            self.box = rand_point()
            self.target = rand_point()
            if math.dist(self.box, self.target) >= MIN_BOX_TARGET_DIST:
                break
        self.cursor = rand_point()
        self.holding = False
        self.steps = 0
        self.done = False
        self.success = False
        return self.state()

    # ---- 状態（入力の 7 つの数値） -----------------------------------------
    def state(self) -> list[float]:
        """座標を -1〜1 に正規化した 7 つの数値を返す。"""
        def nx(x):
            return x / FIELD_W * 2 - 1

        def ny(y):
            return y / FIELD_H * 2 - 1

        return [
            nx(self.cursor[0]), ny(self.cursor[1]),
            nx(self.box[0]), ny(self.box[1]),
            nx(self.target[0]), ny(self.target[1]),
            1.0 if self.holding else 0.0,
        ]

    def set_from_state(self, s: list[float]) -> None:
        """記録された状態から画面を復元する（手本の再生に使う）。"""
        def dx(v):
            return (v + 1) / 2 * FIELD_W

        def dy(v):
            return (v + 1) / 2 * FIELD_H

        self.cursor = [dx(s[0]), dy(s[1])]
        self.box = [dx(s[2]), dy(s[3])]
        self.target = [dx(s[4]), dy(s[5])]
        self.holding = s[6] > 0.5

    # ---- 判定 --------------------------------------------------------------
    def can_grab(self) -> bool:
        half = BOX_SIZE / 2 + GRAB_MARGIN
        return (abs(self.cursor[0] - self.box[0]) <= half
                and abs(self.cursor[1] - self.box[1]) <= half)

    def box_in_target(self) -> bool:
        tol = (TARGET_SIZE - BOX_SIZE) / 2  # 箱が置き場からはみ出さない範囲
        return (abs(self.box[0] - self.target[0]) <= tol
                and abs(self.box[1] - self.target[1]) <= tol)

    # ---- 1 コマ進める ------------------------------------------------------
    def step(self, action: int) -> StepResult:
        """行動を 1 つ実行して世界を 1 コマ進める。"""
        if self.done:
            return StepResult(True, self.success)
        self.steps += 1

        if action < 8:
            ux, uy = DIRECTIONS[action]
            norm = math.hypot(ux, uy)
            self.cursor[0] = min(max(self.cursor[0] + ux / norm * STEP, 0), FIELD_W)
            self.cursor[1] = min(max(self.cursor[1] + uy / norm * STEP, 0), FIELD_H)
            if self.holding:  # つかんでいる間は箱もカーソルと一緒に動く
                self.box = list(self.cursor)
        elif action == GRAB:
            if not self.holding and self.can_grab():
                self.holding = True
                self.box = list(self.cursor)
            # 箱から離れた所で「つかむ」を選んでも何も起きない
        elif action == RELEASE:
            if self.holding:
                self.holding = False
                if self.box_in_target():
                    self.done = True
                    self.success = True
            # つかんでいないのに「離す」を選んでも何も起きない

        return StepResult(self.done, self.success)
