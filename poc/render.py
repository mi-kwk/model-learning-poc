"""画面の描画（段階 2 では「カメラ画像」の代わりになる）。

app.py の画面と、見るモデルに渡す画像の両方が、この同じ描き方を使う。
見るモデルに渡す画像には文字やマウスの十字は描かず、作業領域だけを
IMG_SIZE×IMG_SIZE ピクセルに縮めたものを使う。
"""
from __future__ import annotations

import numpy as np
import pygame

from .env import BOX_SIZE, FIELD_H, FIELD_W, TARGET_SIZE, BoxEnv

IMG_SIZE = 128  # 見るモデルに渡す画像の一辺（600×600 を 128×128 に縮める）

C_FIELD = (255, 255, 255)
C_BOX = (220, 90, 60)        # 置いてある箱
C_BOX_HELD = (245, 180, 40)  # つかまれている箱（色で「つかみ」を見分けられるようにする）
C_TARGET = (60, 170, 110)
C_HAND = (40, 90, 200)
CURSOR_R = 9


def draw_scene(surface: pygame.Surface, env: BoxEnv) -> None:
    """作業領域（置き場・箱・カーソル）を描く。"""
    pygame.draw.rect(surface, C_FIELD, (0, 0, FIELD_W, FIELD_H))
    t = TARGET_SIZE
    tx, ty = env.target
    pygame.draw.rect(surface, C_TARGET, (tx - t / 2, ty - t / 2, t, t), 6, border_radius=6)
    b = BOX_SIZE
    bx, by = env.box
    pygame.draw.rect(surface, C_BOX_HELD if env.holding else C_BOX,
                     (bx - b / 2, by - b / 2, b, b), border_radius=4)
    cx, cy = env.cursor
    pygame.draw.circle(surface, C_HAND, (int(cx), int(cy)), CURSOR_R)


_canvas: pygame.Surface | None = None


def render_image(env: BoxEnv) -> np.ndarray:
    """見るモデルへの入力画像（高さ×幅×3 色、0〜255 の整数）を作る。"""
    global _canvas
    if _canvas is None:
        _canvas = pygame.Surface((FIELD_W, FIELD_H))
    draw_scene(_canvas, env)
    small = pygame.transform.smoothscale(_canvas, (IMG_SIZE, IMG_SIZE))
    return pygame.surfarray.array3d(small).transpose(1, 0, 2).copy()  # (幅,高さ) → (高さ,幅)
