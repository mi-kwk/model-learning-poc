"""段階 2：人が画像に印を付けるアノテーション画面。

見るモデルに渡すのと同じ 128×128 の画像を 4 倍に拡大して表示します。
画像ごとに「カーソル → 箱 → 置き場」の順に中心をクリックし、
箱をつかんでいる画像なら H を押してから Enter で保存します。

人の印は、正解（プログラムが知っている正確な位置）と比べて
「人の印にもどれくらいずれがあるか」「見るモデルと人はどちらが正確か」を確かめるのに使います。

キー: クリック 印を付ける / H つかみの有無 / Backspace 1 つ戻す / Enter 保存して次へ
      Esc 終了して集計を表示（もう一度 Esc で閉じる）
"""
from __future__ import annotations

import json
import random

import numpy as np
import pygame

from poc.env import FIELD_H, FIELD_W, BoxEnv
from poc.render import IMG_SIZE, render_image
from poc.vision import ROOT, VISION_MODEL, load_vision, predict_state, random_scene, state_error_px

LABEL_FILE = ROOT / "data" / "vision" / "human_labels.json"
SCALE = 4
VIEW = IMG_SIZE * SCALE                 # 512
PANEL_W = 380
WIN_W, WIN_H = VIEW + 40 + PANEL_W, VIEW + 40
SEED_BASE = 5_000_000                   # 学習用画像とも評価用配置とも重ならない番号帯
NAMES = ["カーソル（青い丸）", "箱（赤／黄の四角）", "置き場（緑の枠）"]
MARK_COLORS = [(40, 90, 200), (200, 60, 30), (30, 150, 90)]


def jp_font(size: int) -> pygame.font.Font:
    for name in ["yugothic", "meiryo", "msgothic"]:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def load_labels() -> list[dict]:
    if LABEL_FILE.exists():
        return json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    return []


def save_labels(labels: list[dict]) -> None:
    LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    LABEL_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")


def summarize(labels: list[dict]) -> list[str]:
    """人の印と見るモデルの推定を、正解と比べて集計する。"""
    if not labels:
        return ["まだ印がありません。"]
    human = np.array([state_error_px(d["human"], d["true"]) for d in labels])
    human_hold = sum(d["human"][6] == d["true"][6] for d in labels)
    lines = [f"印を付けた画像: {len(labels)} 枚", "",
             "正解とのずれ（600×600 の画面で何ピクセルか、平均）",
             "人の印:",
             f"  カーソル {human[:, 0].mean():.1f} / 箱 {human[:, 1].mean():.1f} / 置き場 {human[:, 2].mean():.1f}",
             f"  つかみの正解 {human_hold}/{len(labels)}"]
    if VISION_MODEL.exists():
        model, _ = load_vision()
        env = BoxEnv()
        errs, hold_ok = [], 0
        for d in labels:
            env.set_from_state(d["true"])
            est = predict_state(model, render_image(env))
            errs.append(state_error_px(est, d["true"]))
            hold_ok += est[6] == d["true"][6]
        e = np.array(errs)
        lines += ["見るモデル:",
                  f"  カーソル {e[:, 0].mean():.1f} / 箱 {e[:, 1].mean():.1f} / 置き場 {e[:, 2].mean():.1f}",
                  f"  つかみの正解 {hold_ok}/{len(labels)}"]
    else:
        lines.append("（vision.pt がないので見るモデルとは比べていません）")
    lines += ["", "参考: 128×128 の画像の 1 ピクセルは、", "600×600 の画面の約 4.7 ピクセル"]
    return lines


class Annotator:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("アノテーション：画像に印を付ける")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.f_s, self.f_m = jp_font(14), jp_font(18)
        self.labels = load_labels()
        self.env = BoxEnv()
        self.summary: list[str] | None = None
        self.next_image()

    def next_image(self) -> None:
        idx = len(self.labels)
        random_scene(self.env, random.Random(SEED_BASE + idx))
        self.true = self.env.state()
        self.image = render_image(self.env)
        self.surface = pygame.transform.scale(
            pygame.surfarray.make_surface(self.image.transpose(1, 0, 2)), (VIEW, VIEW))
        self.clicks: list[tuple[int, int]] = []
        self.holding = False

    def click_to_state(self, px: int, py: int) -> tuple[float, float]:
        """表示上のクリック位置 → -1〜1 の座標。"""
        x = (px - 20) / VIEW * FIELD_W
        y = (py - 20) / VIEW * FIELD_H
        return x / FIELD_W * 2 - 1, y / FIELD_H * 2 - 1

    def save_current(self) -> None:
        if len(self.clicks) < 3:
            return
        human = []
        for c in self.clicks:
            human += list(self.click_to_state(*c))
        human.append(1.0 if self.holding else 0.0)
        self.labels.append({"seed": SEED_BASE + len(self.labels), "human": human, "true": self.true})
        save_labels(self.labels)
        self.next_image()

    def draw(self) -> None:
        self.screen.fill((32, 36, 44))
        self.screen.blit(self.surface, (20, 20))
        for i, (cx, cy) in enumerate(self.clicks):
            c = MARK_COLORS[i]
            pygame.draw.line(self.screen, (0, 0, 0), (cx - 8, cy - 8), (cx + 8, cy + 8), 3)
            pygame.draw.line(self.screen, (0, 0, 0), (cx - 8, cy + 8), (cx + 8, cy - 8), 3)
            pygame.draw.circle(self.screen, c, (cx, cy), 12, 2)
        x0, y = VIEW + 40, 20

        def t(s, font=None, color=(235, 238, 242)):
            nonlocal y
            font = font or self.f_s
            self.screen.blit(font.render(s, True, color), (x0, y))
            y += font.get_linesize() + 2

        if self.summary is not None:
            t("集計", self.f_m)
            for line in self.summary:
                t(line, color=(250, 190, 60) if line.startswith("  ") else (235, 238, 242))
            t("")
            t("Esc で閉じる", color=(150, 158, 170))
            return
        t(f"画像 {len(self.labels) + 1} 枚目（保存済み {len(self.labels)} 枚）", self.f_m)
        t("これは見るモデルに渡す 128×128 の画像を", color=(150, 158, 170))
        t("4 倍に拡大したものです。", color=(150, 158, 170))
        t("")
        for i, name in enumerate(NAMES):
            mark = "済" if i < len(self.clicks) else ("← 次にクリック" if i == len(self.clicks) else "")
            t(f"{i + 1}. {name} の中心  {mark}",
              color=(250, 190, 60) if i == len(self.clicks) else (235, 238, 242))
        t("")
        t(f"つかんでいる（H で切替）: {'はい' if self.holding else 'いいえ'}")
        t("  箱が黄色ならつかんでいる、赤なら置いてある", color=(150, 158, 170))
        t("")
        t("Enter 保存して次へ   Backspace 1 つ戻す", color=(150, 158, 170))
        t("Esc 終了して集計を表示", color=(150, 158, 170))

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        if self.summary is None:
                            self.summary = summarize(self.labels)
                            for line in self.summary:
                                print(line)
                        else:
                            running = False
                    elif self.summary is not None:
                        continue
                    elif ev.key == pygame.K_h:
                        self.holding = not self.holding
                    elif ev.key == pygame.K_BACKSPACE and self.clicks:
                        self.clicks.pop()
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.save_current()
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.summary is None:
                    x, y = ev.pos
                    if 20 <= x < 20 + VIEW and 20 <= y < 20 + VIEW and len(self.clicks) < 3:
                        self.clicks.append((x, y))
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    Annotator().run()
