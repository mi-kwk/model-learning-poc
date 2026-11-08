"""模倣学習 PoC の画面（pygame）。

キー操作:
  1 手本モード   2 自動実行モード（model.pt）   3 ランダム行動モード
  Space 新しい配置   F 今の手本を「失敗」として保存   X 直前の手本の除外を切り替え
  P 直前の手本を再生   T 学習（すべての手本 → model.pt）   L model.pt を読み込み直す
  E 評価（未学習の配置で 20 回）   G 最後の誤差グラフを表示
  S 行動の選び方を切り替え（くじ引き ⇔ 確率が最大の行動）   + - 自動実行の速さ   Esc 終了
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pygame

from poc import env as E
from poc.demos import DEFAULT_DEMO_DIR, Demo, load_demos, new_demo_id, save_demo, usable_demos
from poc.policy import DEFAULT_MODEL, RUNS_DIR, load_model, plot_history, predict_probs, save_model, train_policy
from poc.runner import EVAL_EPISODES, EVAL_RNG_SEED, choose_action, eval_seeds

PANEL_W = 400
WIN_W, WIN_H = E.FIELD_W + PANEL_W, E.FIELD_H
FPS = 60

C_BG = (245, 246, 248)
C_FIELD = (255, 255, 255)
C_PANEL = (32, 36, 44)
C_TEXT = (235, 238, 242)
C_DIM = (150, 158, 170)
C_BOX = (220, 90, 60)
C_TARGET = (60, 170, 110)
C_HAND = (40, 90, 200)
C_MOUSE = (170, 170, 170)
C_BAR = (90, 150, 240)
C_BAR_TOP = (250, 190, 60)

MODE_DEMO, MODE_AUTO, MODE_RANDOM, MODE_REPLAY, MODE_EVAL = "demo", "auto", "random", "replay", "eval"
MODE_LABEL = {
    MODE_DEMO: "手本モード（マウスで箱をドラッグ）",
    MODE_AUTO: "自動実行モード（model.pt が操作）",
    MODE_RANDOM: "ランダム行動モード（比較用）",
    MODE_REPLAY: "手本の再生（記録の確認）",
    MODE_EVAL: "評価中（未学習の配置で 20 回）",
}


def jp_font(size: int) -> pygame.font.Font:
    for name in ["yugothic", "meiryo", "msgothic"]:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("模倣学習 PoC：箱を置き場へ運ぶ")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_s, self.f_m, self.f_l = jp_font(14), jp_font(17), jp_font(22)
        self.rng = np.random.default_rng()

        self.env = E.BoxEnv()
        self.mode = MODE_DEMO
        self.demo: Demo | None = None
        self.last_demo: Demo | None = None
        self.message = "手本モードです。箱をドラッグして緑の置き場へ運んでください。"
        self.model = None
        self.model_meta: dict = {}
        self.probs: np.ndarray | None = None
        self.sample = True            # True: 確率に従ってくじ引き / False: 確率が最大の行動
        self.eval_rng = np.random.default_rng(EVAL_RNG_SEED)
        self.last_action: int | None = None
        self.auto_speed = 2           # 自動実行で 1 画面あたり何コマ進めるか
        self.replay_frames: list[dict] = []
        self.replay_i = 0
        self.eval_queue: list[int] = []
        self.eval_results: list[tuple[bool, int]] = []
        self.overlay: pygame.Surface | None = None
        self.paused_until_key = False
        self.end_wait = 0             # 自動実行で 1 回終わってから次の配置に移るまでのコマ数
        self.demo_counts = (0, 0)     # (記録済み, 学習に使える)
        self.refresh_counts()

        self.try_load_model(quiet=True)
        self.new_layout()

    # ---- 準備 --------------------------------------------------------------
    def new_layout(self, seed: int | None = None) -> None:
        if seed is None:
            seed = int(self.rng.integers(0, 1_000_000))  # 手本・練習用の配置（評価用とは別の番号帯）
        self.env.reset(seed)
        self.probs = None
        self.last_action = None
        self.paused_until_key = False
        if self.mode == MODE_DEMO:
            self.demo = Demo(demo_id="", seed=seed)
            # 手本の開始時は、マウスの位置にカーソルを合わせる
            mx, my = pygame.mouse.get_pos()
            if mx < E.FIELD_W:
                self.env.cursor = [float(mx), float(my)]

    def refresh_counts(self) -> None:
        demos = load_demos()
        self.demo_counts = (len(demos), sum(d.usable for d in demos))

    def try_load_model(self, quiet: bool = False) -> None:
        if DEFAULT_MODEL.exists():
            self.model, self.model_meta = load_model(DEFAULT_MODEL)
            if not quiet:
                self.message = f"model.pt を読み込みました（手本 {self.model_meta.get('num_demos', '?')} 回で学習）。"
        else:
            self.model = None
            if not quiet:
                self.message = "model.pt がまだありません。T で学習してください。"

    def set_mode(self, mode: str) -> None:
        if mode == MODE_AUTO and self.model is None:
            self.message = "model.pt がまだありません。手本を記録して T で学習してください。"
            return
        self.mode = mode
        self.message = MODE_LABEL[mode]
        self.new_layout()

    # ---- 手本の記録 --------------------------------------------------------
    def demo_step(self) -> None:
        """マウスの位置とボタンから、このコマの行動を決めて記録する。"""
        if self.env.done or self.demo is None:
            return
        mx, my = pygame.mouse.get_pos()
        mx = min(mx, E.FIELD_W)
        pressed = pygame.mouse.get_pressed()[0]

        action = None
        if pressed and not self.env.holding and self.env.can_grab():
            action = E.GRAB
        elif not pressed and self.env.holding:
            action = E.RELEASE
        else:
            dx, dy = mx - self.env.cursor[0], my - self.env.cursor[1]
            if (dx * dx + dy * dy) ** 0.5 > E.STEP * 0.75:
                action = E.nearest_direction(dx, dy)   # 最も近い 8 方向に置き換える
        if action is None:
            return  # マウスが止まっているコマは記録しない

        self.demo.frames.append({"state": self.env.state(), "action": action})  # 状態と行動の組
        self.last_action = action
        res = self.env.step(action)
        if res.success:
            self.finish_demo("success")

    def finish_demo(self, result: str) -> None:
        if self.demo is None or not self.demo.frames:
            self.message = "記録がないので保存しませんでした。"
            self.new_layout()
            return
        self.demo.demo_id = new_demo_id()
        self.demo.result = result
        save_demo(self.demo)
        self.last_demo = self.demo
        self.refresh_counts()
        n_ok = self.demo_counts[1]
        label = "成功" if result == "success" else "失敗"
        self.message = (f"{self.demo.demo_id} を「{label}」として保存（{len(self.demo.frames)} コマ）。"
                        f"学習に使える手本 {n_ok} 回。")
        self.new_layout()

    def toggle_exclude(self) -> None:
        if self.last_demo is None:
            demos = load_demos()
            if not demos:
                self.message = "除外できる手本がありません。"
                return
            self.last_demo = demos[-1]
        d = self.last_demo
        d.excluded = not d.excluded
        d.note = "画面から手動で除外" if d.excluded else ""
        save_demo(d)
        self.refresh_counts()
        self.message = f"{d.demo_id} の除外を{'付けました' if d.excluded else '外しました'}。"

    def start_replay(self) -> None:
        demos = load_demos()
        d = self.last_demo or (demos[-1] if demos else None)
        if d is None:
            self.message = "再生できる手本がありません。"
            return
        self.mode = MODE_REPLAY
        self.replay_frames = d.frames
        self.replay_i = 0
        self.message = f"{d.demo_id} を再生中（{d.result}{'・除外' if d.excluded else ''}）。記録された状態を順に表示しています。"

    # ---- 自動実行（閉ループ） ----------------------------------------------
    def auto_step(self) -> None:
        if self.env.done or self.env.steps >= E.MAX_STEPS:
            return
        state = self.env.state()                                  # 1) 今の状態を読む
        if self.mode == MODE_RANDOM:
            self.probs = np.full(E.NUM_ACTIONS, 1 / E.NUM_ACTIONS)
            action = int(self.rng.integers(E.NUM_ACTIONS))
        else:
            self.probs = predict_probs(self.model, state)         # 2) モデルが行動を予測する
            action = choose_action(self.probs, self.rng, self.sample)  #    確率から行動を 1 つ決める
        self.last_action = action
        self.env.step(action)                                     # 3) 画面（世界）が変わる
        # 4) 次のコマでまた新しい状態を読む

    # ---- 学習 --------------------------------------------------------------
    def train(self) -> None:
        demos = usable_demos()
        if not demos:
            self.message = "学習に使える手本（成功・除外なし）がありません。"
            return
        epochs = 400

        def on_epoch(ep, tr, va):
            pygame.event.pump()  # 学習中も画面が固まらないようにする
            if ep % 5 == 0 or ep == epochs:
                self.draw_progress(f"学習中… エポック {ep}/{epochs}", ep / epochs,
                                   f"学習誤差 {tr:.3f}" + (f" / 検証誤差 {va:.3f}" if va is not None else ""))

        model, info = train_policy(demos, epochs=epochs, on_epoch=on_epoch)
        import time
        run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_model"
        plot_path = run_dir / "loss.png"
        plot_history(info["history"], plot_path, f"学習（手本 {len(demos)} 回）", info["used_epoch"])
        meta = {k: v for k, v in info.items() if k != "history"}
        meta["loss_plot"] = str(plot_path)
        save_model(model, DEFAULT_MODEL, meta)
        self.try_load_model(quiet=True)   # 保存したファイルから読み込み直して使う
        acc = info["used_val_acc"]
        self.message = (f"学習完了：手本 {len(demos)} 回・重み {info['num_parameters']} 個を調整。"
                        f"エポック {info['used_epoch']} の重みを使用。"
                        + (f"検証の一致率 {acc * 100:.0f}%。" if acc is not None else "")
                        + " 2 で自動実行、E で評価。")
        self.show_image(plot_path)

    # ---- 評価 --------------------------------------------------------------
    def start_eval(self) -> None:
        if self.model is None:
            self.message = "model.pt がまだありません。T で学習してください。"
            return
        self.mode = MODE_EVAL
        self.eval_queue = eval_seeds(EVAL_EPISODES)
        self.eval_results = []
        self.eval_rng = np.random.default_rng(EVAL_RNG_SEED)  # evaluate.py と同じ乱数の種で始める
        self.env.reset(self.eval_queue.pop(0))

    def eval_step(self) -> None:
        for _ in range(6):  # 評価は早送りで進める
            if self.env.done or self.env.steps >= E.MAX_STEPS:
                self.eval_results.append((self.env.success, self.env.steps))
                if not self.eval_queue:
                    self.finish_eval()
                    return
                self.env.reset(self.eval_queue.pop(0))
            state = self.env.state()
            self.probs = predict_probs(self.model, state)
            self.last_action = choose_action(self.probs, self.eval_rng, self.sample)
            self.env.step(self.last_action)

    def finish_eval(self) -> None:
        ok = [s for s, _ in self.eval_results if s]
        steps = [n for s, n in self.eval_results if s]
        mean = f"{sum(steps) / len(steps):.0f}" if steps else "---"
        how = "くじ引き" if self.sample else "確率が最大"
        self.message = (f"評価結果（{how}）：成功 {len(ok)}/{len(self.eval_results)}（{len(ok) / len(self.eval_results) * 100:.0f}%）"
                        f"・成功時の平均手数 {mean}。ランダム行動との比較は python evaluate.py で。")
        self.mode = MODE_AUTO
        self.paused_until_key = True

    # ---- 描画 --------------------------------------------------------------
    def show_image(self, path: Path) -> None:
        if path and Path(path).exists():
            self.overlay = pygame.image.load(str(path)).convert()

    def text(self, s: str, pos, font=None, color=C_TEXT) -> int:
        font = font or self.f_m
        self.screen.blit(font.render(s, True, color), pos)
        return pos[1] + font.get_linesize()

    def wrap(self, s: str, pos, width: int, font=None, color=C_TEXT) -> int:
        font = font or self.f_s
        x, y = pos
        line = ""
        for ch in s:
            if font.size(line + ch)[0] > width:
                y = self.text(line, (x, y), font, color)
                line = ch
            else:
                line += ch
        if line:
            y = self.text(line, (x, y), font, color)
        return y

    def draw_field(self) -> None:
        pygame.draw.rect(self.screen, C_FIELD, (0, 0, E.FIELD_W, E.FIELD_H))
        t = E.TARGET_SIZE
        tx, ty = self.env.target
        pygame.draw.rect(self.screen, C_TARGET, (tx - t / 2, ty - t / 2, t, t), 4, border_radius=6)
        self.text("置き場", (tx - 22, ty - t / 2 - 22), self.f_s, C_TARGET)
        b = E.BOX_SIZE
        bx, by = self.env.box
        pygame.draw.rect(self.screen, C_BOX, (bx - b / 2, by - b / 2, b, b), border_radius=4)
        cx, cy = self.env.cursor
        pygame.draw.circle(self.screen, C_HAND, (int(cx), int(cy)), 9 if self.env.holding else 7,
                           0 if self.env.holding else 3)
        if self.mode == MODE_DEMO:
            mx, my = pygame.mouse.get_pos()
            if mx < E.FIELD_W:
                pygame.draw.line(self.screen, C_MOUSE, (mx - 8, my), (mx + 8, my))
                pygame.draw.line(self.screen, C_MOUSE, (mx, my - 8), (mx, my + 8))
        if self.env.success:
            self.text("成功！", (E.FIELD_W / 2 - 30, 20), self.f_l, C_TARGET)
        elif self.env.steps >= E.MAX_STEPS:
            self.text(f"{E.MAX_STEPS} 手で打ち切り（失敗）", (E.FIELD_W / 2 - 100, 20), self.f_l, C_BOX)

    def draw_panel(self) -> None:
        x0 = E.FIELD_W + 16
        w = PANEL_W - 32
        pygame.draw.rect(self.screen, C_PANEL, (E.FIELD_W, 0, PANEL_W, WIN_H))
        y = self.text(MODE_LABEL[self.mode], (x0, 12), self.f_l)
        if self.mode == MODE_DEMO:
            total, ok = self.demo_counts
            y = self.text(f"記録済み {total} 回（学習に使える {ok} 回）", (x0, y + 2), self.f_s, C_DIM)
            if self.demo is not None:
                y = self.text(f"この手本の記録: {len(self.demo.frames)} コマ", (x0, y), self.f_s, C_DIM)
        elif self.mode == MODE_EVAL:
            ok = sum(s for s, _ in self.eval_results)
            y = self.text(f"{len(self.eval_results) + 1}/{EVAL_EPISODES} 回目 / これまでの成功 {ok}", (x0, y + 2), self.f_s, C_DIM)
        else:
            y = self.text(f"手数 {self.env.steps}", (x0, y + 2), self.f_s, C_DIM)

        # 入力（状態）
        y += 8
        y = self.text("入力：状態（7 つの数値、-1〜1 に正規化）", (x0, y), self.f_m)
        s = self.env.state()
        for i, name in enumerate(E.STATE_NAMES):
            col = i % 2
            yy = y + (i // 2) * 18
            self.text(f"{name}: {s[i]:+.2f}", (x0 + col * 180, yy), self.f_s, C_DIM)
        y += 4 * 18 + 6

        # 出力（行動）
        y = self.text("出力：10 個の行動の確率", (x0, y), self.f_m)
        probs = self.probs
        for i, name in enumerate(E.ACTION_NAMES):
            yy = y + i * 19
            self.text(name, (x0, yy), self.f_s, C_DIM)
            pygame.draw.rect(self.screen, (60, 66, 78), (x0 + 60, yy + 3, w - 110, 12))
            if probs is not None:
                top = self.last_action == i   # 実際に選ばれた行動を黄色で示す
                pygame.draw.rect(self.screen, C_BAR_TOP if top else C_BAR,
                                 (x0 + 60, yy + 3, int((w - 110) * float(probs[i])), 12))
                self.text(f"{probs[i] * 100:4.0f}%", (x0 + w - 44, yy), self.f_s, C_DIM)
            elif self.last_action == i:
                pygame.draw.rect(self.screen, C_BAR_TOP, (x0 + 60, yy + 3, w - 110, 12))
                self.text("手本", (x0 + w - 44, yy), self.f_s, C_DIM)
        y += 10 * 19 + 6

        meta = self.model_meta
        y = self.text("選び方: " + ("くじ引き（確率に従う）" if self.sample else "確率が最大の行動"),
                      (x0, y), self.f_s, C_DIM)
        if self.model is not None:
            y = self.text(f"model.pt: 手本 {meta.get('num_demos', '?')} 回 / 重み {meta.get('num_parameters', '?')} 個",
                          (x0, y), self.f_s, C_DIM)
        else:
            y = self.text("model.pt: まだありません", (x0, y), self.f_s, C_DIM)

        y += 6
        y = self.wrap(self.message, (x0, y), w, self.f_s, C_BAR_TOP)
        keys = ("1 手本  2 自動  3 ランダム  Space 新配置  F 失敗で保存  X 除外  "
                "P 再生  T 学習  L 読込  E 評価  G グラフ  S 選び方  Esc 終了")
        self.wrap(keys, (x0, WIN_H - 58), w, self.f_s, C_DIM)

    def draw_progress(self, title: str, ratio: float, sub: str) -> None:
        self.draw()
        r = pygame.Rect(60, WIN_H / 2 - 50, E.FIELD_W - 120, 100)
        pygame.draw.rect(self.screen, C_PANEL, r, border_radius=8)
        self.text(title, (r.x + 16, r.y + 10), self.f_m)
        pygame.draw.rect(self.screen, (60, 66, 78), (r.x + 16, r.y + 44, r.w - 32, 14))
        pygame.draw.rect(self.screen, C_BAR, (r.x + 16, r.y + 44, int((r.w - 32) * ratio), 14))
        self.text(sub, (r.x + 16, r.y + 66), self.f_s, C_DIM)
        pygame.display.flip()

    def draw(self) -> None:
        self.screen.fill(C_BG)
        self.draw_field()
        self.draw_panel()
        if self.overlay is not None:
            ow, oh = self.overlay.get_size()
            scale = min((WIN_W - 40) / ow, (WIN_H - 60) / oh, 1.0)
            img = pygame.transform.smoothscale(self.overlay, (int(ow * scale), int(oh * scale)))
            rx, ry = (WIN_W - img.get_width()) // 2, (WIN_H - img.get_height()) // 2
            pygame.draw.rect(self.screen, C_PANEL, (rx - 10, ry - 10, img.get_width() + 20, img.get_height() + 40))
            self.screen.blit(img, (rx, ry))
            self.text("何かキーを押すと閉じます", (rx, ry + img.get_height() + 6), self.f_s, C_DIM)

    # ---- メインループ ------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        if self.overlay is not None:
            self.overlay = None
            return True
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_1:
            self.set_mode(MODE_DEMO)
        elif key == pygame.K_2:
            self.set_mode(MODE_AUTO)
        elif key == pygame.K_3:
            self.set_mode(MODE_RANDOM)
        elif key == pygame.K_SPACE:
            if self.mode in (MODE_REPLAY, MODE_EVAL):
                self.mode = MODE_DEMO
            self.new_layout()
            self.message = MODE_LABEL[self.mode] + "：新しい配置にしました。"
        elif key == pygame.K_f and self.mode == MODE_DEMO:
            self.finish_demo("failure")
        elif key == pygame.K_x:
            self.toggle_exclude()
        elif key == pygame.K_p:
            self.start_replay()
        elif key == pygame.K_t:
            self.train()
        elif key == pygame.K_l:
            self.try_load_model()
        elif key == pygame.K_s:
            self.sample = not self.sample
            self.message = ("行動の選び方：確率に従ってくじ引き" if self.sample
                            else "行動の選び方：確率が最大の行動を必ず選ぶ")
        elif key == pygame.K_e:
            self.start_eval()
        elif key == pygame.K_g:
            plot = self.model_meta.get("loss_plot")
            if plot and Path(plot).exists():
                self.show_image(Path(plot))
            else:
                self.message = "表示できる誤差グラフがありません。"
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.auto_speed = min(self.auto_speed + 1, 10)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.auto_speed = max(self.auto_speed - 1, 1)
        return True

    def run(self) -> None:
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    running = self.handle_key(ev.key)

            if self.overlay is None:
                if self.mode == MODE_DEMO:
                    self.demo_step()
                elif self.mode in (MODE_AUTO, MODE_RANDOM) and not self.paused_until_key:
                    for _ in range(self.auto_speed):
                        self.auto_step()
                    if self.env.done or self.env.steps >= E.MAX_STEPS:
                        self.end_wait += 1
                        if self.end_wait > FPS * 1.5:   # 結果を少し見せてから次の配置へ
                            self.end_wait = 0
                            self.new_layout()
                elif self.mode == MODE_REPLAY:
                    if self.replay_i < len(self.replay_frames):
                        f = self.replay_frames[self.replay_i]
                        self.env.set_from_state(f["state"])
                        self.last_action = f["action"]
                        self.replay_i += 1
                elif self.mode == MODE_EVAL:
                    self.eval_step()

            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        sys.exit(0)
