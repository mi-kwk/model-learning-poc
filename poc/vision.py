"""段階 2：見るモデル（画像 → 7 つの数値）。

分割型の構成:
  画像 ──[見るモデル]──> 推定した 7 つの数値 ──[動くモデル（段階 1 の model.pt）]──> 行動

見るモデルは「画像のどこに カーソル・箱・置き場 があるか、箱をつかんでいるか」を当てる。
出力は段階 1 の入力と同じ 7 つの数値なので、動くモデルは作り直さずにそのまま使える。
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .env import BOX_SIZE, FIELD_H, FIELD_W, BoxEnv
from .render import IMG_SIZE, render_image

ROOT = Path(__file__).resolve().parent.parent
VISION_MODEL = ROOT / "models" / "vision.pt"


class VisionNet(nn.Module):
    """画像（3×128×128）→ 位置 6 つ（-1〜1）＋ つかんでいるかの点数 1 つ。

    1) 畳み込み層（Conv）：画像の小さな範囲を見て「赤い四角」「青い丸」「緑の枠」などの
       模様を見つける部品。重ねて 128×128 → 32×32 の「特徴の地図」にする。
    2) 物ごとに 1 枚ずつ、「ここにありそう」という地図（ヒートマップ）を作る（3 枚）。
    3) 地図の重心を計算して、その物の位置（x, y）とする（spatial soft-argmax）。
    4) つかみは、箱の色（赤か黄色か）が分かればよいので、特徴の地図全体から判定する。
    """

    def __init__(self) -> None:
        super().__init__()
        # BatchNorm（バッチ正規化）は、各層の出力の大きさをそろえて学習を安定させる部品。
        # 入れない版では、初期値しだいでカーソルの位置をまったく学習できないことがあった。
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),   # 128 → 64
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 64 → 32
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.heatmap = nn.Conv2d(64, 3, 1)    # カーソル・箱・置き場の地図 3 枚
        self.hold = nn.Linear(64, 1)          # つかんでいるかの点数
        side = IMG_SIZE // 4
        coords = (torch.arange(side, dtype=torch.float32) + 0.5) / side * 2 - 1  # 各マスの中心（-1〜1）
        self.register_buffer("coords", coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)                                   # (枚数, 64, 32, 32)
        h = self.heatmap(f)                                    # (枚数, 3, 32, 32)
        n, k, hh, ww = h.shape
        prob = torch.softmax(h.view(n, k, -1), dim=2).view(n, k, hh, ww)  # 地図を確率にする
        x_pos = (prob.sum(dim=2) * self.coords).sum(dim=2)     # 重心の x
        y_pos = (prob.sum(dim=3) * self.coords).sum(dim=2)     # 重心の y
        pos = torch.stack([x_pos, y_pos], dim=2).view(n, 6)    # [cx, cy, bx, by, tx, ty]
        hold = self.hold(f.amax(dim=(2, 3)))                   # 地図全体で一番強い反応から判定
        return torch.cat([pos, hold], dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """(枚数, 高さ, 幅, 3) の 0〜255 → (枚数, 3, 高さ, 幅) の 0〜1。

    白い背景が 0、色の付いた物が大きな値になるように反転する（物のある所だけ反応させやすい）。
    """
    return 1.0 - torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0


# ---- 学習データ（画像と正解の組）を作る ---------------------------------------
def random_scene(env: BoxEnv, rng: random.Random) -> None:
    """いろいろな場面を作る。自動実行中に出てきそうな場面を偏りなく含める。"""
    env.reset(rng.randrange(0, 1_000_000))
    env.cursor = [rng.uniform(0, FIELD_W), rng.uniform(0, FIELD_H)]
    r = rng.random()
    if r < 0.30:                      # 箱をつかんで運んでいる
        env.holding = True
        env.box = list(env.cursor)
    elif r < 0.50:                    # 箱のすぐ近くにいるが、まだつかんでいない
        env.holding = False
        env.cursor = [env.box[0] + rng.uniform(-BOX_SIZE, BOX_SIZE),
                      env.box[1] + rng.uniform(-BOX_SIZE, BOX_SIZE)]
    elif r < 0.62:                    # 置き場の近くで箱を離した直後
        env.holding = False
        env.box = [env.target[0] + rng.uniform(-40, 40), env.target[1] + rng.uniform(-40, 40)]
        env.cursor = [env.box[0] + rng.uniform(-10, 10), env.box[1] + rng.uniform(-10, 10)]
    elif r < 0.80:                    # カーソルが画面の端にいる（丸が半分切れて見える）
        # 最初の版ではこの場面が少なく、自動実行中に端で大きく見間違えたため追加した
        env.holding = rng.random() < 0.3
        side = rng.randrange(4)
        edge = rng.uniform(0, 20)
        along = rng.uniform(0, FIELD_W)
        env.cursor = [[edge, along], [FIELD_W - edge, along], [along, edge], [along, FIELD_H - edge]][side]
        if env.holding:
            env.box = list(env.cursor)
    else:                             # 箱から離れた所にいる
        env.holding = False
    env.cursor = [min(max(env.cursor[0], 0), FIELD_W), min(max(env.cursor[1], 0), FIELD_H)]
    env.box = [min(max(env.box[0], 0), FIELD_W), min(max(env.box[1], 0), FIELD_H)]


def make_dataset(n: int, seed: int = 0, on_progress: Callable[[int], None] | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """画像 n 枚と、その正解（7 つの数値）を作る。

    この PoC は画面をプログラムで描いているので、物の正確な位置が分かっている。
    そのため人が印を付けなくても、正解（ラベル）を自動で付けられる（自動アノテーション）。
    """
    rng = random.Random(seed)
    env = BoxEnv()
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        random_scene(env, rng)
        images[i] = render_image(env)
        labels[i] = env.state()
        if on_progress and i % 500 == 0:
            on_progress(i)
    return images, labels


# ---- 学習 ------------------------------------------------------------------
def vision_loss(out: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """位置のずれ（二乗誤差）＋ つかみの当たり外れ（交差エントロピー）。"""
    pos = nn.functional.mse_loss(out[:, :6], y[:, :6])
    hold = nn.functional.binary_cross_entropy_with_logits(out[:, 6], y[:, 6])
    return pos + 0.1 * hold, pos, hold


def pixel_errors(out: torch.Tensor, y: torch.Tensor) -> dict:
    """推定した位置が何ピクセルずれているか（物ごとの平均）と、つかみの正解率。"""
    d = (out[:, :6] - y[:, :6]) / 2 * torch.tensor([FIELD_W, FIELD_H] * 3)
    dist = torch.sqrt(d[:, 0::2] ** 2 + d[:, 1::2] ** 2)  # (枚数, 3)
    hold_acc = ((out[:, 6] > 0).float() == y[:, 6]).float().mean().item()
    return {
        "cursor_px": dist[:, 0].mean().item(),
        "box_px": dist[:, 1].mean().item(),
        "target_px": dist[:, 2].mean().item(),
        "hold_acc": hold_acc,
    }


def train_vision(images: np.ndarray, labels: np.ndarray, epochs: int = 20, lr: float = 1e-3,
                 batch_size: int = 64, val_ratio: float = 0.1, seed: int = 0,
                 on_epoch: Callable[[int, float, float, dict], None] | None = None
                 ) -> tuple[VisionNet, dict]:
    torch.manual_seed(seed)
    n_val = max(1, int(len(images) * val_ratio))
    x_va, y_va = to_tensor(images[:n_val]), torch.from_numpy(labels[:n_val])
    x_tr_np, y_tr = images[n_val:], torch.from_numpy(labels[n_val:])

    model = VisionNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = {"train_loss": [], "val_loss": [], "val_err": []}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.default_rng(seed + epoch).permutation(len(x_tr_np))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = to_tensor(x_tr_np[idx]), y_tr[idx]
            loss, _, _ = vision_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(x_va[i:i + 256]) for i in range(0, len(x_va), 256)])
            val_loss = vision_loss(out, y_va)[0].item()
            err = pixel_errors(out, y_va)
        history["train_loss"].append(total / len(perm))
        history["val_loss"].append(val_loss)
        history["val_err"].append(err)
        if on_epoch:
            on_epoch(epoch, total / len(perm), val_loss, err)

    info = {
        "num_images": len(images),
        "num_train_images": len(images) - n_val,
        "num_val_images": n_val,
        "epochs": epochs,
        "image_size": IMG_SIZE,
        "num_parameters": count_parameters(model),
        "final_val_err": history["val_err"][-1],
        "history": history,
    }
    return model, info


def save_vision(model: VisionNet, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_vision(path: Path = VISION_MODEL) -> tuple[VisionNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = VisionNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})


# ---- 推論：画像を見て 7 つの数値を推定する -----------------------------------
@torch.no_grad()
def predict_state(model: VisionNet, image: np.ndarray) -> list[float]:
    out = model(to_tensor(image[None]))[0]
    pos = out[:6].clamp(-1, 1).tolist()
    holding = 1.0 if out[6].item() > 0 else 0.0
    return pos + [holding]


def vision_observer(model: VisionNet) -> Callable[[BoxEnv], list[float]]:
    """環境の正確な座標ではなく、画像を見て推定した状態を返す「目」。"""
    def observe(env: BoxEnv) -> list[float]:
        return predict_state(model, render_image(env))
    return observe


def state_error_px(est: list[float], true: list[float]) -> list[float]:
    """推定した状態と正解の、物ごとのずれ（ピクセル）。"""
    out = []
    for k in range(3):
        dx = (est[2 * k] - true[2 * k]) / 2 * FIELD_W
        dy = (est[2 * k + 1] - true[2 * k + 1]) / 2 * FIELD_H
        out.append(math.hypot(dx, dy))
    return out
