"""モデル（方策ネット）の定義・学習・保存・読み込み。

方策（policy）とは「状態を受け取って行動を返すもの」のこと。
ここでは 7 → 64 → 64 → 10 の全結合ネットを使う。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .demos import Demo
from .env import NUM_ACTIONS, STATE_DIM

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DEFAULT_MODEL = MODEL_DIR / "model.pt"
RUNS_DIR = ROOT / "runs"


class PolicyNet(nn.Module):
    """7 つの数値 → 10 個の行動それぞれの「点数（ロジット）」。"""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden),   # 7×64 の重み + 64 のバイアス
            nn.ReLU(),
            nn.Linear(hidden, hidden),      # 64×64 + 64
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS), # 64×10 + 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---- 保存と読み込み --------------------------------------------------------
def save_model(model: PolicyNet, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path: Path) -> tuple[PolicyNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PolicyNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})


# ---- 推論（学習済みの重みで行動を選ぶだけ。重みは変わらない） ----------------
@torch.no_grad()
def predict_probs(model: PolicyNet, state: list[float]) -> np.ndarray:
    x = torch.tensor([state], dtype=torch.float32)
    return torch.softmax(model(x), dim=1)[0].numpy()


# ---- 学習 ------------------------------------------------------------------
def demos_to_arrays(demos: list[Demo]) -> tuple[np.ndarray, np.ndarray]:
    states = [f["state"] for d in demos for f in d.frames]
    actions = [f["action"] for d in demos for f in d.frames]
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


def split_demos(demos: list[Demo], val_ratio: float = 0.2, seed: int = 0) -> tuple[list[Demo], list[Demo]]:
    """手本を「回」単位で学習用と検証用に分ける。

    コマ単位で混ぜて分けると、同じ回の隣り合うコマ（ほぼ同じ状態）が両方に入り、
    検証の誤差が実力より良く見えてしまうため、回ごとに分ける。
    """
    if len(demos) < 2:
        return list(demos), []
    order = list(range(len(demos)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(len(demos) * val_ratio)))
    val = [demos[i] for i in order[:n_val]]
    train = [demos[i] for i in order[n_val:]]
    return train, val


def train_policy(
    demos: list[Demo],
    epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 0,
    on_epoch: Callable[[int, float, float | None], None] | None = None,
) -> tuple[PolicyNet, dict]:
    """手本から方策ネットを学習する。

    1 エポック = 学習用データを一通り使って重みを更新すること。
    返り値の history に、エポックごとの学習誤差・検証誤差・検証正解率が入る。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_demos, val_demos = split_demos(demos, seed=seed)
    x_tr, y_tr = demos_to_arrays(train_demos)
    x_va, y_va = demos_to_arrays(val_demos) if val_demos else (None, None)
    if len(x_tr) == 0:
        raise ValueError("学習に使える手本がありません。成功した手本を記録してください。")

    # 「つかむ」「離す」は 1 回の手本で 1 コマずつしか出てこない少数派なので、
    # 少数派の行動を間違えたときの誤差を少し重くして、無視されにくくする。
    counts = np.bincount(y_tr, minlength=NUM_ACTIONS).astype(np.float32)
    weights = np.where(counts > 0, (counts.sum() / (NUM_ACTIONS * np.maximum(counts, 1))) ** 0.5, 1.0)
    weights = np.clip(weights, 0.5, 10.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    model = PolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.tensor(x_tr), torch.tensor(y_tr)
    xv = torch.tensor(x_va) if x_va is not None else None
    yv = torch.tensor(y_va) if y_va is not None else None

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(xt[idx])            # 1) 今の重みで予測する
            loss = loss_fn(logits, yt[idx])    # 2) 手本の行動とのずれ（誤差）を測る
            optimizer.zero_grad()
            loss.backward()                    # 3) 誤差を減らす向きを計算する（勾配）
            optimizer.step()                   # 4) 重みを少しだけ動かす
            total += loss.item() * len(idx)
        train_loss = total / len(xt)

        val_loss = None
        model.eval()
        if xv is not None and len(xv) > 0:
            with torch.no_grad():
                logits = model(xv)
                val_loss = loss_fn(logits, yv).item()
                val_acc = (logits.argmax(1) == yv).float().mean().item()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        if on_epoch:
            on_epoch(epoch, train_loss, val_loss)

    model.eval()
    info = {
        "num_demos": len(demos),
        "num_train_demos": len(train_demos),
        "num_val_demos": len(val_demos),
        "num_train_frames": int(len(x_tr)),
        "num_val_frames": int(len(x_va)) if x_va is not None else 0,
        "demo_ids": [d.demo_id for d in demos],
        "epochs": epochs,
        "num_parameters": count_parameters(model),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "action_counts": counts.astype(int).tolist(),
        "history": history,
    }
    return model, info


def plot_history(history: dict, path: Path, title: str) -> None:
    """学習誤差と検証誤差のグラフを PNG に保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for name in ["Yu Gothic", "Meiryo", "MS Gothic"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=100)
    ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], label="学習データの誤差")
    if history["val_loss"]:
        ax.plot(ep, history["val_loss"], label="検証データの誤差")
    ax.set_xlabel("エポック（学習データを一通り使った回数）")
    ax.set_ylabel("誤差（交差エントロピー）")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
