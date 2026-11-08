"""手本データの保存と読み込み（ロボットで言う「データ作成」と「アノテーション」）。

手本 1 回分を JSON ファイル 1 個に保存する。中身は次のとおり。
- frames: 1 コマごとの {"state": 7 つの数値, "action": 0〜9}
- result: "success"（成功）か "failure"（失敗）
- excluded: true なら学習に使わない（除外の印）
- note: 除外の理由などのメモ
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEMO_DIR = ROOT / "data" / "demos"


@dataclass
class Demo:
    demo_id: str
    seed: int | None
    result: str = "failure"
    excluded: bool = False
    note: str = ""
    created: str = ""
    source: str = "human"          # human（人の手本）か synthetic（動作確認用の自動生成）
    frames: list[dict] = field(default_factory=list)
    path: Path | None = None

    @property
    def usable(self) -> bool:
        """学習に使えるか：成功していて、除外の印がない手本だけを使う。"""
        return self.result == "success" and not self.excluded and len(self.frames) > 0

    def to_json(self) -> dict:
        return {
            "demo_id": self.demo_id,
            "seed": self.seed,
            "result": self.result,
            "excluded": self.excluded,
            "note": self.note,
            "created": self.created,
            "source": self.source,
            "num_frames": len(self.frames),
            "frames": self.frames,
        }


def save_demo(demo: Demo, demo_dir: Path = DEFAULT_DEMO_DIR) -> Path:
    demo_dir.mkdir(parents=True, exist_ok=True)
    if not demo.created:
        demo.created = time.strftime("%Y-%m-%d %H:%M:%S")
    path = demo.path or demo_dir / f"{demo.demo_id}.json"
    path.write_text(json.dumps(demo.to_json(), ensure_ascii=False), encoding="utf-8")
    demo.path = path
    return path


def new_demo_id(demo_dir: Path = DEFAULT_DEMO_DIR) -> str:
    """demo_0001 のような連番を振る。番号順 = 記録した順。"""
    demo_dir.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem.split("_")[1]) for p in demo_dir.glob("demo_*.json")
            if p.stem.split("_")[1].isdigit()]
    return f"demo_{(max(nums) + 1 if nums else 1):04d}"


def load_demos(demo_dir: Path = DEFAULT_DEMO_DIR) -> list[Demo]:
    """記録した順（ファイル名の番号順）に並べて返す。"""
    demos = []
    for p in sorted(demo_dir.glob("demo_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        demos.append(Demo(
            demo_id=d["demo_id"], seed=d.get("seed"), result=d.get("result", "failure"),
            excluded=d.get("excluded", False), note=d.get("note", ""),
            created=d.get("created", ""), source=d.get("source", "human"),
            frames=d.get("frames", []), path=p,
        ))
    return demos


def usable_demos(demo_dir: Path = DEFAULT_DEMO_DIR, limit: int | None = None) -> list[Demo]:
    """学習に使える手本を、記録した順に最大 limit 個返す。"""
    demos = [d for d in load_demos(demo_dir) if d.usable]
    return demos[:limit] if limit else demos
