from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class Shoot:
    """一次射箭记录。

    Attributes:
        score: 本箭的环数，例如 0–10。
        is_x: 是否为 X 环（10 环中的更精确中心）。
    """

    score: int
    is_x: bool = False


@dataclass
class Round:
    """一轮射箭记录。

    使用一个字典保存本轮中每一箭的成绩，key 为箭的序号（从 1 开始），value 为对应的 Shoot。
    """

    shoots: Dict[int, Shoot] = field(default_factory=dict)

    def add_shoot(self, idx: int, shoot: Shoot) -> None:
        """添加或更新一支箭的成绩。

        Args:
            idx: 箭的序号（推荐从 1 开始）。
            shoot: 射箭成绩。
        """

        self.shoots[idx] = shoot


@dataclass
class Record:
    """一次完整的训练或比赛记录。"""

    time: datetime
    name: str
    rounds: List[Round] = field(default_factory=list)

    def add_round(self, round_: Round) -> None:
        """添加一轮成绩。"""

        self.rounds.append(round_)


