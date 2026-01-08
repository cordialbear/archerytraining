"""数据保存模块，支持本地数据库、txt文件保存和远程API保存。"""

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import requests

from data import Record, Round, Shoot


class SaveMode(Enum):
    """保存模式枚举。"""

    LOCAL = "local"
    REMOTE = "remote"
    BOTH = "both"


class IDataSaver(ABC):
    """数据保存接口类，定义通用的数据操作接口。"""

    @abstractmethod
    def save(self, record: Record) -> bool:
        """保存记录。

        Args:
            record: 要保存的记录。

        Returns:
            保存是否成功。
        """
        pass

    @abstractmethod
    def load(self, record_id: Optional[int] = None) -> list[Record]:
        """加载记录。

        Args:
            record_id: 要加载的记录 ID，如果为 None 则加载所有记录。

        Returns:
            记录列表。
        """
        pass


class LocalDataSaver(IDataSaver):
    """本地数据保存器，支持数据库和txt文件保存。"""

    def __init__(
        self,
        db_path: str = "archery_data.db",
        txt_dir: str = "data_records",
        save_to_db: bool = True,
        save_to_txt: bool = True,
    ) -> None:
        """初始化本地数据保存器。

        Args:
            db_path: SQLite 数据库文件路径。
            txt_dir: txt 文件保存目录。
            save_to_db: 是否保存到数据库。
            save_to_txt: 是否保存到txt文件。
        """
        self.db_path = db_path
        self.txt_dir = Path(txt_dir)
        self.save_to_db = save_to_db
        self.save_to_txt = save_to_txt

        # 确保 txt 目录存在
        if self.save_to_txt:
            self.txt_dir.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        if self.save_to_db:
            self._init_database()

    def _init_database(self) -> None:
        """初始化 SQLite 数据库表结构。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建记录表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        # 创建轮次表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL,
                round_index INTEGER NOT NULL,
                FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
            )
            """
        )

        # 创建射箭记录表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS shoots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id INTEGER NOT NULL,
                shoot_index INTEGER NOT NULL,
                score INTEGER NOT NULL,
                is_x INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (round_id) REFERENCES rounds(id) ON DELETE CASCADE
            )
            """
        )

        conn.commit()
        conn.close()

    def save(self, record: Record) -> bool:
        """保存记录到本地（数据库和/或txt文件）。

        Args:
            record: 要保存的记录。

        Returns:
            保存是否成功。
        """
        db_success = True
        txt_success = True

        if self.save_to_db:
            db_success = self._save_to_database(record)

        if self.save_to_txt:
            txt_success = self._save_to_txt(record)

        return db_success and txt_success

    def _save_to_database(self, record: Record) -> bool:
        """保存记录到数据库。

        Args:
            record: 要保存的记录。

        Returns:
            保存是否成功。
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 插入记录
            cursor.execute(
                """
                INSERT INTO records (time, name, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    record.time.isoformat(),
                    record.name,
                    datetime.now().isoformat(),
                ),
            )
            record_id = cursor.lastrowid

            # 插入轮次和射箭记录
            for round_idx, round_data in enumerate(record.rounds):
                cursor.execute(
                    """
                    INSERT INTO rounds (record_id, round_index)
                    VALUES (?, ?)
                    """,
                    (record_id, round_idx),
                )
                round_id = cursor.lastrowid

                for shoot_idx, shoot in round_data.shoots.items():
                    cursor.execute(
                        """
                        INSERT INTO shoots (round_id, shoot_index, score, is_x)
                        VALUES (?, ?, ?, ?)
                        """,
                        (round_id, shoot_idx, shoot.score, 1 if shoot.is_x else 0),
                    )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"保存到数据库失败: {e}")
            return False

    def _save_to_txt(self, record: Record) -> bool:
        """保存记录到 txt 文件。

        Args:
            record: 要保存的记录。

        Returns:
            保存是否成功。
        """
        try:
            # 生成文件名：时间戳_名称.txt
            timestamp = record.time.strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in record.name if c.isalnum() or c in (" ", "-", "_"))
            filename = f"{timestamp}_{safe_name}.txt"
            filepath = self.txt_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"训练记录\n")
                f.write(f"{'=' * 50}\n")
                f.write(f"时间: {record.time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"名称: {record.name}\n")
                f.write(f"{'=' * 50}\n\n")

                for round_idx, round_data in enumerate(record.rounds, 1):
                    f.write(f"第 {round_idx} 轮:\n")
                    f.write(f"{'-' * 30}\n")

                    # 按箭的序号排序
                    sorted_shoots = sorted(round_data.shoots.items(), key=lambda x: x[0])
                    for shoot_idx, shoot in sorted_shoots:
                        x_mark = "X" if shoot.is_x else ""
                        f.write(f"  第 {shoot_idx} 箭: {shoot.score} 环 {x_mark}\n")

                    # 计算本轮总分
                    total_score = sum(s.score for s in round_data.shoots.values())
                    f.write(f"  本轮总分: {total_score} 环\n\n")

                # 计算总成绩
                total_score = sum(
                    sum(s.score for s in r.shoots.values()) for r in record.rounds
                )
                f.write(f"{'=' * 50}\n")
                f.write(f"总成绩: {total_score} 环\n")
                f.write(f"总轮数: {len(record.rounds)}\n")

            return True
        except Exception as e:
            print(f"保存到 txt 文件失败: {e}")
            return False

    def load(self, record_id: Optional[int] = None) -> list[Record]:
        """从数据库加载记录。

        Args:
            record_id: 要加载的记录 ID，如果为 None 则加载所有记录。

        Returns:
            记录列表。
        """
        if not self.save_to_db:
            print("数据库保存未启用，无法加载记录")
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if record_id:
                cursor.execute("SELECT * FROM records WHERE id = ?", (record_id,))
            else:
                cursor.execute("SELECT * FROM records ORDER BY created_at DESC")

            records = []
            for row in cursor.fetchall():
                record_id_val = row["id"]
                record = Record(
                    time=datetime.fromisoformat(row["time"]),
                    name=row["name"],
                )

                # 加载轮次
                cursor.execute(
                    "SELECT * FROM rounds WHERE record_id = ? ORDER BY round_index",
                    (record_id_val,),
                )
                for round_row in cursor.fetchall():
                    round_id = round_row["id"]
                    round_data = Round()

                    # 加载射箭记录
                    cursor.execute(
                        "SELECT * FROM shoots WHERE round_id = ? ORDER BY shoot_index",
                        (round_id,),
                    )
                    for shoot_row in cursor.fetchall():
                        shoot = Shoot(
                            score=shoot_row["score"],
                            is_x=bool(shoot_row["is_x"]),
                        )
                        round_data.add_shoot(shoot_row["shoot_index"], shoot)

                    record.add_round(round_data)

                records.append(record)

            conn.close()
            return records
        except Exception as e:
            print(f"从数据库加载失败: {e}")
            return []


class RemoteDataSaver(IDataSaver):
    """远程数据保存器，通过API保存数据。"""

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        """初始化远程数据保存器。

        Args:
            api_url: 远程 API 地址。
            api_key: 远程 API 认证密钥（可选）。
            timeout: 请求超时时间（秒）。
        """
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout

    def save(self, record: Record) -> bool:
        """通过远程 API 保存记录。

        Args:
            record: 要保存的记录。

        Returns:
            保存是否成功。
        """
        try:
            # 将记录转换为 JSON 格式
            data = self._record_to_dict(record)

            # 准备请求头
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # 发送 POST 请求
            response = requests.post(
                self.api_url,
                json=data,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code in (200, 201):
                return True
            else:
                print(f"远程 API 返回错误: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"远程 API 请求失败: {e}")
            return False
        except Exception as e:
            print(f"保存到远程失败: {e}")
            return False

    def load(self, record_id: Optional[int] = None) -> list[Record]:
        """从远程 API 加载记录。

        Args:
            record_id: 要加载的记录 ID，如果为 None 则加载所有记录。

        Returns:
            记录列表。
        """
        try:
            # 准备请求头
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # 构建请求URL
            url = self.api_url
            if record_id:
                url = f"{self.api_url}/{record_id}"

            # 发送 GET 请求
            response = requests.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                return self._dict_to_records(data)
            else:
                print(f"远程 API 返回错误: {response.status_code} - {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"远程 API 请求失败: {e}")
            return []
        except Exception as e:
            print(f"从远程加载失败: {e}")
            return []

    def _record_to_dict(self, record: Record) -> dict:
        """将 Record 对象转换为字典格式。

        Args:
            record: 要转换的记录。

        Returns:
            字典格式的记录数据。
        """
        return {
            "time": record.time.isoformat(),
            "name": record.name,
            "rounds": [
                {
                    "shoots": {
                        str(idx): {"score": shoot.score, "is_x": shoot.is_x}
                        for idx, shoot in round_data.shoots.items()
                    }
                }
                for round_data in record.rounds
            ],
        }

    def _dict_to_records(self, data: dict | list) -> list[Record]:
        """将字典格式的数据转换为 Record 对象列表。

        Args:
            data: 字典或字典列表格式的数据。

        Returns:
            Record 对象列表。
        """
        records = []
        data_list = data if isinstance(data, list) else [data]

        for item in data_list:
            record = Record(
                time=datetime.fromisoformat(item["time"]),
                name=item["name"],
            )

            for round_data in item.get("rounds", []):
                round_obj = Round()
                for idx_str, shoot_data in round_data.get("shoots", {}).items():
                    shoot = Shoot(
                        score=shoot_data["score"],
                        is_x=shoot_data.get("is_x", False),
                    )
                    round_obj.add_shoot(int(idx_str), shoot)
                record.add_round(round_obj)

            records.append(record)

        return records


class DataSaver(IDataSaver):
    """数据保存器工厂类，根据参数创建对应的保存器实例。"""

    def __init__(
        self,
        mode: str | SaveMode = SaveMode.LOCAL,
        db_path: str = "archery_data.db",
        txt_dir: str = "data_records",
        save_to_db: bool = True,
        save_to_txt: bool = True,
        remote_api_url: Optional[str] = None,
        remote_api_key: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        """初始化数据保存器。

        Args:
            mode: 保存模式，"local"、"remote" 或 "both"。
            db_path: SQLite 数据库文件路径（本地模式）。
            txt_dir: txt 文件保存目录（本地模式）。
            save_to_db: 是否保存到数据库（本地模式）。
            save_to_txt: 是否保存到txt文件（本地模式）。
            remote_api_url: 远程 API 地址（远程模式）。
            remote_api_key: 远程 API 认证密钥（远程模式）。
            timeout: 远程请求超时时间（秒）。
        """
        # 标准化模式参数
        if isinstance(mode, str):
            mode = SaveMode(mode.lower())

        self.mode = mode
        self._local_saver: Optional[LocalDataSaver] = None
        self._remote_saver: Optional[RemoteDataSaver] = None

        # 根据模式创建对应的保存器实例
        if mode in (SaveMode.LOCAL, SaveMode.BOTH):
            self._local_saver = LocalDataSaver(
                db_path=db_path,
                txt_dir=txt_dir,
                save_to_db=save_to_db,
                save_to_txt=save_to_txt,
            )

        if mode in (SaveMode.REMOTE, SaveMode.BOTH):
            if not remote_api_url:
                raise ValueError("远程模式需要提供 remote_api_url 参数")
            self._remote_saver = RemoteDataSaver(
                api_url=remote_api_url,
                api_key=remote_api_key,
                timeout=timeout,
            )

    def save(self, record: Record) -> bool:
        """保存记录，根据模式调用对应的保存器。

        Args:
            record: 要保存的记录。

        Returns:
            保存是否成功。
        """
        if self.mode == SaveMode.LOCAL:
            return self._local_saver.save(record) if self._local_saver else False
        elif self.mode == SaveMode.REMOTE:
            return self._remote_saver.save(record) if self._remote_saver else False
        elif self.mode == SaveMode.BOTH:
            local_success = self._local_saver.save(record) if self._local_saver else True
            remote_success = self._remote_saver.save(record) if self._remote_saver else True
            return local_success and remote_success
        else:
            return False

    def load(self, record_id: Optional[int] = None) -> list[Record]:
        """加载记录，根据模式调用对应的保存器。

        Args:
            record_id: 要加载的记录 ID，如果为 None 则加载所有记录。

        Returns:
            记录列表。
        """
        if self.mode == SaveMode.LOCAL:
            return self._local_saver.load(record_id) if self._local_saver else []
        elif self.mode == SaveMode.REMOTE:
            return self._remote_saver.load(record_id) if self._remote_saver else []
        elif self.mode == SaveMode.BOTH:
            # 优先从本地加载，如果本地没有则从远程加载
            if self._local_saver:
                records = self._local_saver.load(record_id)
                if records:
                    return records
            if self._remote_saver:
                return self._remote_saver.load(record_id)
            return []
        else:
            return []
