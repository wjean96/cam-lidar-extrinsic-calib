"""rosbag2 (sqlite3) reader.

Reads the sqlite3 storage directly (no rosbag2_py needed) and yields
(topic, timestamp_ns, raw_bytes) tuples in time order.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import yaml


@dataclass
class TopicInfo:
    name: str
    msg_type: str
    count: int = 0


class Bag2Reader:
    """Reads a rosbag2 directory (metadata.yaml + *.db3)."""

    def __init__(self, bag_dir: str):
        self.bag_dir = os.path.abspath(bag_dir)
        if not os.path.isdir(self.bag_dir):
            raise NotADirectoryError(f"bag 디렉터리가 아닙니다: {self.bag_dir}")

        self.db_files = sorted(
            os.path.join(self.bag_dir, f)
            for f in os.listdir(self.bag_dir)
            if f.endswith(".db3")
        )
        if not self.db_files:
            raise FileNotFoundError(f"*.db3 파일이 없습니다: {self.bag_dir}")

        self._cons: Dict[int, sqlite3.Connection] = {}
        self.metadata = self._load_metadata()
        self.topics = self._load_topics()

    # ------------------------------------------------------------------ meta
    def _load_metadata(self) -> dict:
        path = os.path.join(self.bag_dir, "metadata.yaml")
        if not os.path.isfile(path):
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_topics(self) -> Dict[str, TopicInfo]:
        topics: Dict[str, TopicInfo] = {}
        info = self.metadata.get("rosbag2_bagfile_information", {})
        for entry in info.get("topics_with_message_count", []):
            md = entry.get("topic_metadata", {})
            topics[md["name"]] = TopicInfo(
                name=md["name"], msg_type=md["type"], count=entry.get("message_count", 0)
            )
        if topics:
            return topics
        # No metadata.yaml: fall back to the topic table inside the db3
        with sqlite3.connect(self.db_files[0]) as con:
            for _id, name, mtype in con.execute("SELECT id, name, type FROM topics"):
                topics[name] = TopicInfo(name=name, msg_type=mtype)
        return topics

    @property
    def duration_s(self) -> float:
        info = self.metadata.get("rosbag2_bagfile_information", {})
        return info.get("duration", {}).get("nanoseconds", 0) / 1e9

    @property
    def start_time_ns(self) -> int:
        info = self.metadata.get("rosbag2_bagfile_information", {})
        return info.get("starting_time", {}).get("nanoseconds_since_epoch", 0)

    def type_of(self, topic: str) -> str:
        if topic not in self.topics:
            raise KeyError(f"bag 에 없는 토픽입니다: {topic}\n사용 가능: {sorted(self.topics)}")
        return self.topics[topic].msg_type

    # ------------------------------------------------------------------ read
    def read(self, topics: Optional[Sequence[str]] = None) -> Iterator[Tuple[str, int, bytes]]:
        """Yield (topic, timestamp_ns, raw_cdr_bytes) in time order."""
        wanted: Optional[List[str]] = list(topics) if topics else None
        if wanted:
            for t in wanted:
                self.type_of(t)  # existence check (raises KeyError)

        for db in self.db_files:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                id2name = {i: n for i, n, _ in con.execute("SELECT id, name, type FROM topics")}
                if wanted:
                    ids = [i for i, n in id2name.items() if n in wanted]
                    if not ids:
                        continue
                    placeholders = ",".join("?" * len(ids))
                    sql = (
                        f"SELECT topic_id, timestamp, data FROM messages "
                        f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp"
                    )
                    cur = con.execute(sql, ids)
                else:
                    cur = con.execute(
                        "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
                    )
                for topic_id, ts, data in cur:
                    yield id2name[topic_id], int(ts), bytes(data)
            finally:
                con.close()

    # -------------------------------------------------------- index / random access
    def index(self, topics: Optional[Sequence[str]] = None) -> List[Tuple[str, int, int, int]]:
        """Return (topic, bag_ts_ns, db_idx, rowid) in time order without reading blobs.

        Finishes instantly even on multi-hundred-MB bags, so the two-pass scheme is:
        pick synchronized pairs from the index first, then decode only those via fetch().
        """
        wanted = list(topics) if topics else None
        if wanted:
            for t in wanted:
                self.type_of(t)

        rows: List[Tuple[str, int, int, int]] = []
        for db_i, db in enumerate(self.db_files):
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                id2name = {i: n for i, n, _ in con.execute("SELECT id, name, type FROM topics")}
                if wanted:
                    ids = [i for i, n in id2name.items() if n in wanted]
                    if not ids:
                        continue
                    ph = ",".join("?" * len(ids))
                    cur = con.execute(
                        f"SELECT topic_id, timestamp, id FROM messages WHERE topic_id IN ({ph})", ids
                    )
                else:
                    cur = con.execute("SELECT topic_id, timestamp, id FROM messages")
                rows.extend((id2name[tid], int(ts), db_i, int(rid)) for tid, ts, rid in cur)
            finally:
                con.close()
        rows.sort(key=lambda r: r[1])
        return rows

    def _con(self, db_idx: int) -> sqlite3.Connection:
        con = self._cons.get(db_idx)
        if con is None:
            con = sqlite3.connect(f"file:{self.db_files[db_idx]}?mode=ro", uri=True)
            self._cons[db_idx] = con
        return con

    def fetch(self, db_idx: int, rowids: Sequence[int]) -> Dict[int, bytes]:
        """Read only the raw CDR payloads of the given rowids."""
        out: Dict[int, bytes] = {}
        ids = list(rowids)
        if not ids:
            return out
        con = self._con(db_idx)
        for i in range(0, len(ids), 500):  # stay under SQLite's variable limit
            chunk = ids[i : i + 500]
            ph = ",".join("?" * len(chunk))
            for rid, data in con.execute(
                f"SELECT id, data FROM messages WHERE id IN ({ph})", chunk
            ):
                out[int(rid)] = bytes(data)
        return out

    def fetch_one(self, db_idx: int, rowid: int) -> bytes:
        row = self._con(db_idx).execute(
            "SELECT data FROM messages WHERE id = ?", (int(rowid),)
        ).fetchone()
        if row is None:
            raise KeyError(f"rowid {rowid} 없음 (db {db_idx})")
        return bytes(row[0])

    def close(self) -> None:
        for con in self._cons.values():
            con.close()
        self._cons.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
