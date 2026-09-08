"""把仿真数据稳定写为 UTF-8 JSON 和列式 CSV。"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def to_serializable(value: Any) -> Any:
    """递归转换 dataclass、NumPy 数组和标量为 JSON 原生类型。"""

    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> Path:
    """以 UTF-8、两空格缩进写入 JSON，并返回绝对或传入形式的路径。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_serializable(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def write_csv_columns(
    path: str | Path,
    columns: Mapping[str, Sequence[Any] | np.ndarray],
) -> Path:
    """按映射插入顺序写入等长列；首行为字段名。"""

    if not columns:
        raise ValueError("columns 不能为空")
    normalized = {name: list(values) for name, values in columns.items()}
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("所有 CSV 列必须等长")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(normalized.keys())
        writer.writerows(zip(*normalized.values()))
    return destination
