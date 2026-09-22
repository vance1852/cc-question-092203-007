"""外部数据来源指纹。

成功导入后，为每个外部文件计算可重复识别的指纹（SHA-256），
并保存规范化后的来源摘要，写入运行结果 results.json，
使后续 AEP / 尾流计算所依据的数据可追溯、可复现。
"""

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


def compute_file_fingerprint(path: str | Path, chunk_size: int = 1 << 16) -> str:
    """计算文件原始字节的 SHA-256 指纹（含 BOM、换行、空白）。

    采用原始字节而非解析结果，任何对源文件的改动都会改变指纹，
    从而能识别“同一路径但内容已变更”的情况。
    """
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SourceFingerprint:
    """单个外部数据源的规范化来源摘要。

    Attributes
    ----------
    kind : str
        数据类型，如 ``"wind_resource"`` / ``"turbine"``
    path : str
        配置中引用的路径（相对路径保留原样）
    resolved_path : str
        解析后的绝对路径
    sha256 : str
        源文件原始字节 SHA-256
    size_bytes : int
        文件大小
    rows : int
        解析出的数据行数（扇区数 / 曲线点数）
    details : dict
        类型相关的规范化摘要（列名、单位转换、归一化策略等）
    """

    kind: str
    path: str
    resolved_path: str
    sha256: str
    size_bytes: int
    rows: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_fingerprint(
    *,
    kind: str,
    path: str | Path,
    referenced_as: Optional[str] = None,
    rows: int = 0,
    details: Optional[dict[str, Any]] = None,
) -> SourceFingerprint:
    """读取文件并构造来源指纹。"""
    p = Path(path)
    return SourceFingerprint(
        kind=kind,
        path=referenced_as if referenced_as is not None else str(p),
        resolved_path=str(p.resolve()),
        sha256=compute_file_fingerprint(p),
        size_bytes=p.stat().st_size,
        rows=rows,
        details=details or {},
    )
