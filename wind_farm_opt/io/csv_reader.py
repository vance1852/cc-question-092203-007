"""通用 CSV 读取器。

支持：
- UTF-8（含 BOM）与 GBK 编码自动回退（测风塔导出软件常见）
- ``#`` / ``;`` 注释行与行内注释
- 表头别名归一化（中英文、常见缩写）
- 物理行号（含表头/注释）与数据行号双追踪
- 列缺失与单元格解析错误定位到文件、行、列
"""

import csv
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import DataImportError, file_error, row_error

# 注释起始符（整行注释或行内注释）
_COMMENT_PREFIXES = ("#", ";")
_INLINE_COMMENT_RE = re.compile(r"\s+[#;].*$")


@dataclass
class ParsedRow:
    """一行已解析的数据。

    Attributes
    ----------
    data_row : int
        数据行序号（从 1 开始，不含表头、注释、空行）
    line : int
        文件中的物理行号（从 1 开始）
    values : dict[str, str]
        归一化列名 -> 原始单元格文本（已 strip）
    raw : list[str]
        原始字段列表
    """

    data_row: int
    line: int
    values: dict[str, str]
    raw: list[str]


def normalize_header(name: str) -> str:
    """归一化表头名称。

    去除括号注释、单位文本，统一小写、去空格与下划线，便于别名匹配。
    例如 ``"Weibull k (-)"`` -> ``"weibullk"``，``"平均风速 (m/s)"`` -> ``"平均风速"``。
    """
    s = name.strip()
    # 去除 BOM
    s = s.lstrip("﻿")
    # 去除括号内的单位说明（中英文括号）
    s = re.sub(r"[（(\[【].*?[）)\]】]", "", s)
    s = s.strip().lower()
    # 仅对 ASCII 名称去下划线/空格/连字符，中文名称保留
    if s.isascii():
        s = re.sub(r"[\s_\-/]+", "", s)
    return s


def resolve_header(
    headers: list[str],
    aliases: dict[str, tuple[str, ...]],
    path: str | Path,
) -> dict[str, int]:
    """把逻辑列名解析为物理列索引。

    Parameters
    ----------
    headers : list[str]
        原始表头（已 strip，未归一化）
    aliases : dict[str, tuple[str, ...]]
        逻辑列名 -> 可接受的归一化别名（第一个一般是规范名）
    path : path
        文件路径，仅用于错误信息

    Returns
    -------
    dict[str, int]
        逻辑列名 -> 列索引
    """
    normalized = [normalize_header(h) for h in headers]
    mapping: dict[str, int] = {}
    for logical, names in aliases.items():
        found: Optional[int] = None
        for alias in names:
            key = normalize_header(alias)
            if key in normalized:
                found = normalized.index(key)
                break
        mapping[logical] = -1 if found is None else found
    return mapping


def read_csv_rows(
    path: str | Path,
    required_columns: dict[str, tuple[str, ...]],
    optional_columns: Optional[dict[str, tuple[str, ...]]] = None,
    *,
    delimiter: Optional[str] = None,
) -> tuple[list[ParsedRow], dict[str, int], list[str]]:
    """读取带表头的 CSV 文件。

    Parameters
    ----------
    path : str | Path
        CSV 文件路径
    required_columns : dict[str, tuple[str, ...]]
        必需逻辑列及其别名
    optional_columns : dict[str, tuple[str, ...]]
        可选逻辑列及其别名
    delimiter : Optional[str]
        分隔符；None 时自动 sniff（回退到逗号）

    Returns
    -------
    tuple[list[ParsedRow], dict[str, int], list[str]]
        数据行、逻辑列名->列索引映射、原始表头
    """
    path = Path(path)
    text = _read_text(path)

    # 拆出物理行，保留行号
    raw_lines = text.splitlines()
    if not raw_lines:
        raise file_error("文件为空", path)

    # 定位表头（跳过注释/空行）
    header_line_idx: Optional[int] = None
    header_text: Optional[str] = None
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue
        header_line_idx = idx
        header_text = line
        break

    if header_text is None:
        raise file_error("未找到表头行", path)

    effective_delimiter = delimiter or _sniff_delimiter(header_text)

    # 表头行同样裁掉行内注释
    header_clean = (
        _INLINE_COMMENT_RE.sub("", header_text)
        if '"' not in header_text
        else header_text
    )
    reader_header = next(csv.reader(io.StringIO(header_clean), delimiter=effective_delimiter))
    headers = [h.strip().lstrip("﻿") for h in reader_header]
    if not any(headers):
        raise row_error(
            "表头为空", path, line=header_line_idx + 1, data_row=0
        )

    all_aliases = dict(required_columns)
    if optional_columns:
        all_aliases.update(optional_columns)
    col_index = resolve_header(headers, all_aliases, path)

    missing = [c for c in required_columns if col_index[c] < 0]
    if missing:
        alias_hint = ", ".join(
            f"{c}({'/'.join(required_columns[c])})" for c in missing
        )
        raise row_error(
            f"缺少必需列: {alias_hint}",
            path,
            line=header_line_idx + 1,
        )

    rows: list[ParsedRow] = []
    data_row = 0
    for idx in range(header_line_idx + 1, len(raw_lines)):
        line = raw_lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue

        # 行内注释裁剪（仅在非引号场景；测风塔表通常没有引号，做保守处理）
        cleaned = _INLINE_COMMENT_RE.sub("", line) if '"' not in line else line
        fields = next(csv.reader(io.StringIO(cleaned), delimiter=effective_delimiter))
        fields = [f.strip() for f in fields]

        # 全空行跳过
        if not any(fields):
            continue

        data_row += 1
        values: dict[str, str] = {}
        for logical, ci in col_index.items():
            if ci >= 0 and ci < len(fields):
                values[logical] = fields[ci]
        rows.append(
            ParsedRow(
                data_row=data_row,
                line=idx + 1,
                values=values,
                raw=fields,
            )
        )

    if not rows:
        raise file_error("表头之后没有任何数据行", path)

    return rows, col_index, headers


def parse_float(
    row: ParsedRow,
    column: str,
    path: str | Path,
    *,
    allow_blank: bool = False,
) -> Optional[float]:
    """从一行中解析浮点值，错误定位到行列。

    支持千分位逗号与英文单位尾巴（如 ``"8.5 m/s"``）的剥离。
    """
    text = row.values.get(column)
    if text is None or text == "":
        if allow_blank:
            return None
        raise row_error(
            f"列 {column} 的值缺失",
            path,
            line=row.line,
            data_row=row.data_row,
            column=column,
        )

    cleaned = text.strip()
    # 仅剥离明确的单位尾巴，避免误伤小数/科学计数法
    cleaned = re.sub(
        r"\s*(?:m/s|m\s*s\^?-?1|kw|mw|w|%|deg|°|h|hours?)?\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace(",", "").strip()

    try:
        return float(cleaned)
    except ValueError:
        raise row_error(
            f"值 '{text}' 无法解析为数值",
            path,
            line=row.line,
            data_row=row.data_row,
            column=column,
        )


def _read_text(path: Path) -> str:
    """读取文本，UTF-8(含 BOM)优先，失败回退 GBK。"""
    if not path.exists():
        raise DataImportError(f"数据文件不存在: {path}", path=str(path))
    if not path.is_file():
        raise DataImportError(f"数据路径不是文件: {path}", path=str(path))
    for encoding in ("utf-8-sig", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise file_error("无法识别的文件编码（已尝试 UTF-8/GBK）", path)


def _sniff_delimiter(header_text: str) -> str:
    """根据表头行猜测分隔符。"""
    candidates = [",", ";", "\t", "|"]
    counts = {d: header_text.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def resolve_relative_path(base_dir: str | os.PathLike[str], ref: str) -> Path:
    """按配置文件所在目录解析相对路径；绝对路径原样返回。

    支持 ``~/`` 用户目录展开。
    """
    p = Path(ref).expanduser()
    if p.is_absolute():
        return p
    return Path(base_dir) / p
