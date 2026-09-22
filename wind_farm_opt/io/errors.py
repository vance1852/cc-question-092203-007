"""外部数据导入错误体系。

所有导入错误都定位到文件，尽可能定位到物理行、数据行与列名，
避免资源分析师在测风塔扇区表与机组曲线表中人工排查。
"""

from pathlib import Path
from typing import Optional


class DataImportError(ValueError):
    """外部数据文件导入错误。

    Parameters
    ----------
    message : str
        错误描述
    path : Optional[str]
        数据文件路径
    line : Optional[int]
        物理行号（从 1 开始，含表头与注释/空行）
    data_row : Optional[int]
        数据行号（从 1 开始，不含表头与注释）
    column : Optional[str]
        列名或列标识
    """

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        line: Optional[int] = None,
        data_row: Optional[int] = None,
        column: Optional[str] = None,
    ) -> None:
        self.message = message
        self.path = str(path) if path is not None else None
        self.line = line
        self.data_row = data_row
        self.column = column
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """格式化为带定位信息的中文消息。"""
        loc: list[str] = []
        if self.path is not None:
            loc.append(f"文件: {self.path}")
        if self.line is not None:
            loc.append(f"物理行: {self.line}")
        if self.data_row is not None:
            loc.append(f"数据行: {self.data_row}")
        if self.column is not None:
            loc.append(f"列: {self.column}")
        if loc:
            return f"{self.message}（{'，'.join(loc)}）"
        return self.message


def file_error(message: str, path: str | Path) -> DataImportError:
    """构造文件级导入错误。"""
    return DataImportError(message, path=str(path))


def row_error(
    message: str,
    path: str | Path,
    *,
    line: Optional[int] = None,
    data_row: Optional[int] = None,
    column: Optional[str] = None,
) -> DataImportError:
    """构造定位到数据行的导入错误。"""
    return DataImportError(
        message, path=str(path), line=line, data_row=data_row, column=column
    )
