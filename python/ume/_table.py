"""A small printable table, the display primitive of every ume result."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import Any


def format_value(value: Any, digits: int = 4) -> str:
    """Format one cell: numbers to `digits` significant digits, everything else with ``str``."""
    if isinstance(value, bool) or value is None:
        return "" if value is None else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value == 0:
            return "0"
        if value != 0 and (abs(value) >= 1e6 or abs(value) < 1e-4):
            return f"{value:.{digits - 1}e}"
        return f"{value:.{digits}g}" if abs(value) >= 1 else f"{value:.{digits}f}"
    return str(value)


class Table:
    """Rows with named columns that print as text in a terminal and as HTML in a notebook.

    Parameters
    ----------
    columns : sequence of str
        Column names.
    rows : sequence of sequence
        One sequence of values per row, in column order.
    title : str, optional
        Printed above the table.
    footer : str, optional
        Printed below the table.

    Examples
    --------
    >>> from ume._table import Table
    >>> print(Table(["metric", "estimate"], [["hours", 0.66]]))
    metric  estimate
    hours     0.6600
    """

    def __init__(
        self,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        title: str | None = None,
        footer: str | None = None,
    ) -> None:
        self.columns = list(columns)
        self.rows = [list(r) for r in rows]
        self.title = title
        self.footer = footer

    def to_dict(self) -> dict[str, list[Any]]:
        """Return the table as ``{column: values}``."""
        return {c: [r[i] for r in self.rows] for i, c in enumerate(self.columns)}

    def to_pandas(self) -> Any:
        """Return the table as a pandas DataFrame (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(self.to_dict())

    def to_polars(self) -> Any:
        """Return the table as a polars DataFrame (requires polars)."""
        import polars as pl

        return pl.DataFrame(self.to_dict(), strict=False)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        cells = [[format_value(v) for v in r] for r in self.rows]
        widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(self.columns)]
        numeric = [all(isinstance(r[i], (int, float)) for r in self.rows) for i in range(len(self.columns))]

        def line(values: Sequence[str]) -> str:
            parts = [v.rjust(w) if num else v.ljust(w) for v, w, num in zip(values, widths, numeric, strict=True)]
            return "  ".join(parts).rstrip()

        out = ([self.title] if self.title else []) + [line(self.columns)] + [line(r) for r in cells]
        if self.footer:
            out.append(self.footer)
        return "\n".join(out)

    def _repr_html_(self) -> str:
        head = "".join(f"<th>{html.escape(c)}</th>" for c in self.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(format_value(v))}</td>" for v in r) + "</tr>" for r in self.rows
        )
        title = f"<caption>{html.escape(self.title)}</caption>" if self.title else ""
        footer = f"<p><small>{html.escape(self.footer)}</small></p>" if self.footer else ""
        return f"<table>{title}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{footer}"
