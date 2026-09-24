"""HTML and XHTML to plain text with the standard-library parser.

Scripts, styles, templates, and other non-content elements are dropped.
Block elements become their own lines, headings stay lines of their own,
list items get a marker, and table cells of a row are joined with " | ".
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from summarizer_web.ingestion.common import Block, BlockKind, collapse_spaces, render_blocks

_SKIPPED = frozenset(
    {"script", "style", "template", "noscript", "svg", "title", "iframe", "canvas", "select"}
)
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_LISTS = frozenset({"ul", "ol", "menu", "dir"})
_ITEMS = frozenset({"li", "dt", "dd"})
_CONTAINERS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "caption", "center",
        "details", "dialog", "div", "dl", "fieldset", "figcaption", "figure", "footer",
        "form", "header", "hgroup", "html", "legend", "main", "nav", "p", "section",
        "summary", "table", "tbody", "tfoot", "thead",
    }
)
_BLOCKS = _HEADINGS | _LISTS | _ITEMS | _CONTAINERS | {"pre", "tr", "td", "th"}
_VOID_BREAKS = frozenset({"hr"})

_DECLARED_CHARSET = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9._:-]+)|<\?xml[^>]+encoding\s*=\s*["']([A-Za-z0-9._:-]+)""",
    re.IGNORECASE,
)
_SOURCE_WHITESPACE = re.compile(r"[ \t\n\r\f]+")
# The WHATWG encoding standard reads these labels as windows-1252.
_WINDOWS_1252_LABELS = frozenset(
    {"ascii", "us-ascii", "iso-8859-1", "iso8859-1", "latin1", "latin-1", "l1", "windows-1252"}
)


def declared_charset(data: bytes) -> str | None:
    """The encoding an HTML or XML document declares in its first 4 KiB."""
    match = _DECLARED_CHARSET.search(data[:4096])
    if match is None:
        return None
    label = (match.group(1) or match.group(2)).decode("ascii").lower()
    return "cp1252" if label in _WINDOWS_1252_LABELS else label


class _BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._fragments: list[str] = []
        self._open: list[str] = []
        self._lists: list[list[int] | None] = []
        self._row_cells: list[int] = []
        self._prefix = ""
        self._skip_depth = 0
        self._break_next = False

    # Structure ------------------------------------------------------------

    def _kind(self) -> BlockKind:
        for tag in reversed(self._open):
            if tag == "pre":
                return "preformatted"
            if tag in _HEADINGS:
                return "heading"
            if tag == "tr":
                return "table_row"
            if tag in _ITEMS:
                return "list_item"
        return "paragraph"

    def _in_row(self) -> bool:
        return bool(self._row_cells)

    def _boundary(self) -> None:
        if self._in_row():
            self._fragments.append(" ")
        else:
            self._flush()

    def _flush(self) -> None:
        text = "".join(self._fragments)
        self._fragments.clear()
        kind = self._kind()
        if kind == "preformatted":
            lines = [line.rstrip() for line in text.split("\n")]
            text = "\n".join(lines).strip("\n")
        else:
            text = collapse_spaces(text)
        if not text.strip():
            return
        if self._prefix:
            text = self._prefix + text
            self._prefix = ""
        self.blocks.append(Block(kind, text, breaks_before=self._break_next))
        self._break_next = False

    def _close_to(self, tag: str, *, stop_at: frozenset[str] = frozenset()) -> bool:
        """Pop open elements down to and including `tag`, not crossing `stop_at`."""
        for index in range(len(self._open) - 1, -1, -1):
            current = self._open[index]
            if current == tag:
                for closed in self._open[index:]:
                    self._forget(closed)
                del self._open[index:]
                return True
            if current in stop_at:
                return False
        return False

    def _forget(self, tag: str) -> None:
        if tag in _LISTS and self._lists:
            self._lists.pop()
        elif tag == "tr" and self._row_cells:
            self._row_cells.pop()

    # Parser callbacks -----------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._fragments.append(" " if self._in_row() else "\n")
            return
        if tag in _VOID_BREAKS:
            self._boundary()
            return
        if tag not in _BLOCKS:
            return
        if tag in ("td", "th"):
            if self._in_row():
                if self._row_cells[-1]:
                    self._fragments.append(" | ")
                self._row_cells[-1] += 1
                self._close_to("td", stop_at=frozenset({"tr"}))
                self._close_to("th", stop_at=frozenset({"tr"}))
                self._open.append(tag)
            else:
                self._fragments.append(" ")
            return
        # Elements that HTML closes implicitly when a sibling starts.
        if tag in _ITEMS:
            for item in _ITEMS:
                self._flush_if_open(item, _LISTS | {"dl"})
        elif tag == "tr":
            self._flush_if_open("tr", frozenset({"table"}))
        elif tag == "p" and self._open and self._open[-1] == "p":
            self._flush()
            self._open.pop()
        self._boundary()
        if not self._in_row() and (tag == "table" or (tag in _LISTS | {"dl"} and not self._lists)):
            self._break_next = True
        self._open.append(tag)
        if tag in _LISTS:
            start = dict(attrs).get("start") or ""
            if tag != "ol":
                self._lists.append(None)
            else:
                self._lists.append([int(start)] if start.isdigit() else [1])
        elif tag == "tr":
            self._row_cells.append(0)
        elif tag in _ITEMS:
            depth = max(len(self._lists) - 1, 0)
            indent = "  " * depth
            if tag == "li":
                counter = self._lists[-1] if self._lists else None
                if counter is None:
                    self._prefix = f"{indent}- "
                else:
                    self._prefix = f"{indent}{counter[0]}. "
                    counter[0] += 1
            else:
                self._prefix = indent + ("  " if tag == "dd" else "")

    def _flush_if_open(self, tag: str, stop_at: frozenset[str]) -> None:
        for current in reversed(self._open):
            if current == tag:
                self._flush()
                self._prefix = ""
                self._close_to(tag, stop_at=stop_at)
                return
            if current in stop_at:
                return

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._fragments.append(" " if self._in_row() else "\n")
            return
        if tag not in _BLOCKS or tag not in self._open:
            return
        if tag in ("td", "th"):
            self._close_to(tag, stop_at=frozenset({"tr"}))
            return
        if tag == "tr" or not self._in_row():
            self._flush()
        else:
            self._fragments.append(" ")
        self._close_to(tag)
        if tag in _ITEMS:
            self._prefix = ""

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        # Source line breaks are spaces in HTML; only <br> and <pre> break lines.
        self._fragments.append(data if "pre" in self._open else _SOURCE_WHITESPACE.sub(" ", data))

    def close(self) -> None:
        super().close()
        self._flush()


def html_to_blocks(markup: str) -> list[Block]:
    parser = _BlockParser()
    parser.feed(markup)
    parser.close()
    return parser.blocks


def html_to_text(markup: str) -> str:
    return render_blocks(html_to_blocks(markup))
