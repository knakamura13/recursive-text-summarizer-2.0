"""Optical character recognition with the Tesseract command-line program (D4).

Pages are recognized in parallel, one single-threaded Tesseract process per
page, each bounded by a timeout. Image uploads (PNG, JPEG, TIFF) are
recognized directly; a multi-page TIFF becomes one page per frame.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from summarizer_web.config import (
    MAX_IMPORT_PAGES,
    OCR_LANGUAGE,
    OCR_PAGE_TIMEOUT_SECONDS,
    OCR_WORKERS,
)
from summarizer_web.ingestion.common import (
    Extraction,
    ImportFailure,
    PageText,
    ProgressCallback,
    clean_text,
    format_page_ranges,
    has_text,
    plural,
)
from summarizer_web.models.api import DocumentFormat, Notice

TESSERACT_INSTALL_HINT = (
    "Install Tesseract (macOS: brew install tesseract; Debian or Ubuntu: "
    "sudo apt install tesseract-ocr) and import the file again."
)


class OcrError(Exception):
    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.timed_out = timed_out


_running: set[subprocess.Popen[bytes]] = set()
_running_lock = threading.Lock()


def kill_running_ocr() -> None:
    """Kill the Tesseract processes this process started; used when an Import
    is stopped so no recognition outlives it."""
    with _running_lock:
        processes = list(_running)
    for process in processes:
        try:
            process.kill()
        except OSError:
            pass


@dataclass(frozen=True)
class Tesseract:
    executable: str
    language: str = OCR_LANGUAGE
    timeout_seconds: float = OCR_PAGE_TIMEOUT_SECONDS

    @classmethod
    def find(cls) -> Tesseract | None:
        executable = shutil.which("tesseract")
        return cls(executable) if executable else None

    def recognize_file(self, path: Path, *, frame: int | None = None) -> str:
        """Recognize an image file; `frame` selects one page of a multi-page TIFF."""
        arguments = [self.executable, str(path), "stdout", "-l", self.language]
        if frame is not None:
            arguments += ["-c", f"tessedit_page_number={frame}"]
        return self._run(arguments, None)

    def recognize_pixels(self, image: bytes, *, dpi: int) -> str:
        """Recognize a PNM image (e.g. a rendered PDF page) sent on stdin."""
        arguments = [self.executable, "stdin", "stdout", "-l", self.language, "--dpi", str(dpi)]
        return self._run(arguments, image)

    def _run(self, arguments: list[str], stdin: bytes | None) -> str:
        environment = dict(os.environ, OMP_THREAD_LIMIT="1")
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
        except OSError as error:
            raise OcrError(f"Tesseract could not be started ({error})") from error
        with _running_lock:
            _running.add(process)
        try:
            try:
                output, errors = process.communicate(stdin, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise OcrError(
                    f"OCR took longer than {self.timeout_seconds:g} s", timed_out=True
                ) from None
        finally:
            with _running_lock:
                _running.discard(process)
        if process.returncode != 0:
            lines = errors.decode("utf-8", errors="replace").strip().splitlines()
            detail = lines[-1] if lines else f"exit code {process.returncode}"
            raise OcrError(f"Tesseract failed: {detail}")
        return clean_text(output.decode("utf-8", errors="replace"))


@dataclass(frozen=True)
class OcrOutcome:
    text: str | None = None
    error: OcrError | None = None


def recognize_pages(
    pages: list[int],
    prepare: Callable[[int], Callable[[], str]],
    progress: ProgressCallback,
    *,
    workers: int = OCR_WORKERS,
) -> dict[int, OcrOutcome]:
    """Recognize `pages` in parallel.

    `prepare(page)` runs on the calling thread (PDF rendering is not
    thread-safe) and returns the recognition call for a worker thread. At most
    two prepared pages per worker wait in memory.
    """
    outcomes: dict[int, OcrOutcome] = {}
    total = len(pages)
    progress("ocr", 0, total, unit="pages")
    pending: dict[Future[str], int] = {}

    def collect(finished: set[Future[str]]) -> None:
        for future in finished:
            page = pending.pop(future)
            try:
                outcomes[page] = OcrOutcome(text=future.result())
            except OcrError as error:
                outcomes[page] = OcrOutcome(error=error)
            progress("ocr", len(outcomes), total, unit="pages")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr") as pool:
        for page in pages:
            while len(pending) >= 2 * workers:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                collect(finished)
            try:
                job = prepare(page)
            except OcrError as error:
                outcomes[page] = OcrOutcome(error=error)
                progress("ocr", len(outcomes), total, unit="pages")
                continue
            pending[pool.submit(job)] = page
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            collect(finished)
    return outcomes


def ocr_problem_notices(outcomes: Mapping[int, OcrOutcome], timeout_seconds: float) -> list[Notice]:
    timed_out = sorted(page for page, o in outcomes.items() if o.error and o.error.timed_out)
    failed = sorted(page for page, o in outcomes.items() if o.error and not o.error.timed_out)
    notices: list[Notice] = []
    if timed_out:
        notices.append(
            Notice(
                code="ocr_timeout",
                severity="warning",
                message=(
                    f"OCR took longer than {timeout_seconds:g} s on {plural(len(timed_out), 'page')} "
                    f"({format_page_ranges(timed_out)}); they stay blank."
                ),
            )
        )
    if failed:
        first = outcomes[failed[0]].error
        notices.append(
            Notice(
                code="ocr_failed",
                severity="warning",
                message=(
                    f"OCR failed on {plural(len(failed), 'page')} ({format_page_ranges(failed)}): "
                    f"{first.message if first else 'unknown error'}."
                ),
            )
        )
    return notices


def tiff_frame_count(path: Path) -> int:
    """Number of images in the main IFD chain of a TIFF or BigTIFF file."""
    with path.open("rb") as handle:
        header = handle.read(16)
        order = "<" if header[:2] == b"II" else ">"
        version = struct.unpack(order + "H", header[2:4])[0]
        if version == 43:
            offset = struct.unpack(order + "Q", header[8:16])[0]
            count_format, count_size, entry_size, next_format, next_size = "Q", 8, 20, "Q", 8
        else:
            offset = struct.unpack(order + "I", header[4:8])[0]
            count_format, count_size, entry_size, next_format, next_size = "H", 2, 12, "I", 4
        frames = 0
        seen: set[int] = set()
        while offset and offset not in seen and frames <= MAX_IMPORT_PAGES:
            seen.add(offset)
            handle.seek(offset)
            raw = handle.read(count_size)
            if len(raw) < count_size:
                break
            entries = struct.unpack(order + count_format, raw)[0]
            frames += 1
            handle.seek(offset + count_size + entries * entry_size)
            raw = handle.read(next_size)
            offset = struct.unpack(order + next_format, raw)[0] if len(raw) == next_size else 0
    return frames


def extract_image(
    path: Path,
    image_format: DocumentFormat,
    progress: ProgressCallback,
    tesseract: Tesseract | None,
) -> Extraction:
    if tesseract is None:
        raise ImportFailure(
            "Images are read with OCR, which needs Tesseract, and it is not installed. "
            + TESSERACT_INSTALL_HINT
        )
    progress("reading")
    frames = tiff_frame_count(path) if image_format == "tiff" else 1
    if frames == 0:
        raise ImportFailure("The TIFF file contains no images.")
    if frames > MAX_IMPORT_PAGES:
        raise ImportFailure(f"The TIFF file has more than {MAX_IMPORT_PAGES:,} pages, the import limit.")

    def prepare(page: int) -> Callable[[], str]:
        frame = page - 1 if frames > 1 else None
        return lambda: tesseract.recognize_file(path, frame=frame)

    outcomes = recognize_pages(list(range(1, frames + 1)), prepare, progress)
    pages: list[PageText] = []
    for number in range(1, frames + 1):
        text = outcomes[number].text
        pages.append(PageText(number, text if text and has_text(text) else "", ocr=text is not None))
    if not any(has_text(page.text) for page in pages):
        errors = [outcome.error for outcome in outcomes.values() if outcome.error is not None]
        if len(errors) == frames:
            raise ImportFailure(f"OCR could not read the image: {errors[0].message}.")
        if frames == 1:
            raise ImportFailure("No text was recognized in the image.")
        raise ImportFailure(f"No text was recognized on any of the {frames:,} pages.")
    notices = ocr_problem_notices(outcomes, tesseract.timeout_seconds)
    blank = [page.number for page in pages if not has_text(page.text)]
    if blank and frames > 1:
        notices.append(
            Notice(
                code="blank_pages",
                message=f"No text on {plural(len(blank), 'page')}: {format_page_ranges(blank)}.",
            )
        )
    return Extraction(image_format, pages=pages, notices=notices)
