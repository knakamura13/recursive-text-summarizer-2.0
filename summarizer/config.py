from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    input_path: Path = Path("input.txt")
    output_path: Path = Path("output.txt")
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 180

    def __post_init__(self) -> None:
        if self.input_path == Path("."):
            raise ValueError("input_path must name a file")
        if self.output_path == Path("."):
            raise ValueError("output_path must name a file")
        if self.input_path.resolve(strict=False) == self.output_path.resolve(
            strict=False
        ):
            raise ValueError("output_path must differ from input_path")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_delay_seconds: float = 1
    backoff_multiplier: float = 2

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if (
            not isfinite(self.initial_delay_seconds)
            or self.initial_delay_seconds <= 0
        ):
            raise ValueError("initial_delay_seconds must be positive")
        if (
            not isfinite(self.backoff_multiplier)
            or self.backoff_multiplier <= 0
        ):
            raise ValueError("backoff_multiplier must be positive")


@dataclass(frozen=True)
class LegacyWorkflowConfig:
    chunk_size: int = 1000
    max_chunks: int = -1
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_chunks == 0 or self.max_chunks < -1:
            raise ValueError("max_chunks must be -1 or a positive integer")
