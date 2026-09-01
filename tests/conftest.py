import logging
import socket
from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def isolate_root_logging() -> Generator[None, None, None]:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    yield

    new_handlers = [
        handler
        for handler in root_logger.handlers
        if handler not in original_handlers
    ]
    for handler in new_handlers:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.handlers[:] = original_handlers
    root_logger.setLevel(original_level)


@pytest.fixture(autouse=True)
def disable_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    def deny_network_access(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is disabled during tests")

    monkeypatch.setattr(socket, "create_connection", deny_network_access)
    monkeypatch.setattr(socket.socket, "connect", deny_network_access)
    yield
