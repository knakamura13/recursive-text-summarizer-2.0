import socket
from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def disable_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    def deny_network_access(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is disabled during tests")

    monkeypatch.setattr(socket, "create_connection", deny_network_access)
    monkeypatch.setattr(socket.socket, "connect", deny_network_access)
    yield
