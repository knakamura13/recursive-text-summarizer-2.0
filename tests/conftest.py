import socket
from collections.abc import Generator

import pytest

from tests.support.legacy_loader import isolated_root_logging
from tests.support.network_guard import (
    OfflineNetworkError,
    deny_network_access,
)


@pytest.fixture(autouse=True)
def isolate_root_logging() -> Generator[None, None, None]:
    with isolated_root_logging():
        yield


@pytest.fixture(autouse=True)
def disable_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    monkeypatch.setattr(socket, "create_connection", deny_network_access)
    monkeypatch.setattr(socket.socket, "connect", deny_network_access)
    yield
