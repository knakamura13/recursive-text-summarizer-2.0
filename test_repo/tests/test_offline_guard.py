import socket

import pytest

from tests.support.network_guard import (
    OfflineNetworkError,
    deny_network_access,
)


def test_create_connection_is_disabled() -> None:
    assert socket.create_connection is deny_network_access
    with pytest.raises(
        OfflineNetworkError,
        match="Network access is disabled during tests",
    ):
        socket.create_connection(("unused.invalid", 443))


def test_raw_socket_connect_is_disabled() -> None:
    assert socket.socket.connect is deny_network_access
    network_socket = socket.socket()
    try:
        with pytest.raises(
            OfflineNetworkError,
            match="Network access is disabled during tests",
        ):
            network_socket.connect(("unused.invalid", 443))
    finally:
        network_socket.close()
