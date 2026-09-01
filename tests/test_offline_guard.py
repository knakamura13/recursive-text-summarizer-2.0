import socket

import pytest


def test_network_access_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("example.com", 443))
