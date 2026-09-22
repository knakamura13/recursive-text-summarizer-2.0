from typing import Any, NoReturn


class OfflineNetworkError(RuntimeError):
    """Raised when a test attempts to open a network connection."""


def deny_network_access(*args: Any, **kwargs: Any) -> NoReturn:
    raise OfflineNetworkError("Network access is disabled during tests")
