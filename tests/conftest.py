"""Keep the suite offline.

Several modules build a LangSmith `Client` at import (`atlas/monitor.py`'s
`client = Client()`, as Chapter 21 prints it), and a `Client` starts a
background thread that fetches `/info` from its endpoint. Without this file
that request goes to the hosted LangSmith API on every test run.

This file runs before any test module is imported, which matters: the
LangSmith SDK reads its environment variables once per process and caches
them (`langsmith.utils.get_env_var` is `lru_cache`d), so they must be set
before the first client is built.

Unless `LANGSMITH_API_KEY` is set (the opt-in for the skip-guarded live
tests), it:

- switches tracing off and points LangSmith at a refused loopback port, so a
  stray client fails on the machine instead of reaching the network;
- guards sockets: a connect or DNS lookup for any non-loopback address is
  refused and recorded, and the run's summary reports what was blocked.

Tests that exercise tracing (`tests/test_tracing.py`) install their own
capturing session; nothing here changes what they assert.
"""

import ipaddress
import os
import socket

import pytest

REFUSED_ENDPOINT = "http://127.0.0.1:9"  # nothing listens on the discard port
OFFLINE = not os.environ.get("LANGSMITH_API_KEY")

blocked: list[str] = []


def _is_local(host: object) -> bool:
    if host is None:
        return True
    name = host.decode() if isinstance(host, bytes) else str(host)
    if name in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(name.split("%")[0]).is_loopback
    except ValueError:
        return False


if OFFLINE:
    for name, value in {
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_ENDPOINT": REFUSED_ENDPOINT,
        "LANGCHAIN_ENDPOINT": REFUSED_ENDPOINT,
    }.items():
        os.environ[name] = value

    _connect = socket.socket.connect
    _connect_ex = socket.socket.connect_ex
    _getaddrinfo = socket.getaddrinfo

    def _guard(address: object) -> None:
        if isinstance(address, tuple) and not _is_local(address[0]):
            blocked.append(f"connect {address[0]}:{address[1]}")
            raise ConnectionRefusedError(f"test suite is offline: {address[0]}")

    def connect(self: socket.socket, address: object) -> None:
        _guard(address)
        return _connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        _guard(address)
        return _connect_ex(self, address)

    def getaddrinfo(host: object, *args: object, **kwargs: object) -> list:
        if not _is_local(host):
            blocked.append(f"dns {host}")
            raise socket.gaierror(socket.EAI_NONAME, "test suite is offline")
        return _getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if not OFFLINE:
        terminalreporter.write_line("network guard: off (LANGSMITH_API_KEY is set)")
        return
    terminalreporter.write_line(
        f"network guard: {len(blocked)} non-loopback attempt(s) blocked"
        + (f": {sorted(set(blocked))}" if blocked else "")
    )
