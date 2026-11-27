"""A supervised llama-server releases its memory after its wrapper exits."""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psutil
import pytest

from songe.config import Config
from songe.memory import kill, owned_command, process_options
from songe.upstream import supervisor


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group termination")
async def test_parking_waits_for_server_process(tmp_path):
    released = tmp_path / "released"
    pid_file = tmp_path / "llama-server.pid"
    command = """
import os
from pathlib import Path
import signal
import sys
import time

def stop(*_):
    time.sleep(0.25)
    Path(sys.argv[1]).touch()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
Path(sys.argv[2]).write_text(str(os.getpid()), encoding="ascii")
signal.pause()
"""
    server = supervisor.LlamaServer(Config(upstream="llamacpp"), tmp_path)
    server.platform = SimpleNamespace(name="darwin")
    process = await asyncio.create_subprocess_exec(
        *owned_command([sys.executable, "-c", command, str(released), str(pid_file)]),
        **process_options(),
        stdout=asyncio.subprocess.DEVNULL,
    )
    server.process = process
    worker = None
    try:
        async with asyncio.timeout(5):
            while True:
                if pid_file.exists() and (ready := pid_file.read_text(encoding="ascii")):
                    break
                await asyncio.sleep(0.01)
        worker = psutil.Process(int(ready))
        await server.stop()
        assert released.exists()
    finally:
        if process.returncode is None:
            kill(process)
            await process.wait()
        if worker is not None and worker.is_running():
            await asyncio.to_thread(psutil.wait_procs, [worker], timeout=3)


async def test_parking_waits_past_grace_after_forced_stop(tmp_path, monkeypatch):
    config = Config(upstream="llamacpp")
    server = supervisor.LlamaServer(config, tmp_path)
    server.platform = SimpleNamespace(name="darwin")
    server.process = SimpleNamespace(pid=123, returncode=None, wait=AsyncMock())
    server.watcher = asyncio.get_running_loop().create_future()
    server.watcher.set_result(None)
    child = object()
    calls = []

    def wait_procs(processes, timeout=None):
        assert processes == [child]
        assert not server.finished.is_set()
        calls.append(("wait", timeout))
        # The child outlives any grace-period wait, even after the kill.
        return (processes, []) if timeout is None else ([], processes)

    monkeypatch.setattr(
        psutil, "Process", lambda pid: SimpleNamespace(children=lambda *, recursive: [child])
    )
    monkeypatch.setattr(psutil, "wait_procs", wait_procs)
    monkeypatch.setattr(supervisor, "terminate", lambda process: calls.append("terminate"))
    monkeypatch.setattr(supervisor, "kill", lambda process: calls.append("kill"))

    await server.stop()

    assert calls == [
        "terminate",
        ("wait", config.guard_grace_seconds),
        "kill",
        ("wait", None),
    ]
    assert server.finished.is_set()
    assert server.process is None and server.watcher is None
