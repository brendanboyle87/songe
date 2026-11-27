"""A supervised llama-server releases its memory after its wrapper exits."""

import asyncio
import os
import sys
from types import SimpleNamespace

import psutil
import pytest

from songe.config import Config
from songe.memory import kill, owned_command, process_options
from songe.upstream.supervisor import LlamaServer


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group termination")
async def test_parking_waits_for_server_process(tmp_path):
    released = tmp_path / "released"
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
print(os.getpid(), flush=True)
signal.pause()
"""
    server = LlamaServer(Config(upstream="llamacpp"), tmp_path)
    server.platform = SimpleNamespace(name="darwin")
    process = await asyncio.create_subprocess_exec(
        *owned_command([sys.executable, "-c", command, str(released)]),
        **process_options(),
        stdout=asyncio.subprocess.PIPE,
    )
    server.process = process
    worker = None
    try:
        ready = await asyncio.wait_for(process.stdout.readline(), timeout=5)
        worker = psutil.Process(int(ready))
        await server.stop()
        assert released.exists()
    finally:
        if process.returncode is None:
            kill(process)
            await process.wait()
        if worker is not None and worker.is_running():
            await asyncio.to_thread(psutil.wait_procs, [worker], timeout=3)
