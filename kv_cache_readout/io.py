from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import uuid
import socket
import time

import torch

from .config import ROOT, file_hash


def pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            error=ctypes.get_last_error()
            if error==87:return False
            raise PermissionError(f'Cannot inspect PID {pid}, Windows error {error}')
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
                raise PermissionError(f'Cannot read exit status for PID {pid}')
            return code.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    return True


def atomic_json(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def atomic_torch(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def read_checkpoint(path: str | Path) -> dict:
    try:
        value = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(Path(path), map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    return value


def code_identity() -> dict:
    files = {}
    for folder in (ROOT / "kv_cache_readout", ROOT / "e2e_cache", ROOT / "e2e_workload",
                   ROOT / "runtime_fixes", ROOT / "scripts"):
        if folder.is_dir():
            for path in sorted(folder.glob("*.py")):
                files[str(path.relative_to(ROOT)).replace("\\", "/")] = file_hash(path)
    return files


@contextlib.contextmanager
def run_lock(directory: str | Path):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    owner = dict(pid=os.getpid(), host=socket.gethostname(), token=uuid.uuid4().hex)
    if lock.exists():
        try:
            raw = json.loads(lock.read_text(encoding='utf-8'))
            prior = raw if isinstance(raw, dict) else dict(pid=int(raw), host=owner['host'])
        except (ValueError, OSError):
            raise RuntimeError(f'Unreadable lock: {lock}; inspect its owner before removing')
        if prior.get('host') != owner['host']:
            raise RuntimeError(f'Lock belongs to another host: {prior}')
        pid = prior.get('pid', 0)
        if type(pid) is not int or pid <= 1:
            raise RuntimeError(f'Invalid lock owner: {prior}')
        try:
            alive=pid_alive(pid)
        except PermissionError:
            raise RuntimeError(f'Lock owner cannot be inspected: {prior}')
        if not alive:
            # Serialize stale-owner recovery using a second exclusive guard.
            guard = root / '.lock-recovery'
            try:
                fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise RuntimeError(f'Another launcher is recovering {lock}')
            try:
                os.close(fd)
                current = json.loads(lock.read_text(encoding='utf-8'))
                if current == raw:
                    lock.unlink()
            finally:
                guard.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Run already active, PID={pid}, directory={root}")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Run is already locked: {root}") from exc
    try:
        os.write(descriptor, json.dumps(owner).encode())
        os.close(descriptor)
        yield
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) == owner:
            lock.unlink()

