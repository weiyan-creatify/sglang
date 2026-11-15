# SPDX-License-Identifier: Apache-2.0
"""Upload checkpoint weights to the GPU from a helper process while the worker imports.

The page-cache prefetch still leaves the worker a pass over the weights once
its imports are done: page cache to pinned buffers to the device. A helper
process makes that pass during the imports instead, through the same pinned
reader, and shares each file's tensors with the worker over CUDA IPC. The
worker clones them into memory of its own, so nothing depends on the helper
after it exits.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import socket
import tempfile
import threading
from multiprocessing.connection import Connection
from typing import Callable

import torch
from safetensors import safe_open

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ADDRESS_ENV = "SGLANG_DIFFUSION_WEIGHT_UPLOAD_ADDRESS"
# Arbitrary; what the helper holds, a component's parameters allocated before
# its files are taken, and the reader's lookahead must fit together.
_MAX_DEVICE_FRACTION = 0.75

KeyFilter = Callable[[str], bool]
# A component's files and, if its loader reads only some tensors, which ones.
ComponentPlan = tuple[list[str], KeyFilter | None]

_connection: Connection | None = None


def start_weight_upload(components: list[ComponentPlan]) -> None:
    """Start a helper that uploads each component's files, in the order given,
    and point this process's workers at it."""
    address = os.path.join(tempfile.mkdtemp(prefix="sglang-weight-upload-"), "socket")
    # Listening before the helper exists lets a worker connect at any time.
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(address)
        listener.listen(1)
        mp.get_context("spawn").Process(
            target=_serve,
            args=(listener, components),
            name="sglang-weight-upload",
            daemon=True,
        ).start()
    # Workers inherit it, including prestarted ones through the handoff.
    os.environ[ADDRESS_ENV] = address


class _Uploads:
    """Uploaded files waiting for the worker, held within a device memory budget."""

    def __init__(self, files: list[str], budget: float):
        self._budget = budget
        self._tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self._held_bytes = 0
        self._pending = set(files)
        self._stalled = False
        self._changed = threading.Condition()

    def add(self, path: str, tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Keep an uploaded file, and wait while holding more than the budget."""
        with self._changed:
            self._tensors[path] = tensors
            self._held_bytes += sum(tensor.nbytes for _, tensor in tensors)
            self._pending.discard(path)
            self._stalled = True
            self._changed.notify_all()
            self._changed.wait_for(lambda: self._held_bytes <= self._budget)
            self._stalled = False

    def finish(self) -> None:
        with self._changed:
            self._pending.clear()
            self._changed.notify_all()

    def take(self, path: str) -> list[tuple[str, torch.Tensor]] | None:
        """A file's tensors once uploaded; None if it will not be, or cannot be
        until the worker takes files it has not asked for."""
        with self._changed:
            self._changed.wait_for(
                lambda: (
                    path in self._tensors or path not in self._pending or self._stalled
                )
            )
            tensors = self._tensors.pop(path, None)
            if tensors is not None:
                self._held_bytes -= sum(tensor.nbytes for _, tensor in tensors)
                self._changed.notify_all()
        return tensors


def _serve(listener: socket.socket, components: list[ComponentPlan]) -> None:
    """Helper entry: upload every file, and hand each one to the worker that asks."""
    from sglang.multimodal_gen.runtime.loader.weight_readers import (
        PinnedParallelReader,
    )
    from sglang.multimodal_gen.runtime.utils.process import (
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()
    largest_component = max(sum(map(os.path.getsize, paths)) for paths, _ in components)
    free_bytes, _ = torch.cuda.mem_get_info()
    uploads = _Uploads(
        files=[path for paths, _ in components for path in paths],
        budget=_MAX_DEVICE_FRACTION * free_bytes - largest_component,
    )

    def upload() -> None:
        try:
            for paths, key_filter in components:
                # One allocation per file keeps the handoff to one IPC handle per file.
                for path, tensors in PinnedParallelReader().iter_files(
                    paths,
                    device="cuda:0",
                    key_filter=key_filter,
                    shared_allocation=True,
                ):
                    uploads.add(path, tensors)
        except Exception:
            logger.warning(
                "Weight upload failed; the worker reads the rest itself", exc_info=True
            )
        finally:
            uploads.finish()

    threading.Thread(target=upload, name="weight-upload", daemon=True).start()
    with listener:
        accepted, _ = listener.accept()
        with Connection(accepted.detach()) as connection:
            while True:
                try:
                    path = connection.recv()
                except EOFError:
                    break
                connection.send(uploads.take(path))
                # The worker has cloned and dropped the file it asked for before
                # this one; freeing it after the reply keeps that off its wait.
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        shutil.rmtree(os.path.dirname(listener.getsockname()), ignore_errors=True)


def _connect() -> Connection | None:
    global _connection
    address = envs.SGLANG_DIFFUSION_WEIGHT_UPLOAD_ADDRESS
    if _connection is None and address:
        client = socket.socket(socket.AF_UNIX)
        try:
            client.connect(address)
        except OSError:
            client.close()
            logger.warning("Weight upload helper is gone; reading weights directly")
            os.environ.pop(ADDRESS_ENV, None)
        else:
            _connection = Connection(client.detach())
    return _connection


def take_uploaded_weights(
    path: str, key_filter: KeyFilter | None
) -> list[tuple[str, torch.Tensor]] | None:
    """The tensors of ``path`` that ``key_filter`` selects, as this process's own,
    if the upload helper has them all.

    Blocks until the helper has uploaded the file. None when there is no helper
    or it did not upload what the caller reads; the caller then reads the file.
    """
    connection = _connect()
    if connection is None:
        return None
    try:
        connection.send(os.path.realpath(path))
        shared = connection.recv()
    except (EOFError, OSError):
        logger.warning("Weight upload helper is gone; reading weights directly")
        close_weight_upload()
        return None
    if shared is None:
        return None
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        wanted = {
            name
            for name in checkpoint.keys()  # noqa: SIM118
            if key_filter is None or key_filter(name)
        }
    # The helper selects tensors by the pipeline's declaration, the caller by
    # its loader; a tensor only the loader reads means reading the file.
    if not wanted.issubset(name for name, _ in shared):
        return None
    # The shared allocation is the helper's and leaves with it; a clone is
    # this process's own and aligned as torch aligns any tensor.
    cloned = [(name, tensor.clone()) for name, tensor in shared if name in wanted]
    # The copies must finish before the shared allocation is unmapped.
    torch.cuda.synchronize()
    return cloned


def close_weight_upload() -> None:
    """Let the helper exit and free what the loaders did not take."""
    global _connection
    connection = _connect()
    if connection is not None:
        connection.close()
        _connection = None
    os.environ.pop(ADDRESS_ENV, None)
