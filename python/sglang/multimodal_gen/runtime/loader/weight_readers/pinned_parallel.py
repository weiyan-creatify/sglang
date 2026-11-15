# SPDX-License-Identifier: Apache-2.0
"""Parallel reads into pinned host slots, uploaded asynchronously to the device.

Built for loads that end on the GPU. Mapping a checkpoint and copying each
tensor with ``.to(device)`` reads one tensor at a time through pageable memory
and pays a page fault per 4 KiB; on a B200 host that moves about 11 GB/s even
when every page is already cached. Here worker threads ``preadv`` large pieces
of the selected tensors into a ring of pinned buffers and copy them to
device tensors on their own streams, which reaches about 45 GB/s from the page
cache and keeps several storage requests in flight when it is not.

Each selected tensor gets its own device allocation, so a caller that adopts a
yielded tensor as a parameter keeps exactly that tensor alive. ``iter_files``
can instead carve a file's tensors from one allocation, for a caller that
shares them with another process and wants one handle per file.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from collections import deque
from typing import Callable, ClassVar, Iterator

import msgspec
import torch

from sglang.multimodal_gen.runtime.loader.readonly_safetensors import _DTYPES

_PIECE_BYTES = 32 << 20
# Measured on B200: 8 to 16 readers saturate the upload from the page cache.
_READER_THREADS = 16
_PINNED_SLOTS = 16
# Bytes of device tensors allocated ahead of the consumer, which bounds the extra
# device memory a load needs beyond its own parameters. Arbitrary; at most a
# quarter of the free memory so small cards keep room for the parameters.
_LOOKAHEAD_BYTES = 12 << 30
_LOOKAHEAD_FREE_FRACTION = 0.25
# Keeps every dtype on an element boundary inside a shared allocation.
_SHARED_ALLOCATION_ALIGNMENT = 16


class _TensorSpec(msgspec.Struct, frozen=True):
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    begin: int
    end: int


class _Fragment(msgspec.Struct, frozen=True):
    tensor_index: int
    tensor_offset: int
    piece_offset: int
    length: int


class _Piece(msgspec.Struct, frozen=True):
    offset: int
    length: int
    fragments: tuple[_Fragment, ...]


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def _plan_file(
    path: str, key_filter: Callable[[str], bool] | None
) -> tuple[list[_TensorSpec], list[_Piece]] | None:
    """Selected tensors of one file and the reads that fill them, or None if a
    dtype is not supported here."""
    header, base = _read_header(path)
    specs = []
    for name, meta in header.items():
        if name == "__metadata__" or (key_filter is not None and not key_filter(name)):
            continue
        dtype = _DTYPES.get(meta["dtype"])
        if dtype is None:
            return None
        begin, end = meta["data_offsets"]
        specs.append(
            _TensorSpec(name, dtype, tuple(meta["shape"]), base + begin, base + end)
        )
    specs.sort(key=lambda spec: spec.begin)
    return specs, _plan_pieces(specs)


def _plan_pieces(specs: list[_TensorSpec]) -> list[_Piece]:
    """Contiguous runs of selected bytes, cut into pieces of at most _PIECE_BYTES."""
    pieces: list[_Piece] = []
    fragments: list[_Fragment] = []
    start = cursor = None
    for index, spec in enumerate(specs):
        position = spec.begin
        while position < spec.end:
            if start is not None and (
                position != cursor or cursor - start >= _PIECE_BYTES
            ):
                pieces.append(_Piece(start, cursor - start, tuple(fragments)))
                fragments, start = [], None
            if start is None:
                start = cursor = position
            length = min(spec.end - position, _PIECE_BYTES - (cursor - start))
            fragments.append(
                _Fragment(index, position - spec.begin, position - start, length)
            )
            position += length
            cursor = position
    if start is not None and cursor > start:
        pieces.append(_Piece(start, cursor - start, tuple(fragments)))
    return pieces


def _views_of_one_allocation(
    specs: list[_TensorSpec], device: torch.device
) -> list[torch.Tensor]:
    offsets, size = [], 0
    for spec in specs:
        offsets.append(size)
        size += spec.end - spec.begin
        size += -size % _SHARED_ALLOCATION_ALIGNMENT
    allocation = torch.empty(size, dtype=torch.uint8, device=device)
    return [
        allocation[offset : offset + spec.end - spec.begin]
        .view(spec.dtype)
        .view(spec.shape)
        for spec, offset in zip(specs, offsets)
    ]


class _FileLoad:
    def __init__(
        self,
        path: str,
        specs: list[_TensorSpec],
        device: torch.device,
        shared_allocation: bool,
    ):
        self.path = path
        self.specs = specs
        if shared_allocation:
            self.tensors = _views_of_one_allocation(specs, device)
        else:
            self.tensors = [
                torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in specs
            ]
        self.bytes = sum(spec.end - spec.begin for spec in specs)
        self.remaining = 0
        self.done = threading.Event()

    def byte_view(self, index: int) -> torch.Tensor:
        return self.tensors[index].reshape(-1).view(torch.uint8)


class _UploadPool:
    """Reader threads sharing a ring of pinned slots and a FIFO of pieces."""

    def __init__(self, device: torch.device):
        self._device = device
        self._slots = deque(
            torch.empty(_PIECE_BYTES, dtype=torch.uint8, pin_memory=True)
            for _ in range(_PINNED_SLOTS)
        )
        self._queue: deque[tuple[_FileLoad, int, _Piece] | None] = deque()
        self._cv = threading.Condition()
        self._fds: dict[str, int] = {}
        self.error: BaseException | None = None
        self._threads = [
            threading.Thread(target=self._run, name=f"weight-upload-{i}", daemon=True)
            for i in range(_READER_THREADS)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, load: _FileLoad, pieces: list[_Piece]) -> None:
        with self._cv:
            if load.path not in self._fds:
                self._fds[load.path] = os.open(load.path, os.O_RDONLY)
            load.remaining = len(pieces)
            if not pieces:
                load.done.set()
            self._queue.extend((load, self._fds[load.path], piece) for piece in pieces)
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._queue.clear()
            self._queue.extend([None] * len(self._threads))
            self._cv.notify_all()
        for thread in self._threads:
            thread.join()
        for fd in self._fds.values():
            os.close(fd)

    def _next(self):
        with self._cv:
            while not self._queue or not self._slots:
                self._cv.wait()
            job = self._queue.popleft()
            if job is None:
                return None, None
            return job, self._slots.popleft()

    def _run(self) -> None:
        stream = torch.cuda.Stream(device=self._device)
        while True:
            job, slot = self._next()
            if job is None:
                return
            load, fd, piece = job
            try:
                self._upload(load, fd, piece, slot, stream)
            except BaseException as error:  # surfaced to the consumer
                self.error = error
            finally:
                with self._cv:
                    self._slots.append(slot)
                    load.remaining -= 1
                    if load.remaining == 0 or self.error is not None:
                        load.done.set()
                    self._cv.notify_all()

    @staticmethod
    def _upload(load, fd, piece, slot, stream) -> None:
        view = memoryview(slot.numpy())
        done = 0
        while done < piece.length:
            got = os.preadv(fd, [view[done : piece.length]], piece.offset + done)
            if got <= 0:
                raise OSError(f"short read of {load.path} at {piece.offset + done}")
            done += got
        with torch.cuda.stream(stream):
            for fragment in piece.fragments:
                destination = load.byte_view(fragment.tensor_index)
                destination[
                    fragment.tensor_offset : fragment.tensor_offset + fragment.length
                ].copy_(
                    slot[
                        fragment.piece_offset : fragment.piece_offset + fragment.length
                    ],
                    non_blocking=True,
                )
        # The slot is reused as soon as this returns, so the copies must be done.
        stream.synchronize()


class PinnedParallelReader:
    name: ClassVar[str] = "pinned_parallel"
    supports_key_filter: ClassVar[bool] = True
    retains_file_mapping: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available() and hasattr(os, "preadv")

    def iter_weights(
        self,
        files: list[str],
        *,
        device: str,
        to_cpu: bool,
        key_filter: Callable[[str], bool] | None = None,
        clone_tensors: bool = True,
        show_progress: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if to_cpu:
            raise ValueError(f"{self.name} reader only loads onto a device")
        for _, tensors in self.iter_files(files, device=device, key_filter=key_filter):
            yield from tensors

    def iter_files(
        self,
        files: list[str],
        *,
        device: str,
        key_filter: Callable[[str], bool] | None = None,
        shared_allocation: bool = False,
    ) -> Iterator[tuple[str, list[tuple[str, torch.Tensor]]]]:
        """Each file with its selected tensors, in order, once they are on the device.

        With ``shared_allocation`` the tensors of a file are views of one
        allocation rather than allocations of their own.
        """
        target = torch.device(device)
        free_bytes, _ = torch.cuda.mem_get_info(target)
        budget = min(_LOOKAHEAD_BYTES, int(free_bytes * _LOOKAHEAD_FREE_FRACTION))
        plans = [(path, _plan_file(path, key_filter)) for path in files]
        pool = _UploadPool(target)
        pending: deque[_FileLoad | tuple[str, None]] = deque()
        upcoming = deque(plans)
        try:
            while upcoming or pending:
                self._admit(pool, upcoming, pending, target, budget, shared_allocation)
                head = pending.popleft()
                if isinstance(head, tuple):
                    yield head[0], list(self._fallback(head[0], target, key_filter))
                    continue
                head.done.wait()
                if pool.error is not None:
                    raise pool.error
                tensors, head.tensors = head.tensors, []
                yield (
                    head.path,
                    [(spec.name, tensor) for spec, tensor in zip(head.specs, tensors)],
                )
        finally:
            pool.close()

    @staticmethod
    def _admit(pool, upcoming, pending, target, budget, shared_allocation) -> None:
        in_flight = sum(load.bytes for load in pending if isinstance(load, _FileLoad))
        while upcoming:
            path, plan = upcoming[0]
            if plan is None:
                upcoming.popleft()
                pending.append((path, None))
                continue
            specs, pieces = plan
            size = sum(spec.end - spec.begin for spec in specs)
            if pending and in_flight + size > budget:
                return
            upcoming.popleft()
            load = _FileLoad(path, specs, target, shared_allocation)
            pool.submit(load, pieces)
            pending.append(load)
            in_flight += size

    @staticmethod
    def _fallback(path, target, key_filter) -> Iterator[tuple[str, torch.Tensor]]:
        from safetensors.torch import safe_open

        with safe_open(path, framework="pt", device=str(target)) as handle:
            for name in handle.keys():  # noqa: SIM118
                if key_filter is None or key_filter(name):
                    yield name, handle.get_tensor(name)
