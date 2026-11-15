# SPDX-License-Identifier: Apache-2.0

"""Import-safe specifications and entry points for diffusion child processes.

``multiprocessing`` unpickles a target's arguments before calling the target.
Runtime objects therefore cannot cross this boundary directly: merely passing
``ServerArgs`` used to import the diffusion configuration graph before worker
bootstrap began. ``ServerArgsPayload`` keeps that object graph opaque until the
child reaches the explicit runtime-activation phase.

``spawn`` re-executes the launching script's module scope earlier still, before
any argument is unpickled, so an offline script may bind only the
``DiffGenerator`` proxy and ``_PRE_ACTIVATION_MODULES`` at module scope; every
other diffusion import belongs inside its ``if __name__ == "__main__":`` guard.
A violation is reported rather than silently tolerated.

Keep module scope limited to the standard library and import-neutral types.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from sglang.multimodal_gen.runtime.server_args import ServerArgs

_DIFFUSION_PREFIX = "sglang.multimodal_gen."
_RUNTIME_NAMESPACES = (
    "sglang.multimodal_gen.runtime",
    "sglang.multimodal_gen.runtime.managers",
)
# What may legitimately be imported this early: bootstrap's own imports, and the
# platform and plugin modules that every plugin loads and the contract keeps
# import-safe. Listing these rather than their complement keeps the check
# complete as subpackages are added.
_PRE_ACTIVATION_MODULES = (
    "sglang.multimodal_gen.envs",
    "sglang.multimodal_gen.runtime.platforms",
    "sglang.multimodal_gen.runtime.utils",
    "sglang.multimodal_gen.runtime.managers.worker_bootstrap",
)
_MAX_REPORTED_MODULES = 5


def _warn_if_runtime_imported_early() -> None:
    """Name the modules that this child imported ahead of its own lifecycle."""
    early = sorted(
        name
        for name in list(sys.modules)
        if name.startswith(_DIFFUSION_PREFIX)
        and name not in _RUNTIME_NAMESPACES
        and not name.startswith(_PRE_ACTIVATION_MODULES)
    )
    if not early:
        return

    listed = ", ".join(early[:_MAX_REPORTED_MODULES])
    if len(early) > _MAX_REPORTED_MODULES:
        listed += f" (+{len(early) - _MAX_REPORTED_MODULES} more)"
    # In a spawned child __main__ is the re-executed launching script, which is
    # the file whose imports have to move.
    script = vars(sys.modules["__main__"]).get("__file__", "the launching script")
    logging.getLogger(__name__).warning(
        "Diffusion runtime modules were imported before this worker initialized "
        "its platform: %s. spawn re-executes %s at module scope in every child, "
        "so these were built ahead of platform initialization and hook "
        "application, and the classes and registrations they created are "
        'already past reach. Move the import inside if __name__ == "__main__": '
        "or into the function that uses it.",
        listed,
        script,
    )


@dataclass(frozen=True, slots=True)
class ServerArgsPayload:
    """A deferred ``ServerArgs`` snapshot safe to unpickle before bootstrap."""

    _pickle: bytes = field(repr=False)

    @classmethod
    def capture(cls, server_args: ServerArgs) -> ServerArgsPayload:
        return cls(pickle.dumps(server_args, protocol=pickle.HIGHEST_PROTOCOL))

    def materialize(self) -> ServerArgs:
        # Importing ServerArgs pulls in pipeline configuration modules. This
        # method must only be called after the process lifecycle is initialized.
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        server_args = pickle.loads(self._pickle)
        if not isinstance(server_args, ServerArgs):
            raise TypeError("Bootstrap payload did not contain diffusion ServerArgs")
        return server_args


@dataclass(frozen=True, slots=True)
class SchedulerProcessSpec:
    """Everything a scheduler child needs, without eagerly importing runtime state."""

    local_rank: int
    rank: int
    server_args: ServerArgsPayload
    pipe_writer: Connection


def _freeze_imported_heap() -> None:
    # What the imports created lives as long as the process. Left tracked, every
    # later full collection traces it again: 0.6 s each in a MiniMax-H3 worker.
    gc.freeze()


def bootstrap_scheduler_process(spec: SchedulerProcessSpec) -> None:
    """Initialize a child in dependency order, then invoke its worker."""
    # Arm PDEATHSIG before any vendor code runs: everything below can block,
    # and a child that hangs there would outlive a dead launcher.
    from sglang.multimodal_gen.runtime.utils.process import (
        defer_full_gc_for_imports,
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()
    _cache_package_metadata()

    # Every rank re-executes the same script, so one rank reporting is enough.
    if spec.rank == 0:
        _warn_if_runtime_imported_early()

    with defer_full_gc_for_imports():
        _initialize_worker_runtime()
        server_args = spec.server_args.materialize()

        # Resolve the function from its module after hook application. A ``from``
        # binding created earlier would retain the unpatched callable.
        from sglang.multimodal_gen.runtime.managers import gpu_worker

        _freeze_imported_heap()

    gpu_worker.run_scheduler_process(
        local_rank=spec.local_rank,
        rank=spec.rank,
        server_args=server_args,
        pipe_writer=spec.pipe_writer,
    )


def _cache_package_metadata() -> None:
    # Before anything imports transformers or diffusers, which probe it.
    from sglang.multimodal_gen.runtime.utils.package_metadata import (
        cache_package_metadata,
    )

    cache_package_metadata()


def _initialize_worker_runtime() -> None:
    """Platform, plugins and hooks, in the order every worker entry needs."""
    # Platform initialization is the first extensible runtime action. In
    # particular it precedes plugin callbacks and hook target resolution, both
    # of which may import arbitrary runtime modules.
    from sglang.multimodal_gen.runtime.platforms import initialize_current_platform

    initialize_current_platform()

    from sglang.multimodal_gen.runtime.platforms.plugins import (
        apply_plugin_hooks,
        load_plugins,
    )

    load_plugins()
    apply_plugin_hooks()


def prestart_scheduler_process(connection: Connection, warn_early: bool) -> None:
    """Import the worker runtime, then serve the spec the launcher sends.

    The imports are most of a worker's startup and need no server arguments, so
    they run while the launcher is still resolving them.
    """
    from sglang.multimodal_gen.runtime.utils.process import (
        defer_full_gc_for_imports,
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()
    _cache_package_metadata()
    if warn_early:
        _warn_if_runtime_imported_early()
    with defer_full_gc_for_imports():
        _initialize_worker_runtime()
        from sglang.multimodal_gen.runtime.managers import gpu_worker

        _freeze_imported_heap()

    handoff = connection.recv()
    if handoff is None:
        return
    spec, environ_changes = handoff
    # The launcher may set variables (allocator, NCCL) after this process started.
    for key, value in environ_changes.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    with defer_full_gc_for_imports():
        server_args = spec.server_args.materialize()
    gpu_worker.run_scheduler_process(
        local_rank=spec.local_rank,
        rank=spec.rank,
        server_args=server_args,
        pipe_writer=connection,
    )


_PRESTARTED: list[tuple[BaseProcess, Connection]] = []
_PRESTART_ENVIRON: dict[str, str] = {}


def prestart_scheduler_processes(count: int) -> None:
    """Spawn ``count`` workers now; launch_server claims them once ready to."""
    context = mp.get_context("spawn")
    _PRESTART_ENVIRON.clear()
    _PRESTART_ENVIRON.update(os.environ)
    for index in range(count):
        parent, child = context.Pipe()
        process = context.Process(
            target=prestart_scheduler_process,
            args=(child, index == 0),
            name=f"sglang-diffusionWorker-{index}",
            daemon=True,
        )
        process.start()
        child.close()
        _PRESTARTED.append((process, parent))


def claim_prestarted_scheduler_processes(
    count: int,
) -> list[tuple[BaseProcess, Connection]]:
    """The prestarted workers if they are exactly the ``count`` needed."""
    workers = list(_PRESTARTED)
    _PRESTARTED.clear()
    if len(workers) == count:
        return workers
    _release(workers)
    return []


def release_prestarted_scheduler_processes() -> None:
    """Let unclaimed prestarted workers exit; for launch paths that spawn their own."""
    workers = list(_PRESTARTED)
    _PRESTARTED.clear()
    _release(workers)


def _release(workers: list[tuple[BaseProcess, Connection]]) -> None:
    for process, connection in workers:
        try:
            connection.send(None)
        except OSError:
            pass
        connection.close()
    for process, _ in workers:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()


def environ_changes_since_prestart() -> dict[str, str | None]:
    """Variables set or removed since prestart; a removed one maps to None."""
    changes: dict[str, str | None] = {
        key: value
        for key, value in os.environ.items()
        if _PRESTART_ENVIRON.get(key) != value
    }
    changes.update({key: None for key in _PRESTART_ENVIRON if key not in os.environ})
    return changes


def bootstrap_http_server_process(
    server_args: ServerArgsPayload, worker_pids: tuple[int, ...] = ()
) -> None:
    from sglang.multimodal_gen.runtime.utils.process import (
        defer_full_gc_for_imports,
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()

    _warn_if_runtime_imported_early()

    with defer_full_gc_for_imports():
        # No initialize_current_platform() here: this child serves HTTP and never
        # touches the device, so it has no reason to bring up a vendor backend.
        from sglang.multimodal_gen.runtime.platforms.plugins import (
            apply_plugin_hooks,
            load_plugins,
        )

        load_plugins()
        apply_plugin_hooks()

        from sglang.multimodal_gen.runtime import launch_server

        materialized = server_args.materialize()

    launch_server.launch_http_server_only(materialized, worker_pids=worker_pids)
