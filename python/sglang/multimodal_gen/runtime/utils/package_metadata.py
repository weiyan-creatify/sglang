# SPDX-License-Identifier: Apache-2.0
"""Answer the package-metadata probes of transformers and diffusers from one pass.

Both libraries call ``importlib.metadata.packages_distributions()`` while they
import, to learn which optional packages are installed. On each call the
standard library stats every file of every installed distribution, about a
second per library in a serving image.

Standard library only: a worker calls this before it imports anything else.
"""

from __future__ import annotations

import collections
import csv
import functools
import importlib.metadata as metadata
import inspect
import os


def _top_level_names(dist: metadata.Distribution) -> list[str]:
    """The top-level names a distribution makes importable, as importlib infers them.

    importlib skips every listed file that is missing; one check per top-level
    entry gives the same names unless part of a package was deleted by hand.
    """
    declared = (dist.read_text("top_level.txt") or "").split()
    if declared:
        return declared
    record = dist.read_text("RECORD")
    if record is None:  # egg-info installs list their files elsewhere
        paths = [str(path) for path in dist.files or ()]
    else:
        paths = [row[0] for row in csv.reader(record.splitlines()) if row]
    entries: dict[str, str] = {}
    for path in paths:
        parts = path.split("/")
        name = parts[0] if len(parts) > 1 else inspect.getmodulename(path) or path
        if "." not in name:
            entries.setdefault(name, parts[0])
    return [
        name
        for name, entry in entries.items()
        if os.path.exists(dist.locate_file(entry))
    ]


def _packages_distributions() -> dict[str, list[str]]:
    """``importlib.metadata.packages_distributions()`` without a stat per file."""
    mapping: dict[str, list[str]] = collections.defaultdict(list)
    for dist in metadata.distributions():
        for name in dict.fromkeys(_top_level_names(dist)):
            mapping[name].append(dist.metadata["Name"])
    return dict(mapping)


def cache_package_metadata() -> None:
    """Cache metadata lookups for this process; its installed packages do not change."""
    if metadata.packages_distributions is not _cached_packages_distributions:
        metadata.packages_distributions = _cached_packages_distributions
        metadata.version = functools.cache(metadata.version)


_cached_packages_distributions = functools.cache(_packages_distributions)
