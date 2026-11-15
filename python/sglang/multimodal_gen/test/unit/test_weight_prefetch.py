"""Startup page-cache prefetch: what the helpers read, and when nothing starts."""

import os
import subprocess
import threading
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.loader import weight_prefetch


@pytest.fixture(autouse=True)
def _no_running_prefetch(monkeypatch):
    monkeypatch.setattr(weight_prefetch, "_planned_bytes", 0)
    monkeypatch.setattr(weight_prefetch, "_done_callbacks", [])
    monkeypatch.setattr(weight_prefetch, "_result", None)


def test_chunks_tile_every_file_exactly(monkeypatch):
    """Helpers read by chunk, so a gap would leave part of a file cold and an
    overlap would read it twice, both without any error."""
    monkeypatch.setattr(weight_prefetch, "_CHUNK_BYTES", 16)
    files = [("a", 40), ("b", 16), ("c", 0), ("d", 5)]
    chunks = weight_prefetch._chunks(files)
    for path, size in files:
        cursor = 0
        for offset, length in sorted((o, n) for p, o, n in chunks if p == path):
            assert offset == cursor
            assert 0 < length <= 16
            cursor += length
        assert cursor == size


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="requires os.preadv")
def test_helpers_run_without_site_packages_and_read_every_byte(tmp_path, monkeypatch):
    """Helpers start with -S: an import from outside the standard library would
    make them exit without reading. The bytes they report must add up to the
    files exactly, across helpers and across chunks of one file."""
    monkeypatch.setattr(weight_prefetch, "_host_memory_headroom_bytes", lambda: 1 << 40)
    monkeypatch.setattr(weight_prefetch, "_PROCESSES", 3)
    sizes = [(1 << 20) + 7, 3, 70 << 20]
    paths = []
    for index, size in enumerate(sizes):
        path = tmp_path / f"shard-{index}.safetensors"
        with open(path, "wb") as f:
            f.truncate(size)
        paths.append(str(path))

    done = threading.Event()
    reported = []

    def on_done(bytes_read, seconds):
        reported.append(bytes_read)
        done.set()

    files = {"transformer": paths[:2], "vae": paths[2:]}
    assert weight_prefetch.start_weight_prefetch(files, on_done=on_done) == sum(sizes)
    assert done.wait(timeout=120)
    assert reported == [sum(sizes)]

    # The launcher asks again once the server arguments resolve: nothing is read twice.
    with patch.object(subprocess, "Popen") as popen:
        assert weight_prefetch.start_weight_prefetch(files, on_done=on_done) == sum(
            sizes
        )
    popen.assert_not_called()
    assert reported == [sum(sizes)] * 2


def test_files_beyond_the_host_headroom_start_nothing(tmp_path, monkeypatch):
    """Reading more than memory can hold would evict the pages it just read."""
    monkeypatch.setattr(weight_prefetch, "_host_memory_headroom_bytes", lambda: 100)
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"x" * 90)

    with patch.object(subprocess, "Popen") as popen:
        assert weight_prefetch.start_weight_prefetch({"vae": [str(path)]}) == 0
    popen.assert_not_called()


def test_plan_follows_the_pipeline_index_and_local_overrides(tmp_path):
    """Null components, directories the index does not declare and components
    overridden with a non-local path are not read."""
    for name in ("transformer", "vae", "unlisted"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.safetensors").write_bytes(b"x")
    override = tmp_path / "encoder.safetensors"
    override.write_bytes(b"xx")
    (tmp_path / "model_index.json").write_text(
        '{"_class_name": "Pipeline", "transformer": ["diffusers", "DiT"],'
        ' "vae": ["diffusers", "VAE"], "text_encoder": ["transformers", "Encoder"],'
        ' "image_encoder": [null, null], "scheduler": ["diffusers", "Scheduler"]}'
    )
    files = weight_prefetch.pipeline_weight_files(
        str(tmp_path),
        revision=None,
        model_subfolder=None,
        model_variant=None,
        component_overrides={"vae": "owner/repo/vae", "text_encoder": str(override)},
    )

    # Largest first: the loaders take components in that order.
    assert list(files.items()) == [
        ("text_encoder", [str(override)]),
        ("transformer", [str(tmp_path / "transformer" / "model.safetensors")]),
        ("vae", []),
        ("scheduler", []),
    ]


def test_plan_reads_a_partition_only_when_the_arguments_name_it(tmp_path):
    """A repository of partitions holds the same model several times; the
    pipeline class picks one, so a guess would read weights nobody loads."""
    index = '{"_class_name": "Pipeline", "transformer": ["diffusers", "DiT"]}'
    (tmp_path / "model_index.json").write_text(index)
    for partition in ("FL2VA", "Ref2VA"):
        weights = tmp_path / partition / "transformer" / "source"
        weights.mkdir(parents=True)
        (weights / "model.safetensors").write_bytes(b"x")
        (tmp_path / partition / "model_index.json").write_text(index)

    def plan(*, model_subfolder=None, model_variant=None):
        return weight_prefetch.pipeline_weight_files(
            str(tmp_path),
            revision=None,
            model_subfolder=model_subfolder,
            model_variant=model_variant,
            component_overrides={},
        )

    nested = str(tmp_path / "FL2VA" / "transformer" / "source" / "model.safetensors")
    assert plan(model_variant="fl2va") == {"transformer": [nested]}
    assert plan(model_subfolder="FL2VA") == {"transformer": [nested]}
    assert plan() == {}
    assert plan(model_variant="hybrid") == {}
