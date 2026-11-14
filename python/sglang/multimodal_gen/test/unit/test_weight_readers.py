"""Which backend reads the checkpoint, and why that choice is not a boolean."""

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader import weight_readers
from sglang.multimodal_gen.runtime.loader.weight_readers import (
    FALLBACK_READER,
    PinnedParallelReader,
    RunaiStreamerReader,
    SafetensorsMmapReader,
    available_reader_names,
    pinned_parallel,
    select_weight_reader,
)


class TestCapabilities:
    def test_the_streamer_cannot_skip_keys(self):
        # it materializes every tensor before yielding any of them
        assert not RunaiStreamerReader.supports_key_filter

    def test_only_the_mapping_backend_leaves_pages_reclaimable(self):
        assert SafetensorsMmapReader.retains_file_mapping
        assert not RunaiStreamerReader.retains_file_mapping
        assert not PinnedParallelReader.retains_file_mapping

    def test_the_fallback_is_always_available(self):
        assert FALLBACK_READER.is_available()
        assert FALLBACK_READER.name in available_reader_names()


class TestSelection:
    def test_an_explicit_request_is_honoured(self):
        assert select_weight_reader(requested="safetensors").name == "safetensors"

    def test_an_unknown_name_is_an_error_not_a_silent_fallback(self):
        with pytest.raises(ValueError, match="unknown weight reader"):
            select_weight_reader(requested="does_not_exist")

    def test_a_key_filter_passes_over_a_backend_that_cannot_filter(self):
        # reading the whole checkpoint to discard most of it is worse than
        # reading the requested part more slowly
        chosen = select_weight_reader(requested="runai_streamer", needs_key_filter=True)
        assert chosen.name == FALLBACK_READER.name

    def test_a_key_filter_leaves_a_capable_backend_alone(self):
        chosen = select_weight_reader(requested="safetensors", needs_key_filter=True)
        assert chosen.name == "safetensors"

    def test_an_unavailable_backend_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            RunaiStreamerReader, "is_available", classmethod(lambda cls: False)
        )
        assert (
            select_weight_reader(requested="runai_streamer").name
            == FALLBACK_READER.name
        )

    def test_the_environment_decides_when_nothing_is_requested(self, monkeypatch):
        monkeypatch.setattr(
            weight_readers.envs, "SGLANG_USE_RUNAI_MODEL_STREAMER", False
        )
        assert select_weight_reader().name == FALLBACK_READER.name

    def test_the_environment_can_ask_for_the_streamer(self, monkeypatch):
        if not RunaiStreamerReader.is_available():
            pytest.skip("run:ai model streamer is not installed")
        monkeypatch.setattr(
            weight_readers.envs, "SGLANG_USE_RUNAI_MODEL_STREAMER", True
        )
        assert select_weight_reader().name == "runai_streamer"

    def test_the_switch_reads_device_bound_loads_pinned(self, monkeypatch):
        monkeypatch.setattr(
            weight_readers.envs, "SGLANG_DIFFUSION_USE_PINNED_WEIGHT_READER", True
        )
        monkeypatch.setattr(
            PinnedParallelReader, "is_available", classmethod(lambda cls: True)
        )
        assert select_weight_reader(device_bound=True).name == "pinned_parallel"
        # it only loads onto a device
        assert select_weight_reader().name != "pinned_parallel"
        # it can skip keys, unlike the streamer
        chosen = select_weight_reader(needs_key_filter=True, device_bound=True)
        assert chosen.name == "pinned_parallel"

    def test_without_the_switch_device_bound_loads_are_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            PinnedParallelReader, "is_available", classmethod(lambda cls: True)
        )
        for runai in (True, False):
            monkeypatch.setattr(
                weight_readers.envs, "SGLANG_USE_RUNAI_MODEL_STREAMER", runai
            )
            assert (
                select_weight_reader(device_bound=True).name
                == select_weight_reader().name
            )

    def test_an_explicit_request_holds_over_the_switch(self, monkeypatch):
        monkeypatch.setattr(
            weight_readers.envs, "SGLANG_DIFFUSION_USE_PINNED_WEIGHT_READER", True
        )
        chosen = select_weight_reader(requested="safetensors", device_bound=True)
        assert chosen.name == "safetensors"


class TestPinnedReadPlan:
    def test_pieces_cover_each_selected_byte_once_and_nothing_else(self, monkeypatch):
        """Tensors are filled from pieces by offset, so a gap or an overlap in
        the plan would leave bytes unwritten or written twice without an error."""
        monkeypatch.setattr(pinned_parallel, "_PIECE_BYTES", 16)

        def spec(name, begin, end):
            return pinned_parallel._TensorSpec(
                name, torch.uint8, (end - begin,), begin, end
            )

        # two adjacent tensors, an unselected gap, then one longer than a piece
        specs = [spec("a", 8, 20), spec("b", 20, 30), spec("c", 64, 120)]
        pieces = pinned_parallel._plan_pieces(specs)

        covered = {index: [] for index in range(len(specs))}
        for piece in pieces:
            assert 0 < piece.length <= 16
            assert piece.offset + piece.length <= 30 or piece.offset >= 64
            for fragment in piece.fragments:
                assert fragment.piece_offset + fragment.length <= piece.length
                tensor = specs[fragment.tensor_index]
                assert (
                    piece.offset + fragment.piece_offset
                    == tensor.begin + fragment.tensor_offset
                )
                covered[fragment.tensor_index].append(
                    (fragment.tensor_offset, fragment.length)
                )
        for index, tensor in enumerate(specs):
            cursor = 0
            for offset, length in sorted(covered[index]):
                assert offset == cursor
                cursor += length
            assert cursor == tensor.end - tensor.begin


@pytest.mark.skipif(not PinnedParallelReader.is_available(), reason="needs CUDA")
def test_pinned_reads_match_safe_open(tmp_path, monkeypatch):
    # Small pieces, so a tensor spans several pieces and a piece several tensors.
    monkeypatch.setattr(pinned_parallel, "_PIECE_BYTES", 4096)
    generator = torch.Generator().manual_seed(0)
    files = {
        "first.safetensors": {
            "large": torch.randn(3, 1000, generator=generator).to(torch.bfloat16),
            "skipped": torch.randn(700, generator=generator),
            "small": torch.arange(10, dtype=torch.int64),
            "empty": torch.empty(0, 4),
            "after_gap": torch.randn(7, 33, generator=generator),
        },
        # a dtype the pinned reader does not decode goes through safe_open
        "fallback.safetensors": {"unsigned": torch.arange(9).to(torch.uint16)},
    }
    paths = []
    for name, tensors in files.items():
        paths.append(str(tmp_path / name))
        save_file(tensors, paths[-1])

    got = dict(
        PinnedParallelReader().iter_weights(
            paths, device="cuda", to_cpu=False, key_filter=lambda n: n != "skipped"
        )
    )

    expected = {
        name: tensor
        for tensors in files.values()
        for name, tensor in tensors.items()
        if name != "skipped"
    }
    assert got.keys() == expected.keys()
    for name, tensor in expected.items():
        assert got[name].device.type == "cuda", name
        assert got[name].dtype == tensor.dtype, name
        assert torch.equal(got[name].cpu(), tensor), name
