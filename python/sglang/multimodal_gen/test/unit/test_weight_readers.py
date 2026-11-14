"""Which backend reads the checkpoint, and why that choice is not a boolean."""

import pytest
import torch

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

    def test_a_device_bound_load_reads_pinned_instead_of_mapping(self, monkeypatch):
        # the mapping hands the device one pageable tensor at a time
        monkeypatch.setattr(
            weight_readers.envs, "SGLANG_USE_RUNAI_MODEL_STREAMER", False
        )
        monkeypatch.setattr(
            PinnedParallelReader, "is_available", classmethod(lambda cls: True)
        )
        assert select_weight_reader(device_bound=True).name == "pinned_parallel"
        assert select_weight_reader().name == FALLBACK_READER.name

    def test_a_filtered_device_bound_load_reads_pinned(self, monkeypatch):
        # the streamer cannot skip keys, so without this it would map
        monkeypatch.setattr(
            PinnedParallelReader, "is_available", classmethod(lambda cls: True)
        )
        chosen = select_weight_reader(
            requested="runai_streamer", needs_key_filter=True, device_bound=True
        )
        assert chosen.name == "pinned_parallel"

    def test_an_explicit_mapping_request_holds_on_a_device_bound_load(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            PinnedParallelReader, "is_available", classmethod(lambda cls: True)
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
