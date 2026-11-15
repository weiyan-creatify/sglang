# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from huggingface_hub.errors import (
    GatedRepoError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

from sglang.cli.utils import get_is_diffusion_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _hub_error(error_type, status_code):
    request = httpx.Request("GET", "https://huggingface.co/test/model")
    response = httpx.Response(status_code, request=request)
    return error_type("test error", response=response)


class TestDiffusionModelDetection(CustomTestCase):
    def setUp(self):
        super().setUp()
        for target in (
            "sglang.cli.utils._is_diffusion_model_from_registry",
            "sglang.cli.utils._is_overlay_diffusion_model",
            "sglang.cli.utils.os.path.isdir",
        ):
            patcher = patch(target, return_value=False)
            patcher.start()
            self.addCleanup(patcher.stop)

        modelscope_patcher = patch(
            "sglang.cli.utils.envs.SGLANG_USE_MODELSCOPE.get", return_value=False
        )
        modelscope_patcher.start()
        self.addCleanup(modelscope_patcher.stop)

    def test_an_index_on_disk_decides_without_the_registry(self):
        """Importing the registry takes seconds, during which `sglang serve`
        cannot start the diffusion workers."""
        registry = patch(
            "sglang.cli.utils._is_diffusion_model_from_registry",
            side_effect=AssertionError("registry consulted"),
        )
        with tempfile.TemporaryDirectory() as model_dir, registry:
            index_path = os.path.join(model_dir, "model_index.json")
            with open(index_path, "w") as f:
                json.dump({"_diffusers_version": "0.32.2"}, f)

            with patch("sglang.cli.utils.os.path.isdir", return_value=True):
                self.assertTrue(get_is_diffusion_model(model_dir))
            with patch(
                "huggingface_hub.try_to_load_from_cache", return_value=index_path
            ):
                self.assertTrue(get_is_diffusion_model("test/cached-diffusion-model"))

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_gated_diffusion_repo_uses_metadata_fallback(
        self, mock_download, mock_hf_api
    ):
        mock_download.side_effect = _hub_error(GatedRepoError, 401)
        mock_hf_api.return_value.model_info.return_value = SimpleNamespace(
            library_name="diffusers"
        )

        self.assertTrue(get_is_diffusion_model("test/gated-diffusion-model"))
        mock_hf_api.return_value.model_info.assert_called_once()

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_gated_non_diffusion_repo_is_not_detected(self, mock_download, mock_hf_api):
        mock_download.side_effect = _hub_error(GatedRepoError, 401)
        mock_hf_api.return_value.model_info.return_value = SimpleNamespace(
            library_name="transformers"
        )

        self.assertFalse(get_is_diffusion_model("test/gated-llm-model"))
        mock_hf_api.return_value.model_info.assert_called_once()

    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_non_gated_hub_errors_do_not_use_metadata_fallback(
        self, mock_download, mock_hf_api
    ):
        for error_type in (RemoteEntryNotFoundError, RepositoryNotFoundError):
            with self.subTest(error_type=error_type.__name__):
                mock_download.side_effect = _hub_error(error_type, 404)
                self.assertFalse(get_is_diffusion_model("test/llm-model"))
        mock_hf_api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
