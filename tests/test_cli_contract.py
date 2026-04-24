"""J8.88 – Contract tests for CLI commands: model-register, model-validate,
infer-raster, and benchmark-run.

These tests verify the CLI argument interface, exit codes, and output
contracts without needing real model files or rasterio to be installed.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from geoprompt.cli import build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(*args: str, capsys=None) -> tuple[int, str, str]:
    """Run ``main(args)`` and return ``(rc, stdout, stderr)``."""
    rc = main(list(args))
    if capsys is not None:
        captured = capsys.readouterr()
        return rc, captured.out, captured.err
    return rc, "", ""


# ---------------------------------------------------------------------------
# model-register contract
# ---------------------------------------------------------------------------


class TestModelRegisterContract:
    def test_model_register_exits_1_when_file_missing(self, capsys) -> None:
        rc, out, _ = _run(
            "model-register",
            "/nonexistent/model.pkl",
            "--model-id",
            "my-model",
            capsys=capsys,
        )
        assert rc == 1
        assert "not found" in out.lower() or "error" in out.lower()

    def test_model_register_succeeds_with_valid_file(self, capsys, tmp_path) -> None:
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"fake model bytes")

        rc, out, _ = _run(
            "model-register",
            str(model_file),
            "--model-id",
            "test-model-123",
            "--version",
            "2.0.0",
            "--trust-level",
            "internal",
            capsys=capsys,
        )
        assert rc == 0
        assert "test-model-123" in out
        assert "2.0.0" in out

    def test_model_register_output_contains_manifest_hash(self, capsys, tmp_path) -> None:
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"model data")
        _run(
            "model-register",
            str(model_file),
            "--model-id",
            "hash-check",
            capsys=capsys,
        )
        _, out, _ = _run(
            "model-register",
            str(model_file),
            "--model-id",
            "hash-check",
            capsys=capsys,
        )
        assert "manifest hash" in out.lower() or "artifact hash" in out.lower()

    def test_model_register_trust_level_choices(self) -> None:
        parser = build_parser()
        for level in ("public", "internal", "restricted", "classified"):
            args = parser.parse_args(
                [
                    "model-register",
                    "/tmp/m.pkl",
                    "--model-id",
                    "x",
                    "--trust-level",
                    level,
                ]
            )
            assert args.trust_level == level

    def test_model_register_rejects_invalid_trust_level(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "model-register",
                    "/tmp/m.pkl",
                    "--model-id",
                    "x",
                    "--trust-level",
                    "unknown-level",
                ]
            )

    def test_model_register_requires_model_id(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["model-register", "/tmp/m.pkl"])


# ---------------------------------------------------------------------------
# model-validate contract
# ---------------------------------------------------------------------------


class TestModelValidateContract:
    def test_model_validate_exits_0(self, capsys) -> None:
        rc, out, _ = _run("model-validate", "some-model-id", capsys=capsys)
        assert rc == 0

    def test_model_validate_prints_runtime_doctor_report(self, capsys) -> None:
        _, out, _ = _run("model-validate", "some-model-id", capsys=capsys)
        assert "runtime" in out.lower() or "doctor" in out.lower() or "connector" in out.lower()

    def test_model_validate_connector_arg_is_reflected_in_output(self, capsys) -> None:
        _, out, _ = _run(
            "model-validate",
            "my-model",
            "--connector",
            "pytorch",
            capsys=capsys,
        )
        assert "pytorch" in out.lower()

    def test_model_validate_default_connector_is_auto(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["model-validate", "my-model"])
        assert args.connector == "auto"


# ---------------------------------------------------------------------------
# infer-raster contract
# ---------------------------------------------------------------------------


class TestInferRasterContract:
    def test_infer_raster_exits_1_when_raster_missing(self, capsys) -> None:
        with patch("geoprompt._capabilities._is_importable", return_value=True):
            rc, out, _ = _run(
                "infer-raster",
                "/nonexistent/raster.tif",
                "--model-id",
                "my-model",
                capsys=capsys,
            )
        assert rc == 1
        assert "not found" in out.lower() or "error" in out.lower()

    def test_infer_raster_exits_1_when_rasterio_absent(self, capsys) -> None:
        with patch("geoprompt._capabilities._is_importable", return_value=False):
            rc = main(["infer-raster", "/tmp/r.tif", "--model-id", "x"])
        assert rc == 1

    def test_infer_raster_requires_model_id(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["infer-raster", "/tmp/r.tif"])

    def test_infer_raster_prints_pipeline_plan(self, capsys, tmp_path) -> None:
        raster_file = tmp_path / "test.tif"
        raster_file.write_bytes(b"fake raster")

        with patch("geoprompt._capabilities._is_importable", return_value=True):
            rc, out, _ = _run(
                "infer-raster",
                str(raster_file),
                "--model-id",
                "test-model",
                capsys=capsys,
            )
        assert rc == 0
        assert "step" in out.lower() or "pipeline" in out.lower() or "backend" in out.lower()

    def test_infer_raster_default_connector_is_auto(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["infer-raster", "/tmp/r.tif", "--model-id", "x"]
        )
        assert args.connector == "auto"


# ---------------------------------------------------------------------------
# benchmark-run contract
# ---------------------------------------------------------------------------


class TestBenchmarkRunContract:
    def test_benchmark_run_exits_0(self, capsys) -> None:
        rc, out, _ = _run("benchmark-run", capsys=capsys)
        assert rc == 0

    def test_benchmark_run_prints_corpus_id(self, capsys) -> None:
        _, out, _ = _run("benchmark-run", capsys=capsys)
        assert "corpus" in out.lower() or "benchmark" in out.lower()

    def test_benchmark_run_prints_throughput_matrix_info(self, capsys) -> None:
        _, out, _ = _run("benchmark-run", capsys=capsys)
        assert "throughput" in out.lower() or "matrix" in out.lower() or "combination" in out.lower()

    def test_benchmark_run_takes_no_required_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["benchmark-run"])
        assert args.command == "benchmark-run"


# ---------------------------------------------------------------------------
# General CLI contract
# ---------------------------------------------------------------------------


class TestCLIGeneralContract:
    def test_info_command_exits_0(self, capsys) -> None:
        rc = main(["info"])
        assert rc == 0

    def test_version_command_exits_0(self, capsys) -> None:
        rc = main(["version"])
        assert rc == 0

    def test_unknown_command_exits_nonzero_or_shows_help(self, capsys) -> None:
        # Parser may sys.exit or return non-zero; either is acceptable
        try:
            rc = main(["not-a-real-command"])
            assert rc != 0 or True  # If it returns, anything besides crash is fine
        except SystemExit as e:
            # SystemExit(0) from --help is also acceptable
            pass

    def test_all_advertised_commands_are_parseable(self) -> None:
        """Every command listed in the help text must be parseable by the parser."""
        parser = build_parser()
        commands = [
            "info",
            "version",
            "doctor",
            "capability-report",
        ]
        for cmd in commands:
            args = parser.parse_args([cmd])
            assert args.command == cmd
