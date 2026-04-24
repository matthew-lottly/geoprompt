"""J8.94 – Deterministic tests for auto/adaptive chunking.

These tests verify that chunking parameters produce consistent, predictable
results across platforms and that throughput optimizations never change
result semantics (row count, data integrity, ordering).

Key invariants:
  1. Total rows out == total rows in (no loss, no duplication).
  2. Row order is preserved within each chunk and across all chunks.
  3. Chunk sizes respect the configured chunk_size boundary.
  4. WORKLOAD_PRESETS have stable, documented values.
  5. limit_rows caps total output regardless of chunk_size.
"""
from __future__ import annotations

from geoprompt.io import WORKLOAD_PRESETS, _iter_frame_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rows(n: int) -> list[dict]:
    return [
        {
            "id": i,
            "value": float(i),
            "geometry": {"type": "Point", "coordinates": [float(i), 0.0]},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# J8.94.1 – WORKLOAD_PRESETS are stable and deterministic across platforms
# ---------------------------------------------------------------------------


class TestWorkloadPresetsAreStable:
    """WORKLOAD_PRESETS values must be deterministic (not runtime-computed)."""

    def test_presets_exist(self) -> None:
        for name in ("small", "medium", "large", "huge"):
            assert name in WORKLOAD_PRESETS, f"Missing preset: {name!r}"

    def test_small_preset_chunk_size(self) -> None:
        assert WORKLOAD_PRESETS["small"]["chunk_size"] == 5000

    def test_medium_preset_chunk_size(self) -> None:
        assert WORKLOAD_PRESETS["medium"]["chunk_size"] == 20000

    def test_large_preset_chunk_size(self) -> None:
        assert WORKLOAD_PRESETS["large"]["chunk_size"] == 50000

    def test_huge_preset_chunk_size(self) -> None:
        assert WORKLOAD_PRESETS["huge"]["chunk_size"] == 100000

    def test_huge_preset_sample_step(self) -> None:
        assert WORKLOAD_PRESETS["huge"]["sample_step"] == 2

    def test_small_preset_limit_rows(self) -> None:
        assert WORKLOAD_PRESETS["small"]["limit_rows"] == 100000

    def test_medium_through_huge_have_no_limit(self) -> None:
        for name in ("medium", "large", "huge"):
            assert WORKLOAD_PRESETS[name]["limit_rows"] is None, (
                f"Preset {name!r} should have no row limit"
            )

    def test_all_presets_have_sample_step_gte_1(self) -> None:
        for name, preset in WORKLOAD_PRESETS.items():
            assert preset["sample_step"] >= 1, f"{name!r} has invalid sample_step"

    def test_chunk_sizes_are_strictly_increasing(self) -> None:
        sizes = [
            WORKLOAD_PRESETS["small"]["chunk_size"],
            WORKLOAD_PRESETS["medium"]["chunk_size"],
            WORKLOAD_PRESETS["large"]["chunk_size"],
            WORKLOAD_PRESETS["huge"]["chunk_size"],
        ]
        for a, b in zip(sizes, sizes[1:]):
            assert a < b, f"Chunk sizes not strictly increasing: {sizes}"


# ---------------------------------------------------------------------------
# J8.94.2 – _iter_frame_chunks: row count invariant
# ---------------------------------------------------------------------------


class TestChunkingRowCountInvariant:
    """Total rows across all chunks equals total input rows (no loss or dup)."""

    def test_exact_multiple_of_chunk_size(self) -> None:
        rows = _make_rows(100)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        total = sum(len(c) for c in chunks)
        assert total == 100

    def test_non_multiple_of_chunk_size(self) -> None:
        rows = _make_rows(47)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        total = sum(len(c) for c in chunks)
        assert total == 47

    def test_single_row(self) -> None:
        rows = _make_rows(1)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        assert sum(len(c) for c in chunks) == 1

    def test_empty_input(self) -> None:
        chunks = list(
            _iter_frame_chunks([], geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        assert chunks == []

    def test_chunk_size_larger_than_input(self) -> None:
        rows = _make_rows(5)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=100, limit_rows=None)
        )
        assert len(chunks) == 1
        assert len(chunks[0]) == 5


# ---------------------------------------------------------------------------
# J8.94.3 – _iter_frame_chunks: chunk boundary enforcement
# ---------------------------------------------------------------------------


class TestChunkingBoundaryEnforcement:
    """No chunk exceeds chunk_size (except the final partial chunk)."""

    def test_no_chunk_exceeds_size(self) -> None:
        rows = _make_rows(99)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        for i, chunk in enumerate(chunks[:-1]):  # all except last
            assert len(chunk) == 10, f"Chunk {i} size {len(chunk)} != 10"

    def test_final_chunk_is_remainder(self) -> None:
        rows = _make_rows(99)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        assert len(chunks[-1]) == 9  # 99 % 10 == 9

    def test_chunk_count_is_ceil_division(self) -> None:
        rows = _make_rows(99)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        import math
        assert len(chunks) == math.ceil(99 / 10)


# ---------------------------------------------------------------------------
# J8.94.4 – _iter_frame_chunks: row order invariant
# ---------------------------------------------------------------------------


class TestChunkingOrderInvariant:
    """Row order must be preserved — chunking must not reorder data."""

    def test_rows_are_in_original_order(self) -> None:
        rows = _make_rows(35)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        all_ids = [row["id"] for chunk in chunks for row in chunk.to_records()]
        assert all_ids == list(range(35)), f"Row order changed: {all_ids[:10]}..."

    def test_value_column_preserved(self) -> None:
        rows = _make_rows(25)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=None)
        )
        all_values = [row["value"] for chunk in chunks for row in chunk.to_records()]
        expected = [float(i) for i in range(25)]
        assert all_values == expected


# ---------------------------------------------------------------------------
# J8.94.5 – limit_rows caps output deterministically
# ---------------------------------------------------------------------------


class TestChunkingLimitRows:
    """limit_rows must cap total output exactly, regardless of chunk_size."""

    def test_limit_rows_caps_output(self) -> None:
        rows = _make_rows(100)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=25)
        )
        total = sum(len(c) for c in chunks)
        assert total == 25

    def test_limit_rows_less_than_chunk_size(self) -> None:
        rows = _make_rows(100)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=50, limit_rows=10)
        )
        total = sum(len(c) for c in chunks)
        assert total == 10

    def test_limit_rows_larger_than_input(self) -> None:
        rows = _make_rows(20)
        chunks = list(
            _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=1000)
        )
        total = sum(len(c) for c in chunks)
        assert total == 20  # capped by available rows, not limit_rows

    def test_limit_rows_zero_raises_or_returns_empty(self) -> None:
        import pytest

        rows = _make_rows(10)
        try:
            chunks = list(
                _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=10, limit_rows=0)
            )
            # If it doesn't raise, it must return empty
            total = sum(len(c) for c in chunks)
            assert total == 0
        except ValueError:
            pass  # Raising is also acceptable


# ---------------------------------------------------------------------------
# J8.94.6 – Invalid chunk_size raises ValueError
# ---------------------------------------------------------------------------


class TestChunkingValidation:
    def test_zero_chunk_size_raises(self) -> None:
        import pytest

        rows = _make_rows(10)
        with pytest.raises(ValueError, match="chunk_size"):
            list(
                _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=0, limit_rows=None)
            )

    def test_negative_chunk_size_raises(self) -> None:
        import pytest

        rows = _make_rows(10)
        with pytest.raises(ValueError, match="chunk_size"):
            list(
                _iter_frame_chunks(rows, geometry="geometry", crs=None, chunk_size=-5, limit_rows=None)
            )
