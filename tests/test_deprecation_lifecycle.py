from __future__ import annotations

import pytest

from geoprompt.ecosystem import deprecated_alias


def test_deprecated_alias_attaches_lifecycle_metadata() -> None:
    @deprecated_alias("new_func", deprecated_in="0.1.0", remove_in="0.2.0", note="use replacement")
    def old_func(x: int) -> int:
        return x + 1

    with pytest.warns(DeprecationWarning, match="Deprecated in 0.1.0"):
        assert old_func(2) == 3

    metadata = getattr(old_func, "_geoprompt_deprecation")
    assert metadata["new_name"] == "new_func"
    assert metadata["deprecated_in"] == "0.1.0"
    assert metadata["remove_in"] == "0.2.0"


def test_deprecated_alias_rejects_non_semver() -> None:
    with pytest.raises(ValueError, match="semantic format"):
        @deprecated_alias("new_func", deprecated_in="v1", remove_in="0.2.0")
        def old_func(x: int) -> int:
            return x


def test_deprecated_alias_rejects_insufficient_grace_window() -> None:
    with pytest.raises(ValueError, match="grace"):
        @deprecated_alias("new_func", deprecated_in="0.1.0", remove_in="0.1.1")
        def old_func(x: int) -> int:
            return x
