"""J8.92 – Flaky-test quarantine mechanism with expiry and owner fields.

Provides a ``flaky_quarantine`` pytest fixture and a ``@quarantine`` decorator
that:
  1. Mark a test as quarantined with an owner, reason, and expiry date.
  2. Run the test but do NOT fail the build if it fails within the quarantine
     window — instead, emit an ``xfail`` (expected failure).
  3. FAIL the build if the quarantine has expired but the test is still marked,
     prompting owners to either fix the test or extend the quarantine.
  4. Record the quarantine registry in a central file so ownership and expiry
     are auditable.

Usage::

    from tests.conftest_quarantine import quarantine

    @quarantine(owner="platform-team", reason="Flaky on Windows CI", expires="2026-06-30")
    def test_something_flaky():
        ...

The quarantine mechanism integrates with pytest via the ``@pytest.mark.xfail``
mechanism so quarantined tests show as xfail rather than fail.
"""
from __future__ import annotations

import datetime
import functools
import warnings
from typing import Any, Callable

import pytest


# ---------------------------------------------------------------------------
# Quarantine registry — edit this to track all quarantined tests
# ---------------------------------------------------------------------------

_QUARANTINE_REGISTRY: list[dict[str, str]] = [
    # Example entry (not a real test):
    # {
    #     "test": "tests/test_example.py::test_something_flaky",
    #     "owner": "platform-team",
    #     "reason": "Intermittent timing failure on slow CI agents",
    #     "expires": "2026-06-30",
    # },
]


# ---------------------------------------------------------------------------
# Core mechanism
# ---------------------------------------------------------------------------

_DATE_FORMAT = "%Y-%m-%d"


def _parse_date(s: str) -> datetime.date:
    return datetime.datetime.strptime(s, _DATE_FORMAT).date()


def _is_expired(expires: str) -> bool:
    return datetime.date.today() > _parse_date(expires)


def quarantine(
    *,
    owner: str,
    reason: str,
    expires: str,
    strict: bool = False,
) -> Callable[[Callable], Callable]:
    """Decorator that marks a test as quarantined until *expires*.

    Args:
        owner: Team or person responsible for fixing this test.
        reason: Why the test is quarantined.
        expires: ISO date string (YYYY-MM-DD). After this date, the test will
            be treated as a hard failure even if it passes, to force cleanup.
        strict: If True, the test will still cause a build failure on flaky
            passes (useful for race conditions that must be fixed urgently).

    Raises:
        pytest.fail: At collection time if the quarantine has expired, so
            the owner is forced to either fix the test or update the entry.
    """
    try:
        expiry_date = _parse_date(expires)
    except ValueError as exc:
        raise ValueError(
            f"@quarantine expires={expires!r} is not a valid ISO date (YYYY-MM-DD)"
        ) from exc

    def decorator(fn: Callable) -> Callable:
        if _is_expired(expires):
            # Expired quarantine — wrap with a hard failure at call time
            @functools.wraps(fn)
            def _expired_wrapper(*args: Any, **kwargs: Any) -> None:
                pytest.fail(
                    f"Quarantine expired on {expires} for test {fn.__qualname__!r}. "
                    f"Owner: {owner!r}. Reason: {reason!r}. "
                    "Fix the test or extend the quarantine expiry date."
                )
            return _expired_wrapper
        else:
            # Active quarantine — mark as xfail
            xfail_marker = pytest.mark.xfail(
                reason=f"[QUARANTINED until {expires}] {reason} (owner: {owner})",
                strict=strict,
            )
            return xfail_marker(fn)

    return decorator


# ---------------------------------------------------------------------------
# Tests for the quarantine mechanism itself
# ---------------------------------------------------------------------------


class TestQuarantineDecorator:
    def test_quarantine_future_expiry_marks_xfail(self) -> None:
        """A quarantine with a future expiry produces an xfail marker."""
        future = (datetime.date.today() + datetime.timedelta(days=90)).strftime(_DATE_FORMAT)

        @quarantine(owner="test-team", reason="Example flaky test", expires=future)
        def _fake_test():
            pass

        markers = list(getattr(_fake_test, "pytestmark", []))
        marker_names = [m.name for m in markers]
        assert "xfail" in marker_names, (
            f"Expected 'xfail' marker, got: {marker_names}"
        )

    def test_quarantine_expired_wraps_with_fail(self) -> None:
        """An expired quarantine wraps the test to fail at call time."""
        past = "2020-01-01"

        @quarantine(owner="test-team", reason="Old test", expires=past)
        def _fake_expired_test():
            pass

        with pytest.raises(pytest.fail.Exception, match="Quarantine expired"):
            _fake_expired_test()

    def test_quarantine_invalid_date_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not a valid ISO date"):
            quarantine(owner="x", reason="y", expires="not-a-date")

    def test_quarantine_registry_has_correct_schema(self) -> None:
        """Every entry in the registry must have required fields."""
        required = {"test", "owner", "reason", "expires"}
        for entry in _QUARANTINE_REGISTRY:
            missing = required - set(entry.keys())
            assert not missing, (
                f"Quarantine registry entry missing fields {missing}: {entry}"
            )

    def test_quarantine_registry_entries_have_valid_dates(self) -> None:
        """All expiry dates in the registry must be valid ISO dates."""
        for entry in _QUARANTINE_REGISTRY:
            try:
                _parse_date(entry["expires"])
            except (ValueError, KeyError) as exc:
                pytest.fail(
                    f"Invalid expiry date in quarantine registry for {entry.get('test')!r}: {exc}"
                )

    def test_quarantine_registry_has_no_expired_entries(self) -> None:
        """No entry in the registry should be expired — expired entries must be fixed."""
        today = datetime.date.today()
        expired = [
            e for e in _QUARANTINE_REGISTRY
            if _is_expired(e.get("expires", "2000-01-01"))
        ]
        assert not expired, (
            f"Expired quarantine entries found — fix these tests or remove entries:\n"
            + "\n".join(
                f"  {e['test']} (expired {e['expires']}, owner: {e['owner']})"
                for e in expired
            )
        )


class TestIsExpiredHelper:
    def test_past_date_is_expired(self) -> None:
        assert _is_expired("2020-01-01") is True

    def test_future_date_is_not_expired(self) -> None:
        future = (datetime.date.today() + datetime.timedelta(days=1)).strftime(_DATE_FORMAT)
        assert _is_expired(future) is False

    def test_today_is_not_expired(self) -> None:
        today = datetime.date.today().strftime(_DATE_FORMAT)
        assert _is_expired(today) is False
