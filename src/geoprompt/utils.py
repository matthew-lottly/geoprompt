"""Framework utilities: parallel processing, caching, configuration, logging, retry, temp files.

Pure-Python helpers for running spatial workloads in parallel, memoizing
expensive results, managing configuration, structured logging, and safe
file operations.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Parallel helpers  (items 1117-1119)
# ---------------------------------------------------------------------------

def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
    use_processes: bool = False,
    timeout: float | None = None,
) -> list[R]:
    """Apply *func* to each item in parallel (threads or processes).

    Returns results in the same order as *items*.
    """
    items_list = list(items)
    if not items_list:
        return []

    pool_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    results: list[R | None] = [None] * len(items_list)

    with pool_cls(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, item): idx for idx, item in enumerate(items_list)
        }
        for future in as_completed(future_to_idx, timeout=timeout):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results  # type: ignore[return-value]


def parallel_apply(
    func: Callable[..., R],
    args_list: Sequence[tuple[Any, ...]],
    *,
    max_workers: int | None = None,
    use_processes: bool = False,
) -> list[R]:
    """Apply *func* with varying arguments in parallel."""
    if not args_list:
        return []

    pool_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    results: list[R | None] = [None] * len(args_list)

    with pool_cls(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, *args): idx for idx, args in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results  # type: ignore[return-value]


def chunked(items: Sequence[T], size: int) -> list[list[T]]:
    """Split *items* into chunks of *size*."""
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Cache / memoization  (items 1843-1844)
# ---------------------------------------------------------------------------

class DiskCache:
    """Simple disk-backed JSON cache keyed by content hash."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "geoprompt_cache"
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, key_data: str) -> Any | None:
        path = self._dir / f"{self._key(key_data)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, key_data: str, value: Any) -> None:
        path = self._dir / f"{self._key(key_data)}.json"
        path.write_text(json.dumps(value), encoding="utf-8")

    def clear(self) -> int:
        count = 0
        for f in self._dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


def memoize(func: Callable[..., R]) -> Callable[..., R]:
    """LRU-cache decorator with a 256-entry limit."""
    return functools.lru_cache(maxsize=256)(func)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Configuration management  (item 1161)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "coordinate_precision": 6,
    "default_crs": "EPSG:4326",
    "max_workers": None,
    "cache_enabled": True,
    "log_level": "INFO",
    "temp_dir": None,
}

_config: dict[str, Any] = dict(_DEFAULT_CONFIG)


def get_config(key: str | None = None) -> Any:
    """Get a configuration value, or the entire config dict if key is None."""
    if key is None:
        return dict(_config)
    return _config.get(key, _DEFAULT_CONFIG.get(key))


def set_config(key: str, value: Any) -> None:
    """Set a configuration value."""
    _config[key] = value


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config.clear()
    _config.update(_DEFAULT_CONFIG)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a JSON file, merging with defaults."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _config.update(data)
    return dict(_config)


def save_config(path: str | Path) -> None:
    """Save current configuration to a JSON file."""
    Path(path).write_text(json.dumps(_config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Structured logging  (item 1232)
# ---------------------------------------------------------------------------

_logger: logging.Logger | None = None


def get_logger(name: str = "geoprompt") -> logging.Logger:
    """Get or create a structured logger for the geoprompt package."""
    global _logger
    if _logger is not None and _logger.name == name:
        return _logger

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, _config.get("log_level", "INFO")))
    _logger = logger
    return logger


def log_operation(operation: str, **kwargs: Any) -> None:
    """Log a spatial operation with structured context."""
    logger = get_logger()
    details = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
    logger.info("[%s] %s", operation, details)


# ---------------------------------------------------------------------------
# Retry / timeout  (items 1226-1227)
# ---------------------------------------------------------------------------

def retry(
    func: Callable[..., R],
    *args: Any,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> R:
    """Retry a function with exponential backoff.

    Raises the last exception if all attempts fail.
    """
    last_exc: BaseException | None = None
    current_delay = delay
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(current_delay)
                current_delay *= backoff
    raise last_exc  # type: ignore[misc]


class Timeout:
    """Context manager that records elapsed time and checks a limit."""

    def __init__(self, seconds: float) -> None:
        self.limit = seconds
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timeout":
        self.start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.monotonic() - self.start

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - (time.monotonic() - self.start))

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.start) >= self.limit


# ---------------------------------------------------------------------------
# Temp file management  (items 1883, 1884)
# ---------------------------------------------------------------------------

@contextmanager
def temp_directory(prefix: str = "geoprompt_") -> Generator[Path, None, None]:
    """Context manager providing a temporary directory that is cleaned up on exit."""
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextmanager
def atomic_write(path: str | Path, mode: str = "w", encoding: str = "utf-8") -> Generator[Any, None, None]:
    """Write to a temporary file, then atomically move into place."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, mode, encoding=encoding if "b" not in mode else None) as f:
            yield f
        # Atomic rename (same filesystem)
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Path normalization  (item 1882)
# ---------------------------------------------------------------------------

def normalize_path(path: str | Path) -> Path:
    """Normalize and resolve a file path."""
    return Path(path).expanduser().resolve()


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary. Returns the Path."""
    p = normalize_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Hash / fingerprint helpers
# ---------------------------------------------------------------------------

def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute a hex digest hash of a file's contents."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_hash(feature: dict[str, Any]) -> str:
    """Compute a content hash for a GeoJSON-like feature dict."""
    return hashlib.sha256(json.dumps(feature, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    # Parallel
    "parallel_map",
    "parallel_apply",
    "chunked",
    # Cache
    "DiskCache",
    "memoize",
    # Config
    "get_config",
    "set_config",
    "reset_config",
    "load_config",
    "save_config",
    # Logging
    "get_logger",
    "log_operation",
    # Retry/timeout
    "retry",
    "Timeout",
    # Temp/file
    "temp_directory",
    "atomic_write",
    "normalize_path",
    "ensure_directory",
    "file_hash",
    "feature_hash",
]
