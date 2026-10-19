"""Library-wide settings."""

from __future__ import annotations

import os

_threads: int | None = None


def default_threads() -> int:
    """``min(4, cores)``, or the ``UME_THREADS`` environment variable when set."""
    env = os.environ.get("UME_THREADS")
    if env:
        return max(1, int(env))
    return max(1, min(4, os.cpu_count() or 1))


def get_threads() -> int:
    """The number of worker threads ume's Rust passes and draw fits use.

    Returns
    -------
    int
        The value set with :func:`set_threads`, else ``UME_THREADS``, else
        ``min(4, cores)``.
    """
    return _threads if _threads is not None else default_threads()


def set_threads(n: int | None) -> None:
    """Set the number of worker threads, or ``None`` to restore the default.

    Passes route each cluster to one worker by a hash of its key and merge the
    workers' results at the end, so the results do not depend on the number of
    threads beyond the order of floating-point additions (about 1e-15 relative).

    Parameters
    ----------
    n : int or None
        At least 1.

    Examples
    --------
    >>> import ume
    >>> ume.set_threads(2)
    >>> ume.get_threads()
    2
    >>> ume.set_threads(None)
    """
    global _threads
    if n is not None and n < 1:
        raise ValueError("threads must be at least 1")
    _threads = n
