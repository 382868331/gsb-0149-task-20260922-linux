"""Exception hierarchy for the moving-GC kernel.

All heap-specific errors derive from :class:`GCError` so callers can catch
the whole family with one ``except`` clause.
"""

from __future__ import annotations


class GCError(Exception):
    """Base class for every error raised by the garbage-collected heap."""


class StaleHandleError(GCError):
    """A well-formed handle that does not name a currently live object.

    Raised when the handle epoch differs from the heap epoch (i.e. it was
    minted before a collection) or when no object currently lives at the
    handle's address.
    """


class HeapCapacityError(GCError):
    """The active semi-space cannot satisfy an allocation.

    The heap has already performed one emergency collection on the
    caller's behalf; the surviving working set genuinely does not fit,
    or the single object is larger than the semi-space capacity.
    """
