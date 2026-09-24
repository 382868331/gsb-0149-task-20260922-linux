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
    handle's address/generation.
    """


class HeapCapacityError(GCError):
    """The target generation cannot satisfy an allocation.

    The heap has already performed the one mandated emergency collection
    on the caller's behalf (a minor collection for young allocations, a
    major collection for old allocations); the surviving working set
    genuinely does not fit, or the single object is larger than the
    generation capacity.
    """


class UnsupportedOperationError(GCError):
    """The operation is only available in a different heap format/version.

    Raised, for example, when generational/ephemeron/finalizer entry points
    are used against a legacy v1 heap (``Heap(capacity)``), or when an
    incremental step is requested from the non-incremental legacy engine.
    """
