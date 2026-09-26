"""moving_gc: a self-contained semi-space moving GC kernel with weak edges.

Public surface::

    from moving_gc import Heap, Handle, CollectionResult
    from moving_gc import GCError, StaleHandleError, HeapCapacityError
    from moving_gc import MAX_CAPACITY

See README.md for the full contract, design notes and limitations.
"""

from __future__ import annotations

from .core import (
    MAX_CAPACITY,
    CollectionResult,
    Handle,
    Heap,
    PinHandle,
)
from .errors import (
    FragmentationError,
    GCError,
    HeapCapacityError,
    StaleHandleError,
    StalePinHandleError,
)

__all__ = [
    "Heap",
    "Handle",
    "CollectionResult",
    "GCError",
    "StaleHandleError",
    "HeapCapacityError",
    "PinHandle",
    "StalePinHandleError",
    "FragmentationError",
    "MAX_CAPACITY",
]
