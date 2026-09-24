"""moving_gc: moving-GC kernels with weak references, ephemerons, finalizers.

Public surface::

    # v1 (legacy, unchanged): single-generation semi-space copying heap
    from moving_gc import Heap, Handle, CollectionResult
    from moving_gc import GCError, StaleHandleError, HeapCapacityError
    from moving_gc import MAX_CAPACITY

    # v2 (new, explicit opt-in): generational incremental heap
    from moving_gc import GenerationalHeap, GenCollectionResult
    from moving_gc import StepReport, CycleInfo, FinalizationEvent
    from moving_gc import Relocation, WeakClear, EphemeronClear
    from moving_gc import OLD, YOUNG, DEFAULT_SCAN_BUDGET

The v1 engine is untouched: ``Heap(capacity)`` behaves exactly as before.
The v2 engine is a separate, explicitly-named class; see README.md for the
full contract, the v1->v2 change list and migration notes.
"""

from __future__ import annotations

from .core import (
    MAX_CAPACITY,
    CollectionResult,
    Handle,
    Heap,
)
from .errors import (
    GCError,
    HeapCapacityError,
    StaleHandleError,
    UnsupportedOperationError,
)
from .gen import (
    DEFAULT_SCAN_BUDGET,
    OLD,
    YOUNG,
    CycleInfo,
    EphemeronClear,
    FinalizationEvent,
    GenCollectionResult,
    GenerationalHeap,
    Relocation,
    StepReport,
    WeakClear,
)

__all__ = [
    # v1
    "Heap",
    "Handle",
    "CollectionResult",
    "MAX_CAPACITY",
    # shared errors
    "GCError",
    "StaleHandleError",
    "HeapCapacityError",
    "UnsupportedOperationError",
    # v2
    "GenerationalHeap",
    "GenCollectionResult",
    "StepReport",
    "CycleInfo",
    "FinalizationEvent",
    "Relocation",
    "WeakClear",
    "EphemeronClear",
    "OLD",
    "YOUNG",
    "DEFAULT_SCAN_BUDGET",
]
