"""moving_gc: moving-garbage-collector kernels.

Two heaps share one data model:

* :class:`Heap` -- the original flat semi-space kernel (unchanged contract);
* :class:`GenHeap` -- the two-generation incremental kernel (young copying,
  old mark-sweep, IU barrier, remembered set, ephemerons, finalizers).

Public surface::

    from moving_gc import (
        Heap, Handle, CollectionResult,
        GenHeap, StepResult, YOUNG, OLD,
        GCError, StaleHandleError, HeapCapacityError,
        CollectionInProgressError, FinalizerError, SealedForReclamationError,
        MAX_CAPACITY,
    )

See README.md for the full contract, design notes and limitations.
"""

from __future__ import annotations

from .core import (
    MAX_CAPACITY,
    CollectionResult,
    Handle,
    Heap,
)
from .errors import (
    CollectionInProgressError,
    FinalizerError,
    GCError,
    HeapCapacityError,
    SealedForReclamationError,
    StaleHandleError,
)
from .generational import OLD, YOUNG, GenHeap, StepResult

__all__ = [
    "Heap",
    "Handle",
    "CollectionResult",
    "GenHeap",
    "StepResult",
    "YOUNG",
    "OLD",
    "GCError",
    "StaleHandleError",
    "HeapCapacityError",
    "CollectionInProgressError",
    "FinalizerError",
    "SealedForReclamationError",
    "MAX_CAPACITY",
]
