"""An intermediate representation between a kernel's statements and its C++.

Two levels live in the same region and may stand next to each other. The
tensor level (`yateto.ir.tensor`) states what the control-flow graph states:
a destination, operands with their index maps, a factor, an accumulation. The
loop level states loops, addresses and scalar operations. `lower()` turns one
into the other, and a statement an external generator takes away is simply one
that is never lowered.

Addresses are affine expressions over the loop indices rather than text, which
is what lets an index be pinned to a value and the address fall out as a
number. That is the whole of the unrolled path: `passes.unroll` pins the
indices of a loop that states its entries, and a sparse operand -- which has
an address for a known entry and none for an index -- becomes addressable.
"""

from .address import address, constantEntry, entry, storesValue
from .affine import Affine, Index
from .analysis import countFlops
from .build import indexMap, load, loopNest, scaleFactor, scaled, zero
from .core import Buffer, Builder, Entries, Op, Region, ValueOp
from .emit import INDEX_PREFIX, CppEmitter
from .fusion import fuseLoops
from .scalarize import buffers, scalarize
from .storage import assign
from .ops import (Arith, Call, Const, Fold, If, Load, Loop, Memset, Pointer,
                  Read, Scope, Store, Yield)
from .passes import unroll
from .tensor import (Broadcast, Copy, Elementwise, FusedGEMMs, LoopOverGEMM,
                     mayFuseGroups, Reduction, TensorOp, Transpose)
