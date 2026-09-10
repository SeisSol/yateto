from .. import ops as operations
from ..ast.log import splitByDistance
from .address import storesValue
from .affine import Index
from .core import Builder
from .ops import Arith, Const, Load, Loop, Memset, Scope


def load(buffer, coords):
  """Read an entry, or the zero the layout keeps in its place.

  A layout that stores nothing at a known entry has no address to read from,
  and the value there is a zero whatever the operation does with it.
  """
  if not storesValue(buffer.memoryLayout, coords, buffer.eqspp):
    return Const(0, buffer.datatype)
  return Load(buffer, coords)


def scaled(value, factor, datatype):
  """`factor * value`, or `value` where the factor does not change it."""
  if factor is None:
    return value
  if isinstance(factor, Const) and factor.value == 1:
    return value
  if isinstance(value, Const) and value.value == 0:
    return Const(0, datatype)
  if isinstance(factor, Const) and factor.value == 0:
    return Const(0, datatype)
  return Arith(operations.Mul(), [factor, value], datatype)


def zero(builder, buffer, writeBox=None):
  """Zero the entries no operation is going to write.

  Given a box, only the addresses outside it are zeroed, in runs: what a box
  leaves out need not be contiguous, but it comes in few enough pieces that
  one call per piece beats one per entry. Without a box, and for a
  destination without axes -- which has no box to speak of -- the whole
  storage is zeroed.
  """
  if not writeBox:
    builder.add(Memset(buffer, 0, buffer.memoryLayout.requiredReals()))
    return

  addresses = sorted(buffer.memoryLayout.notWrittenAddresses(writeBox))
  for run in splitByDistance(addresses) if addresses else []:
    first, last = min(run), max(run)
    builder.add(Memset(buffer, first, last - first + 1))


def loopNest(builder, indices, ranges, simd=True):
  """Nest one loop per index and hand back a builder for the innermost body.

  The last index runs fastest, so it becomes the innermost loop, which is the
  order the memory layouts are built for. A nest with nothing between its
  loops is one iteration space and says so with a `collapse` clause; saying it
  on the innermost loop alone would leave a short loop that is unrolled away
  before the vectoriser sees it.
  """
  if not indices:
    scope = builder.add(Scope())
    return Builder(scope.region)

  nest = list(reversed(indices))
  collapse = len(nest) if simd and len(nest) > 1 else None
  current = builder
  for depth, index in enumerate(nest):
    loop = Loop([index], ranges[index.name],
                simd=simd and len(nest) == 1,
                collapse=collapse if depth == 0 else None)
    current.add(loop)
    current = Builder(loop.region)
  return current


def indexMap(indices):
  """One `Index` per index name, so operands share the loop variables."""
  return {name: Index(name) for name in indices}
