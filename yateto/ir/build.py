from .. import ops as operations
from ..ast.log import splitByDistance
from .address import storesValue
from .affine import Index
from .core import Builder
from .ops import Arith, Const, Load, Loop, Memset, Scope


def scaleFactor(datatype, alpha):
  """Spell a scale factor in the result's datatype.

  A number becomes a literal of that type; a named scalar is a kernel argument
  and is emitted by name, since it already carries its own type.

  Writing the factor in the result's type is what keeps an int32 result from
  being multiplied by a double literal, but it only works while the type can
  hold the factor. It cannot always: every non-zero number is `true`, so a
  boolean result silently scales by one, and an integer one truncates. Neither
  is a scaling, so neither is written.
  """
  if not isinstance(alpha, (int, float)):
    return str(alpha)
  if datatype.isBool() and alpha != 1:
    raise ValueError(
      f'Cannot scale a {datatype} result by {alpha}: every non-zero factor is '
      f'the same boolean, so the factor would be lost. Cast the result to a '
      f'numeric type before scaling it.')
  if datatype.isInteger() and alpha != int(alpha):
    raise ValueError(
      f'Cannot scale a {datatype} result by {alpha}: the factor is not whole '
      f'and writing it in the result\'s type would truncate it. Cast the '
      f'result to a floating type before scaling it.')
  return datatype.literal(alpha)


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

  The first index runs fastest, so it becomes the innermost loop: it is the
  one the memory layouts give a stride of one. A nest with nothing between its
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
