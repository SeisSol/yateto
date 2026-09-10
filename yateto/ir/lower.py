from .. import ops as operations
from ..ast.indices import BoundingBox
from ..type import AddressingMode
from .build import indexMap, load, loopNest, scaleFactor, scaled, zero
from .core import Buffer, Builder, Entries, Region
from .ops import Arith, Const, Fold, Loop, Read, Store, Yield


def lowerScaleAdd(op):
  """``result <op>= alpha * term`` as loops.

  One loop nest over the destination's index space. The operand is read at the
  coordinates its own index tuple names, so a permuted operand addresses the
  same loop variables in another order and one that lacks an index does not
  address it at all -- a transposition and a broadcast are the same lowering
  seen from two index maps.
  """
  region, builder, factors = _prologue(op)
  indices = indexMap(op.result.indices)
  body = _iteration(builder, op, indices)

  term = op.terms[0]
  value = _operand(body, term, indices, op.result.datatype)
  _store(body, op, indices, value, factors)
  return region


def lowerElementwise(op):
  """``result <op>= alpha * f(terms...)`` as loops.

  Every operand is read at the coordinates of its own index tuple, so one that
  lacks an index the destination has is read again for every value of it. An
  operand that is passed by value has no coordinates and is named directly.
  """
  region, builder, factors = _prologue(op)
  indices = indexMap(op.result.indices)
  body = _iteration(builder, op, indices)

  args = [_operand(body, term, indices, op.result.datatype) for term in op.terms]
  value = body.add(Arith(op.optype, args, op.result.datatype))
  _store(body, op, indices, value, factors)
  return region


def lowerReduction(op):
  """``result <op>= alpha * fold(term over one index)`` as loops.

  One loop nest over the destination's index space, and inside it a fold over
  the index the destination does not have. The fold and the store stand in the
  body of that nest like any other pair of operations: whoever reads the
  destination back in the same iteration reads the value, not the memory.
  """
  region, builder, factors = _prologue(op)
  indices = indexMap(op.terms[0].indices)
  body = _iteration(builder, op, indices)

  contribution = Builder()
  value = _operand(contribution, op.terms[0], indices, op.result.datatype)
  contribution.add(Yield(value))
  folded = body.add(Fold(indices[op.sumIndex], op.sumRange, op.optype,
                         contribution.region(), op.result.datatype,
                         name='_acc'))

  # The factor scales the fold, and accumulating into the destination combines
  # with the operation that was folded -- not with an addition, which for
  # anything but a sum would be a different statement.
  value = _scale(body, folded, factors.get(str(op.alpha)), op.result.datatype)
  body.add(Store(Buffer.fromDescription(op.result),
                 [indices[index] for index in op.result.indices],
                 value, op.optype if op.add else None))
  return region


def _scale(builder, value, factor, datatype):
  """`factor * value`, where there is a factor to apply."""
  scaledValue = scaled(value, factor, datatype)
  if scaledValue is not value:
    builder.add(scaledValue)
  return scaledValue


def _prologue(op):
  """The region, a builder for it, and the factors the statement scales by.

  Everything the destination is not going to be written over is zeroed first.
  The factor is made once, outside the nest, and is found again by the
  spelling of what it scales by: one kernel may well scale two statements by
  the same name.
  """
  region = Region()
  builder = Builder(region)

  if not op.add:
    # Where an operand is read entry by entry, which entries of the
    # destination are written is not a box, and the whole of it is zeroed.
    box = None if op.unrolled else BoundingBox(
      [op.loopRanges[index] for index in op.result.indices])
    zero(builder, Buffer.fromDescription(op.result), box)

  factors = {}
  for alpha in (op.alpha,):
    if str(alpha) in factors:
      continue
    factor = _factor(alpha, op.result.datatype)
    if factor is not None:
      factors[str(alpha)] = factor

  for factor in factors.values():
    builder.add(factor)
  return region, builder, factors


def _iteration(builder, op, indices):
  """A builder for the body of the nest the statement runs over."""
  order = [indices[index] for index in op.result.indices]
  if op.unrolled:
    entries = sorted(zip(*op.result.eqspp.nonzero()), key=lambda entry: entry[::-1])
    loop = builder.add(Loop(order, Entries(entries)))
    return Builder(loop.region)
  return loopNest(builder, order, op.loopRanges)


def _operand(builder, term, indices, datatype):
  """How one operand of a statement is read."""
  if not hasattr(term, 'memoryLayout'):
    # an immediate the operation was written with
    return builder.add(Const(term, datatype))
  if term.addressing == AddressingMode.SCALAR:
    # passed by value: it has a name and no storage to address
    return builder.add(Read(term.name, term.datatype, hoist=False))
  buffer = Buffer.fromDescription(term)
  return builder.add(load(buffer, [indices[index] for index in term.indices]))


def _store(builder, op, indices, value, factors):
  """Write the value to the destination, scaled and combined as asked."""
  accumulate, factor = _accumulation(op.add, factors.get(str(op.alpha)))
  scaledValue = scaled(value, factor, op.result.datatype)
  if scaledValue is not value:
    builder.add(scaledValue)
  builder.add(Store(Buffer.fromDescription(op.result),
                    [indices[index] for index in op.result.indices],
                    scaledValue, accumulate))


def _factor(alpha, datatype):
  """The scale factor as a value, or None where it does not scale.

  A statement that states no factor and one that states a factor of one scale
  the same way, which is not at all.

  A factor that does scale is written in the destination's datatype, which is
  what keeps an int32 result from being multiplied by a double literal, and
  what rejects a factor the type cannot hold.
  """
  if alpha is None:
    return None
  if not isinstance(alpha, (int, float)):
    return Read(str(alpha), datatype, name='_alpha')
  if alpha == 1.0:
    return None
  scaleFactor(datatype, alpha)
  return Const(alpha, datatype)


def _accumulation(add, factor):
  """How the store combines with the destination, and what is left to scale by.

  Accumulating a value scaled by minus one is a subtraction, and stating it as
  one leaves nothing to multiply by.
  """
  if not add:
    return None, factor
  if isinstance(factor, Const) and factor.value == -1.0:
    return operations.Sub(), None
  return operations.Add(), factor
