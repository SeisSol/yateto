from .. import ops as operations
from ..ast.indices import BoundingBox
from ..codegen.common import scaleFactor
from ..type import AddressingMode
from .build import indexMap, load, loopNest, scaled, zero
from .core import Buffer, Builder, Entries, Region, ValueOp
from .ops import Arith, Const, Loop, Read, Scope, Store


def lowerScaleAdd(op):
  """``result <op>= alpha * term`` as loops.

  One loop nest over the destination's index space. The operand is read at the
  coordinates its own index tuple names, so a permuted operand addresses the
  same loop variables in another order and one that lacks an index does not
  address it at all -- a transposition and a broadcast are the same lowering
  seen from two index maps.
  """
  region, builder, factor = _prologue(op)
  indices = indexMap(op.result.indices)
  body = _iteration(builder, op, indices)

  term = op.terms[0]
  value = _operand(body, term, indices, op.result.datatype)
  _store(body, op, indices, value, factor)
  return region


def lowerElementwise(op):
  """``result <op>= alpha * f(terms...)`` as loops.

  Every operand is read at the coordinates of its own index tuple, so one that
  lacks an index the destination has is read again for every value of it. An
  operand that is passed by value has no coordinates and is named directly.
  """
  region, builder, factor = _prologue(op)
  indices = indexMap(op.result.indices)
  body = _iteration(builder, op, indices)

  args = [_operand(body, term, indices, op.result.datatype) for term in op.terms]
  value = body.add(Arith(op.optype, args, op.result.datatype))
  _store(body, op, indices, value, factor)
  return region


def lowerFusedElementwise(op):
  """Several element-wise steps in one loop nest.

  A step reads what an earlier one computed, and since that value never leaves
  the nest it is a value of the loop body rather than a buffer. The last step
  writes the destination.
  """
  region, builder, factor = _prologue(op)
  indices = indexMap(op.result.indices)
  body = _iteration(builder, op, indices)

  produced = {}
  for position, member in enumerate(op.members):
    last = position + 1 == len(op.members)
    assert last or not member.step.add, \
      'only the last step of a nest accumulates, and it does so into the result'
    args = [produced[source] if source is not None
            else _operand(body, term, indices, member.datatype)
            for term, source in zip(member.terms, member.step.sources)]
    produced[position] = _step(body, member, args)
    if not last and produced[position] not in args:
      produced[position].name = f'_fused{position}'
      produced[position].materialize = True

  _store(body, op, indices, produced[len(op.members) - 1], factor)
  return region


def _step(builder, member, args):
  """What one step of a nest computes."""
  step = member.step
  if step.optype is not None:
    filled = [argument if isinstance(argument, ValueOp)
              else builder.add(Const(argument, member.datatype))
              for argument in step.fillTerms(args)]
    return builder.add(Arith(step.optype, filled, member.datatype))

  factor = _factor(step.scalar, member.datatype)
  if factor is not None:
    builder.add(factor)
  value = scaled(args[0], factor, member.datatype)
  if value is not args[0]:
    builder.add(value)
  return value


def _prologue(op):
  """The region, a builder for it, and the factor to scale by.

  Everything the destination is not going to be written over is zeroed first,
  and a factor that has a name is read into a local of its own scope -- one
  kernel may well scale two statements by the same name.
  """
  region = Region()
  builder = Builder(region)
  resultBuffer = Buffer.fromDescription(op.result)

  if not op.add:
    # Where an operand is read entry by entry, which entries of the
    # destination are written is not a box, and the whole of it is zeroed.
    box = None if op.unrolled else BoundingBox(
      [op.loopRanges[index] for index in op.result.indices])
    zero(builder, resultBuffer, box)

  factor = _factor(op.alpha, op.result.datatype)
  if isinstance(factor, Read):
    scope = builder.add(Scope())
    builder = Builder(scope.region)
    builder.add(factor)
  elif factor is not None:
    builder.add(factor)
  return region, builder, factor


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


def _store(builder, op, indices, value, factor):
  """Write the value to the destination, scaled and combined as asked."""
  accumulate, factor = _accumulation(op.add, factor)
  scaledValue = scaled(value, factor, op.result.datatype)
  if scaledValue is not value:
    builder.add(scaledValue)
  builder.add(Store(Buffer.fromDescription(op.result),
                    [indices[index] for index in op.result.indices],
                    scaledValue, accumulate))


def _factor(alpha, datatype):
  """The scale factor as a value, or None where it does not scale.

  Written in the destination's datatype, which is what keeps an int32 result
  from being multiplied by a double literal, and what rejects a factor the
  type cannot hold.
  """
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
