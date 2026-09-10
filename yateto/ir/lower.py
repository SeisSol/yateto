from .. import ops as operations
from ..ast.indices import BoundingBox
from ..codegen.common import scaleFactor
from .build import indexMap, load, loopNest, scaled, zero
from .core import Buffer, Builder, Entries, Region
from .ops import Const, Loop, Read, Scope, Store


def lowerScaleAdd(op):
  """``result <op>= alpha * term`` as loops.

  One loop nest over the destination's index space. The operand is read at the
  coordinates its own index tuple names, so a permuted operand addresses the
  same loop variables in another order and one that lacks an index does not
  address it at all -- a transposition and a broadcast are the same lowering
  seen from two index maps.

  Where the operand's layout is sparse there is no address expression to run a
  loop over. The nest then states the entries it visits instead and is
  unrolled before it is emitted, which turns every address into a number and
  every entry the layout does not keep into a zero.
  """
  result, term = op.result, op.terms[0]
  resultBuffer = Buffer.fromDescription(result)
  termBuffer = Buffer.fromDescription(term)

  region = Region()
  builder = Builder(region)

  sparse = term.memoryLayout.isSparse()

  if not op.add:
    # A sparse operand is read entry by entry, so which entries of the
    # destination are written is not a box and the whole of it is zeroed.
    box = None if sparse else BoundingBox(
      [op.loopRanges[index] for index in result.indices])
    zero(builder, resultBuffer, box)

  factor = _factor(op.alpha, result.datatype)
  if isinstance(factor, Read):
    # the local lives in a scope of its own: one kernel may scale two
    # statements by the same name
    scope = builder.add(Scope())
    builder = Builder(scope.region)
    builder.add(factor)
  elif factor is not None:
    builder.add(factor)

  indices = indexMap(result.indices)
  if sparse:
    order = [indices[index] for index in result.indices]
    entries = sorted(zip(*result.eqspp.nonzero()), key=lambda entry: entry[::-1])
    loop = builder.add(Loop(order, Entries(entries)))
    body = Builder(loop.region)
  else:
    body = loopNest(builder, [indices[index] for index in result.indices],
                    op.loopRanges)

  accumulate, factor = _accumulation(op.add, factor)
  value = body.add(load(termBuffer, [indices[index] for index in term.indices]))
  scaledValue = scaled(value, factor, result.datatype)
  if scaledValue is not value:
    body.add(scaledValue)
  body.add(Store(resultBuffer, [indices[index] for index in result.indices],
                 scaledValue, accumulate))

  return region


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
