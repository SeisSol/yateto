from .build import load, scaled
from .core import Region
from .ops import Arith, Const, Load, Loop, Memset, Read, Scope, Store


def unroll(region):
  """Replace every loop that states its entries by copies of its body.

  This is what a sparse operand needs and the only thing it needs: pinning the
  indices turns the coordinates into numbers, an address in a sparse layout
  exists for numbers, and an entry the layout does not keep becomes a zero on
  the way. Everything before this pass sees one loop over a set of entries and
  can reason about it as a loop.
  """
  ops = []
  for op in region.ops:
    for nested in op.regions():
      unroll(nested)
    if isinstance(op, Loop) and op.isUnrollable():
      for entry in op.domain.entries:
        pinned = dict(zip(op.index, entry))
        ops.extend(_clone(op.region, pinned, {}))
    else:
      ops.append(op)
  region.ops = ops
  return region


def _clone(region, pinned, mapping):
  """A copy of the region's ops with `pinned` indices replaced by their values.

  `mapping` carries values from the original ops to their copies. A value the
  region reads but does not produce -- a factor hoisted out of the loop, say --
  is not in it and is referred to unchanged.
  """
  cloned = []
  for op in region.ops:
    copy = _cloneOp(op, pinned, mapping)
    if copy is not None:
      cloned.append(copy)
    mapping[id(op)] = copy
  return cloned


def _cloneOp(op, pinned, mapping):
  value = lambda operand: mapping.get(id(operand), operand)
  coords = lambda op: [coord.substituted(pinned) for coord in op.coords]

  if isinstance(op, Load):
    # folds to a zero where the layout keeps nothing at the pinned entry
    return load(op.buffer, coords(op))
  if isinstance(op, Const):
    return Const(op.value, op.datatype)
  if isinstance(op, Read):
    return Read(op.expression, op.datatype, op.name)
  if isinstance(op, Arith):
    args = [value(arg) for arg in op.args]
    if _isScaling(op, args):
      return scaled(args[1], args[0], op.datatype)
    return Arith(op.operation, args, op.datatype)
  if isinstance(op, Store):
    return Store(op.buffer, coords(op), value(op.value), op.accumulate)
  if isinstance(op, Memset):
    return Memset(op.buffer, op.offset, op.count)
  if isinstance(op, Scope):
    copy = Scope(Region())
    copy.region.extend(_clone(op.region, pinned, mapping))
    return copy
  if isinstance(op, Loop):
    copy = Loop(op.index, op.domain, Region(), op.simd, op.collapse)
    copy.region.extend(_clone(op.region, pinned, mapping))
    return copy

  raise NotImplementedError(f'{type(op).__name__} cannot be cloned yet.')


def _isScaling(op, args):
  """Whether the copy of this operation may fold as a scaling.

  A multiplication by a constant is the one arithmetic the pinning can change:
  the operand it scales may have become a zero.
  """
  from .. import ops as operations
  return (op.operation == operations.Mul() and len(args) == 2
          and isinstance(args[0], Const))
