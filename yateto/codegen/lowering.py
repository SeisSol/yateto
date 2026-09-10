"""Turning the statements of a region into what writes them.

A statement reaches the region stated over tensors, whichever backend has
claimed it: a destination, operands with their index maps, a factor, an
accumulation. What writes it is settled here -- loops and scalar operations
put in its place, or a call, for a backend that writes the statement into the
kernel itself and reports what it did when it does.

Settling it here rather than while the region is built is what lets a pass
read a statement as the statement it is. A contraction is a contraction until
this runs; afterwards it is a nest around somebody's matrix product, and there
is nothing left to recognise.
"""

from .. import ir


def lower(region, gemm_cfg):
  """Replace every statement still stated over tensors by what writes it."""
  ops = []
  for op in region.ops:
    for nested in op.regions():
      lower(nested, gemm_cfg)
    if isinstance(op, ir.TensorOp):
      ops.extend(_written(op, gemm_cfg))
    else:
      ops.append(op)
  region.ops = ops
  return region


def _arguments(op, gemm_cfg):
  """What the backend of this statement is handed beyond the statement itself.

  A contraction is run as matrix products and which kernel performs one is the
  gemm configuration's decision, so its backend is asked with the
  configuration in hand. No other statement has anything to choose from.
  """
  return (gemm_cfg,) if isinstance(op, ir.LoopOverGEMM) else ()


def _written(op, gemm_cfg):
  """What the backend that claimed this statement makes of it.

  A backend that lowers hands back a region and this stands in the statement's
  place. One that does not writes the statement itself, and then all that is
  left here is a call saying which buffers it reaches -- a call that has not
  been asked may touch anything, and then nothing can be said about what the
  kernel keeps in memory.
  """
  arguments = _arguments(op, gemm_cfg)
  lowering = getattr(op.generator, 'lower', None)
  if lowering is not None:
    return lowering(*arguments).ops

  operands = [term for term in [op.result] + op.terms
              if hasattr(term, 'memoryLayout')]
  return [ir.Call(lambda cpp, cache: op.generator.generate(cpp, cache, *arguments),
                  reads=[ir.Buffer.fromDescription(term) for term in operands],
                  writes=[ir.Buffer.fromDescription(op.result)])]
