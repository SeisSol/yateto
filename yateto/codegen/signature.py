"""What a kernel needs from whoever calls it, read off its statements.

The interface is what the statements name: the operands they reach, the values
their guards are decided on, the factors they scale by, and what a contraction
asks to be prefetched. All of that stands in the region, so all of it is asked
of the region -- and a statement that stands nowhere, because it can never
run, asks for nothing.

An operand answers here in two shapes. Most are tensor descriptions, as the
statements state them; the values a guard is decided on are the values
themselves. Both say which tensor stands behind the name, whether the kernel
writes it and what its datatype is, which is all the interface is built from.
"""

from .. import ir
from ..type import DerivedScalar, Tensor


def operands(region):
  """Every operand the region names, one per name.

  A name is one tensor -- the kernel declares it once -- so which of two
  mentions answers for it makes no difference: they say the same about what
  stands behind the name.
  """
  found = {}
  for op in region.walk():
    if isinstance(op, ir.TensorOp):
      reached, written = op.touched()
      for term in reached + written:
        found.setdefault(term.name, term)
    elif isinstance(op, ir.If):
      for value in op.condition.variables():
        found.setdefault(str(value), value)
  return found


def globals(region):
  """The operands the caller hands over, in a stable order."""
  return [operand for _, operand in sorted(operands(region).items())
          if operand.isGlobal()]


def prefetch(region):
  """The tensors a contraction asks to have fetched while it runs."""
  asked = {op.prefetch for op in region.walk()
           if isinstance(op, ir.LoopOverGEMM) and op.prefetch is not None}
  return sorted(asked, key=lambda tensor: tensor.name())


def _factors(region):
  return {op.alpha for op in region.walk()
          if isinstance(op, ir.TensorOp) and isinstance(op.alpha, Tensor)}


def scalars(region):
  """The scalars the caller sets, i.e. the ones in the kernel signature."""
  found = _factors(region)
  # a derived scalar is computed in the prologue, so it also pulls in the
  # named scalars its expression reads
  for derived in [scalar for scalar in found if isinstance(scalar, DerivedScalar)]:
    found = found | derived.dependencies()
  return {scalar for scalar in found if not scalar.temporary}


def derivedScalars(region):
  """The scalars the kernel computes before it does anything else."""
  derived = [scalar for scalar in _factors(region)
             if isinstance(scalar, DerivedScalar)]
  return sorted(derived, key=lambda scalar: scalar.name())
