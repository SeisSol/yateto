from .core import Entries
from .ops import Arith, Loop, Store


def countFlops(region):
  """The arithmetic the region performs, counted over the trip counts.

  One per arithmetic operation and one per accumulating store, which is the
  addition the store performs. A value that is spelled at several use sites is
  still one operation here: whether the compiler recomputes it is a question
  about the emitted expression, not about the arithmetic the kernel does.
  """
  total = 0
  for op in region.ops:
    if isinstance(op, Arith):
      total += 1
    elif isinstance(op, Store):
      total += 0 if op.accumulate is None else 1
    elif isinstance(op, Loop):
      total += countFlops(op.region) * _tripCount(op)
    else:
      for nested in op.regions():
        total += countFlops(nested)
  return total


def _tripCount(loop):
  if isinstance(loop.domain, Entries):
    return len(loop.domain)
  return loop.domain.size()
