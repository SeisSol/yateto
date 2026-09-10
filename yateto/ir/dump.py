"""A region, written down.

Not to be read back -- to be looked at. A generated kernel says what it does
and not what it was made of, and the two are several passes apart: a statement
stated over tensors becomes loops, the loops are fused, what a loop computes
once is read back rather than stored. Putting both down next to the code lets
a line of it be traced to the statement it came from.

Values are numbered in the order they are defined, so a use points backwards
at a line that is above it.
"""

from .core import Region, ValueOp
from .ops import (Arith, Call, Const, Fold, If, Load, Loop, Memset, Pointer,
                  Read, Scope, Store, Yield)
from .tensor import TensorOp


def dump(region):
  """The region as lines, one operation each, nested regions indented."""
  return _Dump().lines(region, 0)


class _Dump:
  def __init__(self):
    self._names = {}

  def lines(self, region, depth):
    written = []
    for op in region.ops:
      written.append('  ' * depth + self._op(op))
      for nested in op.regions():
        written.extend(self.lines(nested, depth + 1))
    return written

  def _value(self, op):
    """What a use of a value points at: the line that defined it."""
    if id(op) not in self._names:
      self._names[id(op)] = f'%{len(self._names)}'
    return self._names[id(op)]

  def _entry(self, op):
    coords = ', '.join(coord.ccode() for coord in op.coords)
    return f'{op.buffer.name}[{coords}]'

  def _op(self, op):
    if isinstance(op, TensorOp):
      terms = ', '.join(getattr(term, 'name', str(term)) for term in op.terms)
      factor = '' if op.alpha == 1.0 else f' * {op.alpha}'
      return (f'{op.result.name} {"+=" if op.add else "="} '
              f'{type(op).__name__}({terms}){factor}')
    if isinstance(op, Const):
      return f'{self._value(op)} = const {op.value}'
    if isinstance(op, Read):
      return f'{self._value(op)} = read {op.expression}'
    if isinstance(op, Load):
      return f'{self._value(op)} = load {self._entry(op)}'
    if isinstance(op, Arith):
      args = ' '.join(self._value(arg) for arg in op.operands())
      return f'{self._value(op)} = {op.operation} {args}'
    if isinstance(op, Store):
      how = '=' if op.accumulate is None else f'{op.accumulate}='
      return f'store {self._entry(op)} {how} {self._value(op.value)}'
    if isinstance(op, Memset):
      return f'memset {op.buffer.name}[{op.offset}:{op.offset + op.count}] = {op.value}'
    if isinstance(op, Loop):
      indices = ', '.join(index.name for index in op.index)
      simd = ' simd' if op.simd else ''
      if op.isUnrollable():
        return f'loop {indices} over {len(op.domain.entries)} entries{simd}'
      return (f'loop {indices} in [{op.domain.start}, {op.domain.stop})'
              f'{simd}')
    if isinstance(op, Fold):
      return (f'{self._value(op)} = fold {op.index.name} in '
              f'[{op.domain.start}, {op.domain.stop}) over {op.operation}')
    if isinstance(op, Yield):
      return f'yield {self._value(op.value)}'
    if isinstance(op, Pointer):
      return f'{self._value(op)} = pointer into {op.buffer.name}'
    if isinstance(op, Call):
      reads = ', '.join(buffer.name for buffer in op.reads)
      writes = ', '.join(buffer.name for buffer in op.writes)
      return f'call reading {reads or "nothing"} writing {writes or "nothing"}'
    if isinstance(op, If):
      return f'if {op.condition.ccode()}'
    if isinstance(op, Scope):
      return 'scope'
    return repr(op)
