from ..codegen.common import INDEX_PREFIX
from ..ops import CBinaryOperatorMixin
from .address import address
from .core import ValueOp
from .ops import (Arith, Call, Const, Fold, If, Load, Loop, Memset, Pointer,
                  Read, Scope, Store, Yield)


class CppEmitter:
  """Writes a region out as C++.

  A value is spelled where it is used unless it has to be a local. It has to
  be one when it is read more than once, when it is read from a loop it is not
  defined in, or when it asked to be: in each of those cases spelling it again
  would be a second read of something the kernel may itself write, or a second
  evaluation of an expression that was meant to be evaluated once.
  """

  def __init__(self, cpp, routineCache=None, prefix=INDEX_PREFIX):
    self._cpp = cpp
    self._routineCache = routineCache
    self._prefix = prefix
    self._names = {}
    self._counter = 0

  def emit(self, region):
    self._names = {}
    self._counter = 0
    self._locals = self._findLocals(region)
    self._emitRegion(region)

  def _findLocals(self, region, depth=0, defined=None, uses=None, locals=None):
    """Which values need a local of their own."""
    if defined is None:
      defined, uses, locals = {}, {}, set()
    for op in region.ops:
      for operand in op.operands():
        uses[id(operand)] = uses.get(id(operand), 0) + 1
        if uses[id(operand)] > 1 or defined.get(id(operand), depth) != depth:
          # a literal and a name are spelled again wherever they are used;
          # there is nothing to recompute and nothing to read twice
          if not isinstance(operand, (Const, Read)):
            locals.add(id(operand))
      if isinstance(op, ValueOp):
        defined[id(op)] = depth
        if op.materialize:
          locals.add(id(op))
      for nested in op.regions():
        self._findLocals(nested, depth + 1, defined, uses, locals)
    return locals

  def _emitRegion(self, region):
    for op in region.ops:
      self._emitOp(op)

  def _emitOp(self, op):
    if isinstance(op, Fold):
      self._emitFold(op)
      return
    if isinstance(op, Pointer):
      name = self._name(op)
      offset = address(op.buffer.memoryLayout, op.coords, op.axes)
      start = '' if offset.isConstant() and offset.constant() == 0 \
              else f' + {offset.ccode(self._prefix)}'
      const = 'const' if op.const else ''
      self._cpp(f'{op.datatype.ctype()} {const}* {name} = {op.buffer.name}{start};')
      return
    if isinstance(op, Call):
      op.flops = op.generate(self._cpp, self._routineCache)
      return
    if isinstance(op, Yield):
      # what the region contributes is spelled by whoever holds it
      return
    if isinstance(op, ValueOp):
      if id(op) in self._locals:
        name = self._name(op)
        self._cpp(f'{op.datatype.ctype()} const {name} = {self._expression(op)};')
      return
    if isinstance(op, Store):
      self._cpp(self._store(op))
      return
    if isinstance(op, Memset):
      pointer = op.buffer.name if op.offset == 0 else f'{op.buffer.name} + {op.offset}'
      self._cpp.memset(pointer, op.count, op.buffer.datatype.ctype())
      return
    if isinstance(op, If):
      with self._cpp.If(op.condition):
        self._emitRegion(op.region)
      return
    if isinstance(op, Scope):
      with self._cpp.AnonymousScope():
        self._emitRegion(op.region)
      return
    if isinstance(op, Loop):
      self._emitLoop(op)
      return
    raise NotImplementedError(f'{type(op).__name__} has no C++ spelling.')

  def _emitLoop(self, loop):
    if loop.isUnrollable():
      raise ValueError(
        'a loop that states its entries has no C++ spelling; unroll it first.')
    if loop.collapse is not None:
      self._cpp(f'#pragma omp simd collapse({loop.collapse})')
    elif loop.simd:
      self._cpp('#pragma omp simd')
    index = f'{self._prefix}{loop.index[0].name}'
    domain = loop.domain
    with self._cpp.For(f'int {index} = {domain.start}; {index} < {domain.stop}; ++{index}'):
      self._emitRegion(loop.region)

  def _emitFold(self, fold):
    """An accumulator, a loop over the index, and a combination per step."""
    name = self._name(fold)
    neutral = fold.operation.neutralLiteral(fold.datatype)
    self._cpp(f'{fold.datatype.ctype()} {name} = {neutral};')
    index = f'{self._prefix}{fold.index.name}'
    domain = fold.domain
    with self._cpp.For(
        f'int {index} = {domain.start}; {index} < {domain.stop}; ++{index}'):
      self._emitRegion(fold.region)
      self._cpp(self._combine(name, fold.operation, self._value(fold.yielded())))

  def _combine(self, target, operation, value):
    """`target <op>= value`, however that operation is spelled."""
    if operation is None:
      return f'{target} = {value};'
    if isinstance(operation, CBinaryOperatorMixin):
      return f'{target} {operation.cppname()}= {value};'
    return f'{target} = {operation.callstr(target, value)};'

  def _store(self, store):
    return self._combine(self._access(store.buffer, store.coords),
                         store.accumulate, self._value(store.value))

  def _access(self, buffer, coords):
    return f'{buffer.name}[{address(buffer.memoryLayout, coords).ccode(self._prefix)}]'

  def _name(self, value):
    if id(value) not in self._names:
      self._names[id(value)] = value.name or f'{self._prefix}v{self._counter}'
      self._counter += 1
    return self._names[id(value)]

  def _value(self, value):
    """How a use of this value reads: its name, or the expression itself."""
    if id(value) in self._locals:
      return self._name(value)
    return self._expression(value)

  def _expression(self, value):
    if isinstance(value, Const):
      return value.datatype.literal(value.value)
    if isinstance(value, Read):
      return value.expression
    if isinstance(value, Load):
      return self._access(value.buffer, value.coords)
    if isinstance(value, Arith):
      return self._arith(value)
    raise NotImplementedError(f'{type(value).__name__} has no C++ spelling.')

  def _arith(self, value):
    args = [self._operand(arg) for arg in value.args]
    if isinstance(value.operation, CBinaryOperatorMixin) and len(args) == 2:
      return f'{args[0]} {value.operation.cppname()} {args[1]}'
    return value.operation.callstr(*args)

  def _operand(self, value):
    """An operand of an operation, parenthesised where it has to be.

    A name, a literal and an entry of a buffer bind tighter than any operator
    and are written as they are; anything built from an operator is bracketed
    rather than compared against a precedence table.
    """
    code = self._value(value)
    if id(value) in self._locals or isinstance(value, (Const, Read, Load)):
      return code
    return f'({code})'
