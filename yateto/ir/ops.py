from .affine import Affine
from .core import Entries, Op, Region, ValueOp


class Const(ValueOp):
  """A literal.

  It has no address and is never read twice, so it is spelled wherever it is
  used and never becomes a local.
  """

  def __init__(self, value, datatype):
    super().__init__(datatype)
    self.value = value

  def __repr__(self):
    return f'Const({self.value})'


class Read(ValueOp):
  """A scalar that already has a name in the generated code.

  A kernel argument or member, that is. `hoist` reads it once into a local:
  as far as the compiler can tell, a store through one of the kernel's
  pointers may land on it, and reading it inside a loop would ask that
  question per iteration. Asked for rather than deduced, because naming
  something again costs nothing on its own.
  """

  def __init__(self, expression, datatype, name=None, hoist=True):
    super().__init__(datatype, name)
    self.expression = str(expression)
    self.materialize = hoist

  def __repr__(self):
    return f'Read({self.expression})'


class Arith(ValueOp):
  """An operation from `yateto.ops` applied to values."""

  def __init__(self, operation, args, datatype):
    super().__init__(datatype)
    self.operation = operation
    self.args = list(args)

  def operands(self):
    return tuple(self.args)

  def __repr__(self):
    return f'Arith({self.operation})'


class Load(ValueOp):
  """One entry of a buffer, named by its coordinates.

  The coordinates are logical, one per axis of the buffer's layout: turning
  them into an address is the layout's business and happens on the way out.
  A layout that stores nothing at a constant coordinate makes the load a
  zero, which is how a sparse operand loses its reads.
  """

  def __init__(self, buffer, coords, datatype=None):
    super().__init__(datatype if datatype is not None else buffer.datatype)
    self.buffer = buffer
    self.coords = tuple(Affine.of(coord) for coord in coords)

  def __repr__(self):
    return f'Load({self.buffer.name})'


class Store(Op):
  """Write a value to one entry of a buffer.

  `accumulate` is None for an assignment, or the operation the stored value is
  combined with what is already there.
  """

  def __init__(self, buffer, coords, value, accumulate=None):
    self.buffer = buffer
    self.coords = tuple(Affine.of(coord) for coord in coords)
    self.value = value
    self.accumulate = accumulate

  def operands(self):
    return (self.value,)

  def __repr__(self):
    return f'Store({self.buffer.name})'


class Memset(Op):
  """Zero `count` entries of a buffer, starting at `offset`.

  Addressed in storage rather than in coordinates: a run of entries that no
  operation writes need not be a box in the logical space, and zeroing it is
  cheapest as one call.
  """

  def __init__(self, buffer, offset, count):
    self.buffer = buffer
    self.offset = int(offset)
    self.count = int(count)

  def __repr__(self):
    return f'Memset({self.buffer.name}, {self.offset}, {self.count})'


class Loop(Op):
  """Run a region over an iteration domain.

  A `Range` domain binds one index and emits as a loop. An `Entries` domain
  binds one index per coordinate and states the entries to visit; it has no
  loop to emit and has to be unrolled first.
  """

  def __init__(self, indices, domain, region=None, simd=False, collapse=None):
    self.index = tuple(indices)
    self.domain = domain
    self.region = region if region is not None else Region()
    #: Ask for `#pragma omp simd` on this loop.
    self.simd = simd
    #: Ask for a `collapse(n)` clause, `n` being the depth of the nest that
    #: starts here. Only the outermost loop of a nest carries it.
    self.collapse = collapse
    if isinstance(domain, Entries):
      for entry in domain.entries:
        assert len(entry) == len(self.index), \
          'an entry states one coordinate per index of the loop'
    else:
      assert len(self.index) == 1, 'a range binds one index'

  def isUnrollable(self):
    return isinstance(self.domain, Entries)

  def regions(self):
    return (self.region,)

  def indices(self):
    return self.index

  def __repr__(self):
    return f'Loop({", ".join(index.name for index in self.index)})'


class Pointer(ValueOp):
  """Where a buffer starts, for whoever works on part of it.

  An external kernel is handed a pointer and works from there, so the offset
  is formed once, outside whatever loop moves it. Which axes the offset counts
  is the caller's to say: the ones the loop moves along.
  """

  def __init__(self, buffer, coords, axes=None, name=None, const=True):
    super().__init__(buffer.datatype, name)
    self.buffer = buffer
    self.coords = tuple(Affine.of(coord) for coord in coords)
    self.axes = axes
    self.const = const
    self.materialize = True

  def __repr__(self):
    return f'Pointer({self.buffer.name})'


class Call(Op):
  """A statement someone else writes.

  The IR says where the operands are and leaves the rest to the generator that
  claimed the statement. What arithmetic that generator performs is its own to
  report, and it reports it when it writes itself, since which generator takes
  a statement decides how much work it turns out to be.
  """

  def __init__(self, generate, reads=None, writes=None, operands=()):
    self.generate = generate
    #: The values the callee is handed. It names them in whatever it writes,
    #: so nothing may take them away.
    self._operands = list(operands)
    #: The buffers the callee touches, or None where it does not say. Not
    #: saying is not the same as touching nothing: a call that has not been
    #: asked what it reads may read anything, and nothing may be moved across
    #: it or dropped because of it.
    self.reads = None if reads is None else list(reads)
    self.writes = None if writes is None else list(writes)
    #: What the callee reported, once it has written itself.
    self.flops = None

  def states(self):
    """Whether this call says what it touches."""
    return self.reads is not None and self.writes is not None

  def touches(self):
    """The buffers, whichever way it touches them."""
    return list(self.reads or ()) + list(self.writes or ())

  def names(self):
    return {buffer.name for buffer in self.touches()}

  def operands(self):
    return tuple(self._operands)

  def __repr__(self):
    return 'Call'


class Yield(Op):
  """The value a region hands back to the operation that holds it."""

  def __init__(self, value):
    self.value = value

  def operands(self):
    return (self.value,)

  def __repr__(self):
    return 'Yield'


class Fold(ValueOp):
  """Fold an index away with an operation.

  The region computes what one step of the index contributes and ends on a
  `Yield` saying which value that is. The operation combines the
  contributions, starting from its own neutral element, and what is left when
  the index is gone is this value.

  A fold keeps a running value, which is the one thing here that is written
  more than once, so it always gets a local of its own.
  """

  def __init__(self, index, domain, operation, region=None, datatype=None,
               name=None):
    super().__init__(datatype, name)
    self.index = index
    self.domain = domain
    self.operation = operation
    self.region = region if region is not None else Region()
    self.materialize = True

  def yielded(self):
    """The value one step of the index contributes."""
    terminator = self.region.ops[-1]
    assert isinstance(terminator, Yield), 'a fold\'s region ends on a Yield'
    return terminator.value

  def regions(self):
    return (self.region,)

  def indices(self):
    return (self.index,)

  def __repr__(self):
    return f'Fold({self.index.name}, {self.operation})'


class If(Op):
  """A region that runs only where a condition holds.

  The condition is the guard itself and not the C++ that tests it. What it is
  decided on is then still a question that can be asked of the region -- the
  values a kernel reads to pick a branch are values it is handed, and how the
  test is spelled is emission's business like every other spelling.
  """

  def __init__(self, condition, region=None):
    self.condition = condition
    self.region = region if region is not None else Region()

  def regions(self):
    return (self.region,)

  def __repr__(self):
    return f'If({self.condition!r})'


class Scope(Op):
  """A region emitted inside braces of its own."""

  def __init__(self, region=None):
    self.region = region if region is not None else Region()

  def regions(self):
    return (self.region,)

  def __repr__(self):
    return 'Scope'
