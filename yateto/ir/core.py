from .affine import Affine, Index


class Buffer:
  """A named piece of storage the generated code addresses.

  A buffer is not a value: it is written through, and two loads from it are
  two loads. Its memory layout answers where an entry sits and whether it is
  stored at all, which is what lets an address be formed from coordinates
  rather than from text.
  """

  __slots__ = ('name', 'datatype', 'memoryLayout', 'eqspp')

  def __init__(self, name, datatype, memoryLayout, eqspp=None):
    self.name = str(name)
    self.datatype = datatype
    self.memoryLayout = memoryLayout
    #: Which entries the operand has a value at, which is not the same question
    #: as which entries the layout keeps room for: a layout may store an entry
    #: that is structurally zero, and what is in that room is nobody's promise.
    self.eqspp = eqspp

  @classmethod
  def fromDescription(cls, description):
    """The buffer behind a tensor description, as the code generators hand it over."""
    return cls(description.name, description.datatype, description.memoryLayout,
               description.eqspp)

  def __repr__(self):
    return f'Buffer({self.name})'


class Entries:
  """An iteration domain given as the entries to visit.

  A sparse operand has no address expression to run a loop over, so the loop
  that reads it states its entries instead. Emission does not accept such a
  loop; a pass has to pin the indices first, which is what turns the entries
  into constant addresses.
  """

  __slots__ = ('entries',)

  def __init__(self, entries):
    self.entries = [tuple(int(coordinate) for coordinate in entry) for entry in entries]

  def __len__(self):
    return len(self.entries)

  def __repr__(self):
    return f'Entries({len(self.entries)})'


class Region:
  """An ordered list of operations."""

  __slots__ = ('ops',)

  def __init__(self, ops=None):
    self.ops = list(ops) if ops else []

  def append(self, op):
    self.ops.append(op)
    return op

  def extend(self, ops):
    self.ops.extend(ops)

  def __iter__(self):
    return iter(self.ops)

  def __len__(self):
    return len(self.ops)

  def walk(self):
    """Every op in the region and in the regions its ops hold, outermost first."""
    for op in self.ops:
      yield op
      for region in op.regions():
        yield from region.walk()


class Op:
  """One operation.

  An operation that produces a value *is* that value (see `ValueOp`), so an
  operand is the operation that computed it. There is at most one result, and
  every use is a reference to the producing operation, which makes counting
  uses a walk rather than a bookkeeping exercise.
  """

  def operands(self):
    """The values this op reads."""
    return ()

  def regions(self):
    """The regions this op holds."""
    return ()

  def indices(self):
    """The loop indices this op introduces."""
    return ()


class ValueOp(Op):
  """An operation that yields a scalar."""

  def __init__(self, datatype, name=None):
    self.datatype = datatype
    #: What to call the local, should the emitter give this value one.
    self.name = name
    #: Force a local even where the value would inline. A value the emitter
    #: would otherwise spell at its use site is read again at every use, which
    #: for anything but a literal is a repeated read of memory the kernel may
    #: itself write.
    self.materialize = False


class Builder:
  """Appends to one region."""

  __slots__ = ('_region',)

  def __init__(self, region=None):
    self._region = region if region is not None else Region()

  def add(self, op):
    self._region.append(op)
    return op

  def region(self):
    return self._region

  def nested(self):
    return Builder()
