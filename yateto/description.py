"""A tensor operand, as a statement states it.

Two things meet in one object and it is worth keeping them apart. What the
statement says about the operand -- the index tuple it is read over, which
entries it has values at, the type of those entries -- and where it is read
from: a name, a layout, the tensor of the caller's that stands behind it, and
whether the kernel writes it. `readFrom` is that split, said once.
"""

class TensorDescription(object):
  def __init__(self, name, memoryLayout, eqspp, is_compute_constant=False, is_temporary=False, values=None, datatype=None, addressing=None, tensor=None, writable=False):
    """

    Args:
      name (str): tensor's symbol name
      memoryLayout:
      eqspp:
      is_compute_constant (bool): if true then sparsity patterns and numerical values of tensor
          elements are known at compile time
      is_temporary (bool): if true then the description is for a temporary tensor which
          usually results from a result of an intermediate computation
      values (Union[np.ndarray, None]): the values of the compute_constant tensor, if they are known at compile time
      datatype (Datatype): the datatype of the tensor elements
      addressing (AddressingMode): the addressing mode for the tensor
      tensor (Union[Tensor, None]): the tensor this names, where it names one.
          A statement is described in the kernel's own terms; which tensor of
          the caller's stands behind a name is what the kernel's interface is
          built from, and only the outermost operands have one.
      writable (bool): whether the kernel writes what this names, anywhere
    """
    self.name = name
    self.memoryLayout = memoryLayout
    self.eqspp = eqspp
    self.is_compute_constant = is_compute_constant
    self.is_temporary = is_temporary
    self.values = values
    self.datatype = datatype
    self.addressing = addressing
    self.tensor = tensor
    #: What it names a slice of, where it names one. Set before the flag
    #: below, because whether the kernel writes a slice is asked of what it
    #: slices.
    self._views = None
    self.writable = writable

  @property
  def writable(self):
    # Asked of the storage: whether the kernel writes a slice of something is
    # whether it writes that something.
    return self.viewed()._writable

  @writable.setter
  def writable(self, value):
    self.viewed()._writable = value

  def viewed(self):
    """What it names a slice of, which for most operands is itself."""
    return self._views if self._views is not None else self

  def isView(self):
    return self._views is not None

  def setWritable(self, name):
    if self.name == name:
      self.writable = True

  def isPassedByValue(self):
    """Whether this operand is handed over by value rather than by pointer."""
    return self.tensor is not None and self.tensor.isPassedByValue()

  def variables(self):
    """The storage this reaches, which for a slice is what it slices."""
    return {self.viewed()}

  def resultCompatible(self, result):
    """Whether what this holds fits in a destination laid out like that."""
    return result.memoryLayout.isCompatible(self.eqspp)

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    # Two operands of the same name name the same storage. Whether the tensors
    # behind that name agree is a property of the kernel signature and is
    # reported there, with the context needed for a useful message.
    if not isinstance(other, TensorDescription):
      return NotImplemented
    if self.name != other.viewed().name:
      return False
    # A slice is the same as something else only where that something names
    # the same slice.
    return not self.isView() or self.memoryLayout == other.memoryLayout

  def isGlobal(self):
    """Whether the caller hands this over, which is what puts it in the interface."""
    return self.tensor is not None and not self.tensor.temporary

  def isLocal(self):
    """Whether the kernel is the only one who ever sees it."""
    return not self.isGlobal() and (self.tensor is None or not self.tensor.temporary)

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name

class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout, eqspp, is_compute_constant=False, is_temporary=False, values=None, datatype=None, addressing=None, tensor=None, writable=False):
    super().__init__(name, memoryLayout, eqspp, is_compute_constant, is_temporary, values, datatype, addressing, tensor, writable)
    self.indices = indices

  @classmethod
  def view(cls, operand, memoryLayout, eqspp):
    """A slice of what `operand` names: the same storage, another layout.

    It stands for the same tensor and answers for the same name -- a slice of
    C is written into C -- but for fewer of its entries, which is what tells
    two slices of one operand apart.
    """
    base = operand.viewed()
    sliced = cls(base.name, base.indices, memoryLayout, eqspp,
                 base.is_compute_constant, base.is_temporary, base.values,
                 base.datatype, base.addressing, base.tensor, base.writable)
    sliced._views = base
    return sliced

  def maySubstitute(self, when, by):
    """Whether it may be read from `by` instead, and still be read at all."""
    return self.substituted(when, by).memoryLayout.isCompatible(self.eqspp)

  def substituted(self, when, by, memoryLayout=None):
    """Read from `by` where this is the operand being replaced.

    What the statement says about the operand stays with the statement: it
    still reads the entries it read, over the indices it read them with. Only
    where it reads them from changes.
    """
    return self.readFrom(by) if self == when else self

  @classmethod
  def statement(cls, indices, eqspp, datatype):
    """What a statement says about an operand, before it is read anywhere."""
    return cls(None, indices, None, eqspp, datatype=datatype)

  def readFrom(self, source):
    """The same operand, read from `source`'s storage instead of its own.

    What the statement says about the operand stays: the index tuple it is
    read over, the entries it has values at, and the type of those entries
    are the statement's and not the storage's. Everything about where it is
    read comes from the storage -- the name, the layout, which of the
    caller's tensors stands behind it, whether the kernel writes it, and
    whether its values are known before the kernel runs.

    Which is the same split the description is built by, said once instead of
    once per way of building one.
    """
    return IndexedTensorDescription(
      source.name, self.indices, source.memoryLayout, self.eqspp,
      source.is_compute_constant, source.is_temporary, source.values,
      self.datatype, source.addressing, source.tensor, source.writable)

  @classmethod
  def fromVar(cls, var, indices):
    """The operand, over the indices it is read with and nothing else said."""
    return cls.statement(indices, var.eqspp, var.datatype).readFrom(var)
