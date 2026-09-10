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
    self.writable = writable

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

  @classmethod
  def fromNode(cls, name, node):
    return cls(name, node.memoryLayout(), node.eqspp())

class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout, eqspp, is_compute_constant=False, is_temporary=False, values=None, datatype=None, addressing=None, tensor=None, writable=False):
    super().__init__(name, memoryLayout, eqspp, is_compute_constant, is_temporary, values, datatype, addressing, tensor, writable)
    self.indices = indices

  @classmethod
  def statement(cls, indices, eqspp, datatype):
    """What a statement says about an operand, before it is read anywhere."""
    return cls(None, indices, None, eqspp, datatype=datatype)

  @classmethod
  def fromNode(cls, var, node):
    """The operand: what the node says about it, read from where the variable is."""
    return cls.statement(node.indices, node.eqspp(),
                         node.viewed().datatype).readFrom(var)

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
