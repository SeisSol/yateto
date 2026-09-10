from .. import ops
from ..guard import Guard
from ..ir.tensor import mayFuseGroups
from collections import OrderedDict
from typing import Dict, List

def _productGroups(node):
  """The index groups a product is formed over, or None where it is no product."""
  from ..ast.node import LoopOverGEMM
  if not isinstance(node, LoopOverGEMM):
    return None
  return node.m(), node.n(), node.k()


class ProgramAction(object):
  """One statement of the kernel, stated over tensors.

  A destination, the operands read into it, whether the value is added to what
  is already there, a factor, and the guard it runs under. And, for whoever is
  to write it, which kind of statement it is and what that kind says beyond its
  operands. The tree it was first written down in answers none of it.
  """

  #: What a statement says beyond its operands, per kind. Read off the node
  #: once, where the node still is, and never again.
  _FACTS = ('groups', 'prefetch', 'optype', 'termTemplate', 'nodeTermIndices',
            'loopIndices', 'transA', 'transB', 'sumIndex', 'datatype')

  @classmethod
  def copy(cls, result, operand, add, scalar=None, condition=True):
    """A statement that reads one operand as it stands."""
    return cls('Copy', result, [operand], operand.indices, operand.eqspp,
               add, scalar, condition)

  @classmethod
  def of(cls, node, result, operands, add, scalar=None, condition=True):
    """The statement a node states, over these operands."""
    ask = lambda name: getattr(node, name)() if hasattr(node, name) else None
    return cls(type(node).__name__, result, operands, node.indices, node.eqspp(),
               add, scalar, condition,
               groups=_productGroups(node),
               prefetch=node.prefetch,
               optype=getattr(node, 'optype', None),
               termTemplate=getattr(node, 'termTemplate', None),
               nodeTermIndices=getattr(node, 'nodeTermIndices', None),
               loopIndices=ask('loopIndices'),
               transA=ask('transA'),
               transB=ask('transB'),
               sumIndex=ask('sumIndexName'),
               datatype=node.datatype)

  def __init__(self, kind, result, operands, indices, eqspp, add, scalar=None,
               condition=True, **facts):
    #: Which kind of statement it is, which is what decides who writes it.
    self.kind = kind
    self.result = result
    self.operands = operands
    #: What the statement computes -- the indices it is stated over and the
    #: entries it has values at.
    self.indices = indices
    self.eqspp = eqspp
    self.add = add
    self.scalar = scalar
    #: The guard it runs under. A guard, not something a guard is made of: it
    #: is asked for far more often than it is set.
    self.condition = Guard.coerce(condition)
    for name in self._FACTS:
      setattr(self, name, facts.get(name))

  def isCopy(self):
    """Whether the statement reads one operand as it stands."""
    return self.kind == 'Copy'

  def copied(self):
    """The operand a plain copy reads."""
    return self.operands[0]

  def isCompound(self):
    return self.add

  def hasTrivialScalar(self):
    # One, because a scaling is a multiplication: Elementwise.scalingOperands
    # only recognises one under ops.Mul(), which is also the only ring the
    # contraction backends implement. A scaling under another ring would need
    # that operation's neutral element here.
    return self.scalar is None or self.scalar == ops.Mul().neutral()

  def getGuard(self):
    return self.condition

  def fillTerms(self, terms):
    """The operands in their original order, with the immediates put back."""
    return [terms[index] if template is None else template
            for template, index in zip(self.termTemplate, self.nodeTermIndices)]

  def reads(self):
    """The storage the operands reach."""
    return set([var.viewed() for var in self.operands])

  def variables(self):
    V = self.reads()
    if self.add:
      V = V | self.result.variables()
    return V

  def guardVariables(self):
    """Variables read to evaluate this action's guard."""
    return self.getGuard().variables()

  def allVariables(self):
    return self.variables() | self.guardVariables()

  def setVariablesWritable(self, name):
    self.result.setWritable(name)
    for operand in self.operands:
      operand.setWritable(name)

  def mayReadOperands(self, layouts):
    """Whether the operands can be read as matrices, laid out like this.

    Only a product asks anything: everything else reads an operand entry by
    entry, and no layout stops that.
    """
    if self.groups is None:
      return True
    m, n, k = self.groups
    return mayFuseGroups(self.operands[0].indices, (m, k), layouts[0]) \
       and mayFuseGroups(self.operands[1].indices, (k, n), layouts[1])

  def resultCompatible(self, result):
    """Whether what this computes fits a destination laid out like that."""
    c1 = result.memoryLayout.isCompatible(self.eqspp)
    c2 = self.groups is None or mayFuseGroups(
      self.indices, (self.groups[0], self.groups[1]), result.memoryLayout)
    return c1 and c2

  def standsUp(self):
    """Whether this is a statement that can still be generated.

    Three things: every operand is read from storage keeping the entries it
    has values at, the operands of a product can still be read as matrices,
    and what the statement computes fits the destination it is written to.
    The same three a rewrite has to leave true, asked of the statement as it
    stands.
    """
    layouts = [operand.memoryLayout for operand in self.operands]
    return all(layout.isCompatible(operand.eqspp)
               for layout, operand in zip(layouts, self.operands)) \
       and self.mayReadOperands(layouts) \
       and self.result.memoryLayout.isCompatible(self.result.eqspp) \
       and self.resultCompatible(self.result)

  def rhs(self):
    """What is read, as it would be written down."""
    if self.isCopy():
      return str(self.operands[0])
    return '{}({})'.format(self.kind, ', '.join(str(o) for o in self.operands))


class LiveSet:
  """Maps a variable to the guard under which it is live.

  Liveness is a may-analysis, so over-approximating is always safe. The lattice
  is deliberately coarse: joining two different guards yields "live
  unconditionally". That keeps the guard language purely conjunctive -- neither
  the join nor the kill needs a disjunction or a negation.
  """

  def __init__(self, data: dict):
    self.data = {k: Guard.coerce(v) for k, v in data.items()}

  def __sub__(self, other):
    """Remove variables written by an action.

    A variable that is only written under a guard survives where the guard does
    not hold, so it stays live; we widen it to "unconditionally live" rather
    than tracking the complement of the guard.
    """
    if isinstance(other, dict):
      other = LiveSet(other)

    result = dict(self.data)
    for var, guard in other.data.items():
      if var not in result:
        continue
      if guard.isAlways():
        del result[var]
      else:
        result[var] = Guard.always()

    return LiveSet(result)

  def __or__(self, other):
    if isinstance(other, dict):
      other = LiveSet(other)

    result = dict(self.data)
    for var, guard in other.data.items():
      if var in result and result[var] != guard:
        result[var] = Guard.always()
      elif var not in result:
        result[var] = guard

    return LiveSet(result)

  def __contains__(self, element):
    """``(var, guard) in live`` asks whether var is live anywhere `guard` holds."""
    if isinstance(element, tuple):
      var, guard = element
      if var not in self.data:
        return False
      return not (self.data[var] & Guard.coerce(guard)).isNever()
    return element in self.data

  def guardOf(self, var):
    return self.data.get(var, Guard.never())

  def variables(self):
    return set(self.data)

  def __repr__(self):
    return f'LiveSet({self.data})'
