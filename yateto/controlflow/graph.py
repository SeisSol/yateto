from ..ast.node import Node
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


class Expression(object):
  @classmethod
  def of(cls, node, memoryLayout, variables):
    """The statement a node states, over these operands."""
    return cls(node, memoryLayout, variables, node.indices, node.eqspp(),
               _productGroups(node), node.prefetch,
               getattr(node, 'optype', None), getattr(node, 'termTemplate', None),
               getattr(node, 'nodeTermIndices', None))

  def __init__(self, node, memoryLayout, variables, indices, eqspp, groups,
               prefetch, optype=None, termTemplate=None, nodeTermIndices=None):
    #: The node this was built from, for the generator that is to write it:
    #: which backend takes the statement, and what that backend is told
    #: beyond the operands, is read off it and off nothing else here.
    self.node = node
    self.memoryLayout = memoryLayout
    self._variables = variables
    #: What the statement computes -- the indices it is stated over and the
    #: entries it has values at.
    self.indices = indices
    self.eqspp = eqspp
    #: The index groups a product is formed over, where the statement is one.
    self.groups = groups
    #: The tensor to fetch while the statement runs, where one was assigned.
    self.prefetch = prefetch
    #: The operation an element-wise statement or a reduction applies, and how
    #: its operands are filled in around the immediates it was written with.
    self.optype = optype
    self.termTemplate = termTemplate
    self.nodeTermIndices = nodeTermIndices

  def fillTerms(self, terms):
    """The operands in their original order, with the immediates put back."""
    return [terms[index] if template is None else template
            for template, index in zip(self.termTemplate, self.nodeTermIndices)]

  def variables(self):
    return set([var.viewed() for var in self._variables])

  def variableList(self):
    return self._variables

  def maySubstitute(self, when, by):
    operands = [var.substituted(when, by) for var in self._variables]
    layouts = [operand.memoryLayout for operand in operands]
    c1 = all(layouts[i].isCompatible(var.eqspp) for i,var in enumerate(self._variables))
    return c1 and self.mayReadOperands(layouts)

  def mayReadOperands(self, layouts):
    """Whether the operands can be read as matrices, laid out like this.

    Only a product asks anything: everything else reads an operand entry by
    entry, and no layout stops that.
    """
    if self.groups is None:
      return True
    m, n, k = self.groups
    return mayFuseGroups(self._variables[0].indices, (m, k), layouts[0]) \
       and mayFuseGroups(self._variables[1].indices, (k, n), layouts[1])

  def substituted(self, when, by, memoryLayout):
    return Expression(self.node, memoryLayout,
                      [var.substituted(when, by) for var in self._variables],
                      self.indices, self.eqspp, self.groups, self.prefetch,
                      self.optype, self.termTemplate, self.nodeTermIndices)

  def resultCompatible(self, result):
    c1 = result.memoryLayout.isCompatible(self.eqspp)
    c2 = self.groups is None or mayFuseGroups(
      self.indices, (self.groups[0], self.groups[1]), result.memoryLayout)
    return c1 and c2

  def __str__(self):
    return '{}({})'.format(type(self.node).__name__, ', '.join([str(var) for var in self._variables]))

  def setWritable(self, name):
    for v in self._variables:
      v.setWritable(name)

class ProgramAction(object):
  def __init__(self, result, term, add, scalar=None, condition=True):
    self.result = result
    self.term = term
    self.add = add
    self.scalar = scalar
    self.condition = condition

  def isRHSExpression(self):
    return isinstance(self.term, Expression)

  def isRHSVariable(self):
    return not self.isRHSExpression()

  def isCompound(self):
    return self.add

  def hasTrivialScalar(self):
    # One, because a scaling is a multiplication: Elementwise.scalingOperands
    # only recognises one under ops.Mul(), which is also the only ring the
    # contraction backends implement. A scaling under another ring would need
    # that operation's neutral element here.
    return self.scalar is None or self.scalar == ops.Mul().neutral()

  def variables(self):
    V = self.term.variables()
    if self.add:
      V = V | self.result.variables()
    return V

  def guardVariables(self):
    """Variables read to evaluate this action's guard."""
    return self.getGuard().variables()

  def allVariables(self):
    return self.variables() | self.guardVariables()

  def maySubstitute(self, when, by, result = True, term = True):
    maySubsTerm = self.term.maySubstitute(when, by)
    maySubsResult = self.result.maySubstitute(when, by)

    rsubs = self.result.substituted(when, by) if result else self.result
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout) if term else self.term

    compatible = tsubs.resultCompatible(rsubs)

    return (not term or maySubsTerm) and (not result or maySubsResult) and compatible

  def substituted(self, when, by, guard=None, result = True, term = True):
    """Replace `when` by `by`.

    `guard` is passed only when the substitution redirects this action's *write
    target* onto a variable another action writes under that guard; the action
    then inherits it. A read substitution leaves the guard alone -- conjoining
    there would restrict statements that are not themselves conditional.
    """
    rsubs = self.result.substituted(when, by) if result else self.result
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout) if term else self.term
    gsubs = self.condition if guard is None else (self.getGuard() & guard)
    return ProgramAction(rsubs, tsubs, self.add, self.scalar, gsubs)

  def setVariablesWritable(self, name):
    self.result.setWritable(name)
    self.term.setWritable(name)

  def getGuard(self):
    return Guard.coerce(self.condition)



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
