from ..ast.node import Node
from .. import ops
from ..guard import Guard
from collections import OrderedDict
from typing import Dict, List

class Expression(object):
  def __init__(self, node, memoryLayout, variables):
    self.node = node
    self.memoryLayout = memoryLayout
    self._variables = variables

  @property
  def eqspp(self):
    return self.node.eqspp()

  def variables(self):
    return set([var.viewed() for var in self._variables])

  def variableList(self):
    return self._variables

  def maySubstitute(self, when, by):
    layouts = [var.substituted(when, by).memoryLayout for var in self._variables]
    c1 = all(layouts[i].isCompatible(var.eqspp) for i,var in enumerate(self._variables))
    c2 = self.node.argumentsCompatible(layouts)
    return c1 and c2

  def substituted(self, when, by, memoryLayout):
    return Expression(self.node, memoryLayout, [var.substituted(when, by) for var in self._variables])

  def resultCompatible(self, result):
    c1 = result.memoryLayout.isCompatible(self.eqspp)
    c2 = self.node.resultCompatible(result.memoryLayout)
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
