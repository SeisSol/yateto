from ..ast.node import Node, FusedGEMMs, LoopOverGEMM
from ..ast.indices import Indices
from collections import OrderedDict
from typing import Dict, List
from ..type import ScalarMixin

class Variable(object):
  def __init__(self, name, writable, memoryLayout, eqspp=None, tensor=None, is_temporary=False, datatype=None):
    self.name = name
    self.writable = writable
    self.tensor = tensor
    self._memoryLayout = memoryLayout
    self._eqspp = eqspp
    self.is_temporary = is_temporary
    self.datatype = datatype

  def variables(self):
    return {self}

  def maySubstitute(self, when, by):
    return self.substituted(when, by).memoryLayout().isCompatible(self.eqspp())

  def substituted(self, when, by, memoryLayout=None):
    return by if self == when else self

  def resultCompatible(self, result):
    return result.memoryLayout().isCompatible(self.eqspp())

  def isPassedByValue(self):
    """Whether this operand is handed over by value rather than by pointer."""
    return self.tensor is not None and self.tensor.isPassedByValue()

  def isGlobal(self):
    return self.tensor is not None and not self.tensor.temporary

  def isLocal(self):
    return not self.isGlobal() and (self.tensor is None or not self.tensor.temporary)

  def memoryLayout(self):
    return self._memoryLayout

  def eqspp(self):
    return self._eqspp

  def __hash__(self):
    return hash(self.name)

  def __str__(self):
    return self.name

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    isEq = self.name == other.viewed().name # and self._memoryLayout == other._memoryLayout
    assert not isEq or (self.writable == other.viewed().writable and self._memoryLayout == other.viewed()._memoryLayout)
    return isEq

  def setWritable(self, name):
    if self.name == name:
      self.writable = True

  def viewed(self):
    return self

class VariableView(object):
  def __init__(self, variable, memoryLayout, eqspp):
    self.variable = variable.viewed()
    self._memoryLayout = memoryLayout
    self._eqspp = eqspp

  @property
  def tensor(self):
    return self.variable.tensor

  @property
  def writable(self):
    return self.variable.writable

  @property
  def is_temporary(self):
    return self.variable.is_temporary

  def maySubstitute(self, when, by):
    return self.substituted(when, by).memoryLayout().isCompatible(self.eqspp())

  def substituted(self, when, by, memoryLayout=None):
    return by if self == when else self

  @property
  def datatype(self):
    return self.variable.datatype

  def isPassedByValue(self):
    return self.variable.isPassedByValue()

  def viewed(self):
    return self.variable

  def variables(self):
    return {self.variable}

  def resultCompatible(self, result):
    return result.memoryLayout().isCompatible(self.eqspp())

  def isGlobal(self):
    return self.variable.isGlobal()

  def isLocal(self):
    return self.variable.isLocal()

  def memoryLayout(self):
    return self._memoryLayout

  def eqspp(self):
    return self._eqspp

  def __hash__(self):
    return hash(self.variable.name)

  def __str__(self):
    return f'{self.variable.name}'

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    isEq = self.variable == other.viewed() and self._memoryLayout == other._memoryLayout
    return isEq

  def setWritable(self, name):
    self.variable.setWritable(name)

class Expression(object):
  def __init__(self, node, memoryLayout, variables):
    self.node = node
    self._memoryLayout = memoryLayout
    self._variables = variables

  def memoryLayout(self):
    return self._memoryLayout

  def eqspp(self):
    return self.node.eqspp()

  def variables(self):
    return set([var.viewed() for var in self._variables])

  def variableList(self):
    return self._variables

  def maySubstitute(self, when, by):
    layouts = [var.substituted(when, by).memoryLayout() for var in self._variables]
    c1 = all(layouts[i].isCompatible(var.eqspp()) for i,var in enumerate(self._variables))
    c2 = self.node.argumentsCompatible(layouts)
    return c1 and c2

  def substituted(self, when, by, memoryLayout):
    return Expression(self.node, memoryLayout, [var.substituted(when, by) for var in self._variables])

  def resultCompatible(self, result):
    c1 = result.memoryLayout().isCompatible(self.eqspp())
    c2 = self.node.resultCompatible(result.memoryLayout())
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
    return self.scalar is None or self.scalar == 1.0

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
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout()) if term else self.term

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
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout()) if term else self.term
    gsubs = self.condition if guard is None else (self.getGuard() & guard)
    return ProgramAction(rsubs, tsubs, self.add, self.scalar, gsubs)

  def setVariablesWritable(self, name):
    self.result.setWritable(name)
    self.term.setWritable(name)

  def getGuard(self):
    return Guard.coerce(self.condition)


# TODO: probably should be a subclass of ProgramAction
class FusedActions(object):
  def __init__(self):
    self._actions: List[ProgramAction] = []
    self._variables: List[Variable] = []
    self._adds: List[bool] = []
    self._scalars = []
    self._conditions = []

  def add(self, action: ProgramAction) -> None:
    if not isinstance(action.term.node, LoopOverGEMM):
      raise ValueError(f'fused actions are applied only to LoopOverGEMM, '
                       f'given: {type(action.term.node)}')

    self._actions.append(action)
    self._variables.append(action.result)
    self._variables.extend(action.term.variableList())
    self._adds.append(action.add)
    self._scalars.append(action.scalar)
    self._conditions.append(action.condition)

  def gen_program_action(self) -> ProgramAction:
    last_action: ProgramAction = self._actions[-1]
    return ProgramAction(result=last_action.result,
                         term=self._gen_expr(),
                         add=self._adds,
                         scalar=self._scalars,
                         condition=self._conditions)

  def _gen_expr(self) -> Expression:
    node = FusedGEMMs()
    for action in self._actions:
      node.add(action.term.node)

    result: Variable = self._actions[-1].result
    return Expression(node=node,
                      memoryLayout=result.memoryLayout(),
                      variables=self._variables)

  def is_empty(self) -> bool:
    return len(self._actions) == 0


class ProgramPoint(object):
  def __init__(self, action):
    self.action = action
    self.live = None
    self.initBuffer = None
    self.bufferMap = None


class Guard:
  """A conjunction of literals over condition values.

  A literal is a ``(variable, version)`` pair mapped to the polarity the value
  must have. Versions matter because a condition tensor may be written inside
  the kernel: two reads separated by a write denote different values and must
  not be treated as the same literal.

  Guards form a meet-semilattice under conjunction. That is all the guard
  language needs -- ``assignIf`` produces a single positive literal, and nesting
  conjoins. Disjunction would require a richer representation (a truth-table
  bitmask over the condition variables is the cheapest one that stays exact),
  but nothing produces a disjunctive guard.
  """

  __slots__ = ('_literals', '_never')

  def __init__(self, literals=None, never=False):
    self._literals = dict(literals) if literals else dict()
    self._never = never

  @classmethod
  def always(cls):
    return cls()

  @classmethod
  def never(cls):
    return cls(never=True)

  @classmethod
  def literal(cls, variable, version=0, polarity=True):
    return cls({(variable, version): polarity})

  @classmethod
  def coerce(cls, value):
    if isinstance(value, Guard):
      return value
    if isinstance(value, bool):
      return cls.always() if value else cls.never()
    if isinstance(value, list):
      # FusedActions carry one guard per fused GEMM
      guard = cls.always()
      for entry in value:
        guard = guard & cls.coerce(entry)
      return guard
    return cls.literal(value)

  def isAlways(self):
    return not self._never and len(self._literals) == 0

  def isNever(self):
    return self._never

  def __and__(self, other):
    other = Guard.coerce(other)
    if self._never or other._never:
      return Guard.never()
    literals = dict(self._literals)
    for key, polarity in other._literals.items():
      if literals.get(key, polarity) != polarity:
        return Guard.never()
      literals[key] = polarity
    return Guard(literals)

  __rand__ = __and__

  def implies(self, other):
    """Whether this guard holds only where `other` does.

    Exact for conjunctions: a superset of literals is the stronger formula.
    """
    other = Guard.coerce(other)
    if self._never or other.isAlways():
      return True
    if other._never:
      return False
    return all(self._literals.get(key, None) is polarity
               for key, polarity in other._literals.items())

  def literals(self):
    """The literals as ``(variable, version, polarity)`` triples, in a stable order."""
    return [(var, version, polarity)
            for (var, version), polarity in sorted(self._literals.items(),
                                                   key=lambda kv: (str(kv[0][0]), kv[0][1]))]

  def variables(self):
    return {var for var, _ in self._literals if isinstance(var, Variable)}

  def _key(self):
    return (self._never, frozenset(self._literals.items()))

  def __eq__(self, other):
    return isinstance(other, Guard) and self._key() == other._key()

  def __hash__(self):
    return hash(self._key())

  def __bool__(self):
    raise TypeError('A Guard is not a bool; use isAlways()/isNever()/implies().')

  def ccode(self):
    if self.isAlways():
      return 'true'
    if self.isNever():
      return 'false'
    # a by-value operand is named directly, a by-pointer one is dereferenced
    printvar = lambda var: f'{var}' if var.isPassedByValue() \
                           else f'{var}[{var.memoryLayout().addressString(Indices())}]'
    formatlit = lambda var, polarity: printvar(var) if polarity else f'!{printvar(var)}'
    return ' && '.join(f'({formatlit(var, polarity)})'
                       for var, _, polarity in self.literals())

  def __repr__(self):
    if self.isAlways():
      return 'Guard(always)'
    if self.isNever():
      return 'Guard(never)'
    body = ', '.join(f'{"" if polarity else "~"}{var}@{version}'
                     for var, version, polarity in self.literals())
    return f'Guard({body})'


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


class FusedProgramPoint(ProgramPoint):
  def __init__(self, action: FusedActions):
    super().__init__(action.gen_program_action())
