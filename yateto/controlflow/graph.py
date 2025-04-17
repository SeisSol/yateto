from ..ast.node import Node, FusedGEMMs, LoopOverGEMM
from ..ast.indices import Indices
from collections import OrderedDict
from typing import Dict, List
from ..type import Scalar

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

  def isGlobal(self):
    return self.tensor is not None

  def isLocal(self):
    return not self.isGlobal()

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
    isEq = self.name == other.name
    assert not isEq or (self.writable == other.writable and self._memoryLayout == other._memoryLayout)
    return isEq

  def setWritable(self, name):
    if self.name == name:
      self.writable = True


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
    return set([var for var in self._variables])

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

  def maySubstitute(self, when, by, result = True, term = True):
    maySubsTerm = self.term.maySubstitute(when, by)
    maySubsResult = self.result.maySubstitute(when, by)

    rsubs = self.result.substituted(when, by) if result else self.result
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout()) if term else self.term

    compatible = tsubs.resultCompatible(rsubs)

    return (not term or maySubsTerm) and (not result or maySubsResult) and compatible

  def substituted(self, when, by, result = True, term = True):
    rsubs = self.result.substituted(when, by) if result else self.result
    tsubs = self.term.substituted(when, by, rsubs.memoryLayout()) if term else self.term
    return ProgramAction(rsubs, tsubs, self.add, self.scalar, self.condition)

  def setVariablesWritable(self, name):
    self.result.setWritable(name)
    self.term.setWritable(name)


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

# a rather primitive CNF implementation.
# do not overuse (i.e. avoid conditional assigns where possible)

class CNFClause:
  def __init__(self, variables):
    if isinstance(variables, list):
      self.variables = {var: True for var in variables}
    else:
      self.variables = variables
    self.fulfilled = False
  
  def negateVariables(self):
    return {var: ~self.variables[var] for var in self.variables}

  def unite(self, clause):
    output = CNFClause([])
    for v in self.variables:
      if v in clause and clause.variables[v] != self.variables[v]:
        output.fulfilled = True
    if not output.fulfilled:
      output.variables = {**self.variables, **clause.variables}
    return output
  
  def __repr__(self):
    formatvar = lambda name: f'{name}' if self.variables[name] else f'~{name}'
    return f'[{", ".join(formatvar(var) for var in self.variables)}]'
  
  def ccode(self):
    # for now, only allow scalarly-indexed variables
    printvar = lambda var: f'{var}' if isinstance(var, Scalar) else f'{var}[{var.memoryLayout().addressString(Indices())}]'
    formatvar = lambda name: f'{printvar(name)}' if self.variables[name] else f'!{printvar(name)}'
    return f'({" || ".join(formatvar(var) for var in self.variables)})'

class CNFCondition:
  def __init__(self, data):
    if isinstance(data, bool):
      if data == True:
        self.clauses = []
      elif data == False:
        self.clauses = [CNFClause([])]
    else:
      self.clauses = [CNFClause([data])]
  
  def _prune(self):
    newclauses = []
    for clause in self.clauses:
      if clause.fulfilled:
        newclauses = []
        break
      else:
        if len(clause.variables) == 0:
          newclauses = [clause]
          break
        else:
          newclauses += [clause]
    self.clauses = newclauses

  def tautology(self):
    return len(self.clauses) == 0
  
  def unfulfillable(self):
    return any(not clause.fulfilled and len(clause.variables) == 0 for clause in self.clauses)

  def __not__(self):
    if self.tautology():
      return CNFCondition(False)
    
    # this is the actually painful step (as it's also pretty inefficient right now)
    result = CNFCondition(True)
    for clause in self.clauses:
      clauseInv = CNFCondition(True)
      negVar = clause.negateVariables()
      clauseInv.clauses = [CNFClause({var: negVar[var]}) for var in negVar]

      result = result | clauseInv
    return result

  def __rand__(self, other):
    if not isinstance(other, CNFCondition):
      other = CNFCondition(other)
    
    clauses = self.clauses + other.clauses

    condition = CNFCondition(True)
    condition.clauses = clauses
    condition._prune()

    return condition

  def __ror__(self, other):
    if not isinstance(other, CNFCondition):
      other = CNFCondition(other)
    
    clauses = [clause.unite(oclause) for clause in self.clauses for oclause in other.clauses]
    condition = CNFCondition(True)
    condition.clauses = clauses
    condition._prune()

    return condition
  
  def __repr__(self):
    return f'{self.clauses}'
  
  def ccode(self):
    return f'({" && ".join(clause.ccode() for clause in self.clauses)})'

class LiveSet:
  def __init__(self, data: dict):
    self.data = data
  
  def __sub__(self, other):
    if isinstance(other, dict):
      other = LiveSet(other)
    
    result = {k:self.data[k] for k in self.data}

    for var in other.data:
      if var in result:
        result[var] &= ~other.data[var]
        if not result[var]:
          result.remove(var)

    return LiveSet(result)
  
  def __or__(self, other):
    if isinstance(other, dict):
      other = LiveSet(other)
    
    result = {k:self.data[k] for k in self.data}

    for var in other.data:
      if var in result:
        result[var] |= other.data[var]

    return LiveSet(result)
  
  def __contains__(self, element):
    if isinstance(element, tuple):
      return element[0] in self.data and (~self.data[element[0]] & element[1]).unfulfillable()
    else:
      return element in self.data
  
  def variables(self):
    return set(k for k in self.data)

class FusedProgramPoint(ProgramPoint):
  def __init__(self, action: FusedActions):
    super().__init__(action.gen_program_action())
