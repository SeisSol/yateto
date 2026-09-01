import collections
from ..ast.visitor import Visitor
from yateto import Scalar
from .graph import *
from ..memory import DenseMemoryLayout
from ..ast.node import Permute, Node, Broadcast

class AST2ControlFlow(Visitor):
  TEMPORARY_RESULT = '_tmp'

  def __init__(self, simpleMemoryLayout=False):
    self._tmp = 0
    self._cfg = []
    self._writable = set()
    self._simpleMemoryLayout = simpleMemoryLayout
    self._guard = [Guard.always()]
    # a condition tensor may be rewritten inside the kernel, so every write
    # starts a new version and guards refer to (variable, version)
    self._version = collections.defaultdict(int)
    # the guard a given version was produced under; reading it is only
    # meaningful where that guard held, so it is conjoined at every use
    self._definitionGuard = dict()

  def cfg(self):
    return self._cfg + [ProgramPoint(None)]

  def _ml(self, node):
    return DenseMemoryLayout(node.shape()) if self._simpleMemoryLayout else node.memoryLayout()

  def _addTransformOp(self, permute, variable):
    if not self._simpleMemoryLayout:
      permute.setEqspp( permute.computeSparsityPattern() )
      permute.computeMemoryLayout()
    permute.datatype = permute[0].datatype
    result = self._nextTemporary(permute)
    action = ProgramAction(result, Expression(permute, self._ml(permute), [variable]), False, condition=self._guard[-1])
    self._addAction(action)
    return result

  def _addPermuteIfRequired(self, indices, term, variable):
    result = variable
    if indices != term.indices:
      # always assume that we write into the _bigger_ output
      # (otherwise, there'd need to be a reduction/IndexSum first)
      assert term.indices <= indices

      order = [idx for idx in indices if idx in term.indices]
      termOrder = [idx for idx in term.indices]

      intermediate = variable
      inode = term
      if order != termOrder:
        # permute needed, run before broadcast
        inode = Permute.subPermute(term, indices)
        intermediate = self._addTransformOp(inode, variable)

      result = intermediate
      if len(term.indices) != len(indices):
        # broadcast needed, more output than input indices
        result = self._addTransformOp(Broadcast(inode, indices), intermediate)

    return result

  def generic_visit(self, node):
    variables = [self.visit(child) for child in node]

    result = self._nextTemporary(node)
    action = ProgramAction(result, Expression(node, self._ml(node), variables), False, condition=self._guard[-1])
    self._addAction(action)

    return result

  def visit_SliceView(self, node):
    var = self.visit(node.term())
    ml = node.getMemoryLayout(var.memoryLayout())
    return VariableView(var, ml, node.eqspp())

  def visit_Add(self, node):
    variables = [self.visit(child) for child in node]
    assert len(variables) >= 1

    variables.sort(key=lambda var: int(not var.writable) + int(not var.isGlobal()))

    tmp = self._nextTemporary(node)
    add = False
    for i,var in enumerate(variables):
      rhs = self._addPermuteIfRequired(node.indices, node[i], var)
      action = ProgramAction(tmp, rhs, add, condition=self._guard[-1])
      self._addAction(action)
      add = True

    return tmp

  def visit_ScalarMultiplication(self, node):
    variable = self.visit(node.term())

    result = self._nextTemporary(node)
    action = ProgramAction(result, variable, False, node.scalar(), condition=self._guard[-1])
    self._addAction(action)

    return result

  def visit_Assign(self, node):
    outerGuard = self._guard[-1]

    # The condition is evaluated to decide the branch, so it is computed
    # outside the new guard -- before it is pushed.
    if isinstance(node.condition(), Node):
      conditionVar = self.visit(node[2])
      version = self._version[conditionVar.name]
      myGuard = Guard.literal(conditionVar, version) \
                & self._definitionGuard.get((conditionVar.name, version), Guard.always())
    else:
      myGuard = Guard.coerce(node.condition())

    self.updateWritable(node[0].name())

    guard = outerGuard & myGuard

    # The whole right-hand side runs under the guard, not just the final store.
    self._guard.append(guard)
    try:
      rVar = self.visit(node[1])
      rhs = self._addPermuteIfRequired(node.indices, node.rightTerm(), rVar)

      lVar = self.visit(node[0])
      self._addAction(ProgramAction(lVar, rhs, False, condition=guard))
    finally:
      self._guard.pop()

    name = node[0].name()
    self._version[name] += 1
    self._definitionGuard[(name, self._version[name])] = guard

    return lVar

  def visit_IndexedTensor(self, node):
    return Variable(node.name(), node.name() in self._writable, self._ml(node), node.eqspp(), node.tensor, datatype=node.datatype, is_temporary=node.tensor.temporary)

  def visit_IfThenElse(self, node):
    raise NotImplementedError(
      'IfThenElse is not lowered yet; use yateto.functions.where (Elementwise(Ternary)).')

  def _addAction(self, action):
    self._cfg.append(ProgramPoint(action))

  def _nextTemporary(self, node):
    name = f'{self.TEMPORARY_RESULT}{self._tmp}'
    self._tmp += 1
    return Variable(name, True, self._ml(node), node.eqspp(), is_temporary=True, datatype=node.datatype)

  def updateWritable(self, name):
    self._writable = self._writable | {name}
    # Set variables writable that were added beforehand
    for pp in self._cfg:
      if pp.action:
        pp.action.setVariablesWritable(name)

class SortedGlobalsList(object):
  def visit(self, cfg):
    V = set()
    for pp in cfg:
      if pp.action:
        V = V | pp.action.result.variables() | pp.action.allVariables()
    return sorted([var for var in V if var.isGlobal()], key=lambda x: str(x))

class SortedPrefetchList(object):
  def visit(self, cfg):
    V = set()
    for pp in cfg:
      if pp.action and pp.action.isRHSExpression() and pp.action.term.node.prefetch is not None:
        V = V | {pp.action.term.node.prefetch}
    return sorted([v for v in V], key=lambda x: x.name())

class ScalarsSet(object):
  def visit(self, cfg):
    S = set()
    for pp in cfg:
      if pp.action:
        if isinstance(pp.action.scalar, Scalar):
          S = S | {pp.action.scalar}
    return S

class PrettyPrinter(object):
  def __init__(self, printPPState = False):
    self._printPPState = printPPState

  def visit(self, cfg):
    for pp in cfg:
      if self._printPPState:
        if pp.live:
          print('L =', pp.live)
        if pp.initBuffer:
          print('Init =', pp.initBuffer)
      if pp.action:
        actionRepr = str(pp.action.term)
        if pp.action.scalar is not None:
          actionRepr = str(pp.action.scalar) + ' * ' + actionRepr
        print( '  {} {} {}'.format(pp.action.result, '+=' if pp.action.add else '=', actionRepr) )
