from ..ast.visitor import Visitor
from yateto import Scalar
from .graph import *
from ..memory import DenseMemoryLayout
from ..ast.node import Permute

class AST2ControlFlow(Visitor):
  TEMPORARY_RESULT = '_tmp'
  
  def __init__(self, simpleMemoryLayout=False):
    self._tmp = 0
    self._cfg = []
    self._writable = set()
    self._simpleMemoryLayout = simpleMemoryLayout
  
  def cfg(self):
    return self._cfg + [ProgramPoint(None)]

  def _ml(self, node):
    return DenseMemoryLayout(node.shape()) if self._simpleMemoryLayout else node.memoryLayout()

  def _addPermuteIfRequired(self, indices, term, variable):
    if indices != term.indices:
      permute = Permute(term, indices)
      if not self._simpleMemoryLayout:
        permute.setEqspp( permute.computeSparsityPattern() )
        permute.computeMemoryLayout()
      result = self._nextTemporary(permute)
      action = ProgramAction(result, Expression(permute, self._ml(permute), [variable]), False)
      self._addAction(action)
      return result
    return variable

  def generic_visit(self, node):
    variables = [self.visit(child) for child in node]
    
    result = self._nextTemporary(node)
    action = ProgramAction(result, Expression(node, self._ml(node), variables), False)
    self._addAction(action)
    
    return result
  
  def visit_Add(self, node):
    variables = [self.visit(child) for child in node]
    assert len(variables) > 1

    variables.sort(key=lambda var: int(not var.writable) + int(not var.isGlobal()))

    tmp = self._nextTemporary(node)
    add = False
    for i,var in enumerate(variables):
      rhs = self._addPermuteIfRequired(node.indices, node[i], var)
      action = ProgramAction(tmp, rhs, add)
      self._addAction(action)
      add = True
    
    return tmp
  
  def visit_ScalarMultiplication(self, node):
    variable = self.visit(node.term())

    result = self._nextTemporary(node)
    action = ProgramAction(result, variable, False, node.scalar())
    self._addAction(action)
    
    return result
  
  def visit_Assign(self, node):
    self.updateWritable(node[0].name())
    variables = [self.visit(child) for child in node]

    rhs = self._addPermuteIfRequired(node.indices, node.rightTerm(), variables[1])
    action = ProgramAction(variables[0], rhs, False)
    self._addAction(action)
    
    return variables[0]
  
  def visit_IndexedTensor(self, node):
    return Variable(node.name(), node.name() in self._writable, self._ml(node), node.eqspp(), node.tensor)
  
  def _addAction(self, action):
    self._cfg.append(ProgramPoint(action))

  def _nextTemporary(self, node):
    name = '{}{}'.format(self.TEMPORARY_RESULT, self._tmp)
    self._tmp += 1
    return Variable(name, True, self._ml(node), node.eqspp())

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
        V = V | pp.action.result.variables() | pp.action.variables()
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
