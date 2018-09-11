from ..ast.visitor import Visitor
from .graph import *

class AST2ControlFlow(Visitor):
  TEMPORARY_RESULT = '_tmp'
  
  def __init__(self):
    self._tmp = 0
    self._cfg = []
    self._writable = set()
  
  def cfg(self):
    return self._cfg + [ProgramPoint(None)]
    
  def generic_visit(self, node):
    variables = [self.visit(child) for child in node]
    
    result = self._nextTemporary()
    action = ProgramAction(result, Expression(node, variables))
    self._addAction(action)
    
    return result
  
  def visit_Add(self, node):
    variables = [self.visit(child) for child in node]    
    assert len(variables) > 1

    variables.sort(key=lambda var: int(not var.writable) + int(not var.isGlobal()))

    lastTmp = variables[0]
    for var in variables[1:]:
      tmp = self._nextTemporary()
      action = ProgramAction(tmp, lastTmp, var)
      lastTmp = tmp
      self._addAction(action)
    
    return tmp
  
  def visit_Assign(self, node):
    self._writable = self._writable | {node[0].name()}
    variables = [self.visit(child) for child in node]
    
    last = self._cfg[-1]
    if variables[0] in last.action.variables():
      action = ProgramAction(variables[0], variables[1])
      self._addAction(action)
    else:
      last.action.result = variables[0]
    
    return variables[0]
  
  def visit_IndexedTensor(self, node):
    return Variable(node.name(), node.name() in self._writable)
  
  def _addAction(self, action):
    self._cfg.append(ProgramPoint(action))

  def _nextTemporary(self):
    name = '{}{}'.format(self.TEMPORARY_RESULT, self._tmp)
    self._tmp += 1
    return Variable(name, True)

class PrettyPrinter(object):
  def __init__(self, printPPState = False):
    self._printPPState = printPPState

  def visit(self, cfg):
    for pp in cfg:
      if self._printPPState and pp.living:
        print('Living: ' + str(pp.living))
      if pp.action:
        print('  {} = {}'.format(pp.action.result, ' + '.join([str(term) for term in pp.action])))
