from ..tensor import Tensor
from ..ast.node import Node

class Variable(object):
  def __init__(self, name, writable):
    self.name = name
    self.writable = writable
  
  def variables(self):
    return {self}
  
  def substituted(self, when, by):
    return by if self == when else self
  
  def isGlobal(self):
    return Tensor.isValidName(self.name)

  def isLocal(self):
    return not self.isGlobal()
  
  def __hash__(self):
    return hash(self.name)
  
  def __str__(self):
    return self.name
  
  def __repr__(self):
    return str(self)
  
  def __eq__(self, other):
    isEq = self.name == other.name
    assert not isEq or self.writable == other.writable
    return isEq

class Expression(object):
  def __init__(self, node, variables):
    self.node = node
    self._variables = variables
  
  def variables(self):
    return set([var for var in self._variables])
  
  def substituted(self, when, by):
    return Expression(self.node, [var.substituted(when, by) for var in self._variables])
  
  def __str__(self):
    return '{}({})'.format(type(self.node).__name__, ', '.join([str(var) for var in self._variables]))

class ProgramAction(object):
  def __init__(self, result, term, add):
    self.result = result
    self.term = term
    self.add = add

  def isRHSExpression(self):
    return isinstance(self.term, Expression)

  def isRHSVariable(self):
    return not self.isRHSExpression()
  
  def isCompound(self):
    return self.add

  def variables(self):
    V = self.term.variables()
    if self.add:
      V = V | self.result.variables()
    return V
  
  def substituted(self, when, by, result = True, term = True):
    return ProgramAction(self.result.substituted(when, by) if result else self.result, self.term.substituted(when, by) if term else self.term, self.add)

class ProgramPoint(object):
  def __init__(self, action):
    self.action = action
    self.living = None
    
