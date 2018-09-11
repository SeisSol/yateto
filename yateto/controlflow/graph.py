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
    return not self.name.startswith('_')

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
  def __init__(self, result, *terms):
    self.result = result
    self.terms = list(terms)
  
  def __iter__(self):
    return iter(self.terms)
  
  def _isExpression(self, term):
    return isinstance(term, Expression)
  
  def isPlusEquals(self):
    return len(self.terms) > 1 and self.result in self.simpleTerms()
  
  def containsExpression(self):
    return any([self._isExpression(term) for term in self.terms])
  
  def simpleTerms(self):
    return [term for term in self.terms if not self._isExpression(term)]
  
  def prependTerms(self, *terms):
    newTerms = terms
    newTerms.extend(self.terms)
    self.terms = newTerms
    
  def variables(self):
    V = set()
    for term in self.terms:
      V = V | term.variables()
    return V
    
  def localVariables(self):
    V = set()
    for term in self.terms:
      V = V | set([var for var in term.variables() if var.isLocal()])
    if self.result.isLocal():
      return V | {self.result}
    return V
  
  def substituted(self, when, by):
    return ProgramAction(self.result.substituted(when, by), *[term.substituted(when, by) for term in self.terms])    

class ProgramPoint(object):
  def __init__(self, action):
    self.action = action
    self.living = None
    
