import copy
import sys
from io import StringIO
from ..memory import DenseMemoryLayout
from ..ast.node import Add, IndexedTensor
from ..ast.visitor import Visitor
from .code import Cpp
from .common import TensorDescription
from .factory import KernelFactory
from . import copyscaleadd

class KernelGenerator(Visitor):
  ARGUMENT_NAME = 'p'
  TEMPORARY_RESULT = '_tmp'
  
  class Temporary(object):
    def __init__(self, number, size):
      self.name = '_tmp' + str(number)
      self.size = size
  
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch
    self._tmp = dict()
    self._freeTmp = list()
    self._tensors = dict()
    self._factory = None
  
  def generate(self, node):
    structName = 'test__params'
    cpp = self._cpp
    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as self._cpp:
      self._factory = KernelFactory(self._cpp, self._arch)
      with self._cpp.Function('test', '{}& {}'.format(structName, self.ARGUMENT_NAME)):
        self.visit(node)
      function = functionIO.getvalue()
    with cpp.Struct(structName):
      for baseName, maxGroup in self._tensors.items():
        if maxGroup:
          cpp('{}* {}[{}];'.format(self._arch.typename, baseName, maxGroup+1))
        else:
          cpp('{}* {};'.format(self._arch.typename, baseName))
    cpp(function)

  def generic_visit(self, node, **kwargs):
    resultName = kwargs['resultName'] if 'resultName' in kwargs else None
    if not resultName:
      resultName = self._getTemporary(node)

    names = [self.visit(child) for child in node]
    add = kwargs['add'] if 'add' in kwargs else False
    self._callFactory(node, resultName, names, add)
    self._freeTemporary(names)
    return resultName
  
  def visit_Assign(self, node, **kwargs):
    # Identity operation, e.g. Q['ij'] <= Q['ij']
    if isinstance(node[1], IndexedTensor) and node[0].name() == node[1].name():
      return node[0].name()

    # We may use the target buffer directly, if it is not a child of the source node
    timesContained = self._nodeContainsTensor(node[1], node[0].name())
    resultName = node[0].name() if timesContained == 0 or (timesContained == 1 and isinstance(node[1], Add)) else None
    names = [self.visit(child, resultName=resultName) for child in node]
    # Copy if target buffer was not used directly
    if resultName == None:
      description = copyscaleadd.Description(
        alpha = 1.0,
        beta = 0.0,
        result = IndexedTensorDescription.fromNode(self._addArgument(names[0]), node.leftTerm()),
        term = IndexedTensorDescription.fromNode(self._addArgument(names[1]), node.rightTerm()),
      )
      generator = copyscaleadd.generator(self._arch, description)
      generator.generate(self._cpp)
    return resultName
  
  def visit_Add(self, node, **kwargs):
    resultName = kwargs['resultName'] if 'resultName' in kwargs else None
    if not resultName:
      resultName = self._getTemporary(node)

    add = False
    names = list()
    for child in node:
      names.append( self.visit(child, resultName=resultName, add=add) )
      add = True
    # Optimisation for the case that a tensor appears on the LHS and once on the RHS
    if self._nodeContainsTensor(node, resultName) == 1:
      pos = -1
      for p,child in enumerate(node):
        if isinstance(child, IndexedTensor) and child.name() == resultName:
          pos = p
          break
      children = [child for child in node]
      del children[pos]
      del names[pos]
      tmpNode = copy.copy(node)
      tmpNode.setChildren(children)
      self._callFactory(tmpNode, resultName, names, True)
    else:
      self._callFactory(node, resultName, names, False)
    self._freeTemporary(names)
    return resultName
  
  def visit_IndexedTensor(self, node, **kwargs):
    bn = node.tensor.baseName()
    g = node.tensor.group()
    if bn in self._tensors:
      p = self._tensors[bn]
      if p is not None and g is not None:
        self._tensors[bn] = max(p, g)
      elif not (p is None and g is None):
        raise ValueError('Grouped tensors ({}) and single tensors ({}) may not appear mixed in a kernel.'.format(node.name(), bn))        
    else:
      self._tensors[bn] = g
    return node.name()
  
  def _addArgument(self, name):
    return '{}.{}'.format(self.ARGUMENT_NAME, name) if not self._isTemporary(name) else name
  
  def _callFactory(self, node, resultName, names, add):
    resultName = self._addArgument(resultName)
    names = [self._addArgument(name) for name in names]
    self._factory.create(node, resultName, names, add)
  
  def _getTemporary(self, node):
    size = node.memoryLayout().requiredReals()
    name = None
    minSize = sys.maxsize
    for n in self._freeTmp:
      if size <= self._tmp[n] and size <= minSize:
        name = n
        minSize = size
    
    if not name:
      name = '{}{}'.format(self.TEMPORARY_RESULT, len(self._tmp))
      self._cpp('{} {}[{}];'.format(self._arch.typename, name, size))
      self._tmp[name] = size
    return name
  
  def _isTemporary(self, name):
    return name.startswith(self.TEMPORARY_RESULT)
  
  def _freeTemporary(self, names):
    for name in names:
      if self._isTemporary(name) and name not in self._freeTmp:
        self._freeTmp.append(name)
  
  def _nodeContainsTensor(self, node, name):
    times = 0
    for child in node:
      if isinstance(child, IndexedTensor) and child.name() == name:
        times += 1
    return times
