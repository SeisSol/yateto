import copy
import sys
from io import StringIO
from ..memory import DenseMemoryLayout
from ..ast.node import Add, IndexedTensor
from ..ast.visitor import Visitor, ComputeOptimalFlopCount
from .code import Cpp
from .factory import KernelFactory

DEFAULT_NAMESPACE = 'yateto'
MODIFIERS = 'static constexpr'

class KernelGenerator(Visitor):
  ARGUMENT_NAME = 'p'
  TEMPORARY_RESULT = '_tmp'
  NAMESPACE = 'kernel'
  EXECUTE_NAME = 'execute'
  NONZEROFLOPS_NAME = 'NonZeroFlops'
  HARDWAREFLOPS_NAME = 'HardwareFlops'
  
  class Buffer(object):
    def __init__(self, name, node):
      self.name = name
      self.node = node
  
  def __init__(self, cpp, arch, routineCache, namespace=DEFAULT_NAMESPACE):
    self._cpp = cpp
    self._arch = arch
    self._routineCache = routineCache
    self._tmp = dict()
    self._freeTmp = list()
    self._tensors = dict()
    self._factory = None
    self._namespace = namespace
    self._flops = 0
  
  def generate(self, name, node):
    self._cpp.includeSys('cassert')
    with self._cpp.Namespace(self._namespace):
      with self._cpp.Namespace(self.NAMESPACE):
        cpp = self._cpp
        functionIO = StringIO()
        function = ''
        with Cpp(functionIO) as self._cpp:
          self._factory = KernelFactory(self._cpp, self._arch)
          self.visit(node)
          function = functionIO.getvalue()
        with cpp.Struct(name):
          nonZeroFlops = ComputeOptimalFlopCount().visit(node)
          cpp('{} {} {} = {};'.format(MODIFIERS, self._arch.uintTypename, self.NONZEROFLOPS_NAME, nonZeroFlops))
          cpp('{} {} {} = {};'.format(MODIFIERS, self._arch.uintTypename, self.HARDWAREFLOPS_NAME, self._flops))
          cpp.emptyline()
          for baseName, groups in self._tensors.items():
            if groups:
              size = max(groups)+1
              cpp('{}* {}[{}] = {{{}}};'.format(self._arch.typename, baseName, size, ', '.join(['nullptr']*size)))
            else:
              cpp('{}* {} = nullptr;'.format(self._arch.typename, baseName))
          cpp.emptyline()
          with cpp.Function(self.EXECUTE_NAME):
            for baseName, groups in self._tensors.items():
              if groups:
                for g in groups:
                  cpp('assert({}[{}] != nullptr);'.format(baseName, g))
              else:
                cpp('assert({} != nullptr);'.format(baseName))
            cpp(function)
    return self._flops

  def generic_visit(self, node, **kwargs):
    result = kwargs['result'] if 'result' in kwargs else None
    if not result:
      result = self._getTemporary(node)

    names = [self.visit(child) for child in node]
    add = kwargs['add'] if 'add' in kwargs else False
    self._callFactory(node, result, names, add)
    self._freeTemporary(names)
    return result.name
  
  def visit_Assign(self, node, **kwargs):
    # Identity operation, e.g. Q['ij'] <= Q['ij']
    if isinstance(node[1], IndexedTensor) and node[0].name() == node[1].name():
      return node[0].name()

    # We may use the target buffer directly, if it is not a child of the source node
    timesContained = self._nodeContainsTensor(node[1], node[0].name())
    result = self.Buffer(node[0].name(), node[0]) if timesContained == 0 or (timesContained == 1 and isinstance(node[1], Add)) else None
    names = [self.visit(child, result=result) for child in node]
    # Copy if target buffer was not used directly
    if result is None or isinstance(node[1], IndexedTensor):
      self._callFactory(node, result, names, False)
    self._freeTemporary(names)
    return node[0].name()
  
  def visit_Add(self, node, **kwargs):
    result = kwargs['result'] if 'result' in kwargs else None
    if not result:
      result = self._getTemporary(node)

    add = False
    names = list()
    for child in node:
      names.append( self.visit(child, result=result, add=add) )
      add = True
    # Optimisation for the case that a tensor appears on the LHS and once on the RHS
    if self._nodeContainsTensor(node, result.name) == 1:
      pos = -1
      for p,child in enumerate(node):
        if isinstance(child, IndexedTensor) and child.name() == result.name:
          pos = p
          break
      children = [child for child in node]
      del children[pos]
      del names[pos]
      tmpNode = copy.copy(node)
      tmpNode.setChildren(children)
      self._callFactory(tmpNode, result, names, True)
    else:
      self._callFactory(node, result, names, False)
    self._freeTemporary(names)
    return result.name
  
  def visit_IndexedTensor(self, node, **kwargs):
    bn = node.tensor.baseName()
    g = node.tensor.group()
    if bn in self._tensors:
      p = self._tensors[bn]
      if p is not None and g is not None:
        self._tensors[bn] = p | {g}
      elif not (p is None and g is None):
        raise ValueError('Grouped tensors ({}) and single tensors ({}) may not appear mixed in a kernel.'.format(node.name(), bn))        
    else:
      self._tensors[bn] = {g} if g is not None else None
    return node.name()
  
  def _addArgument(self, name):
    return '{}.{}'.format(self.ARGUMENT_NAME, name) if not self._isTemporary(name) else name
  
  def _callFactory(self, node, result, names, add):
    resultName = self._addArgument(result.name)
    names = [self._addArgument(name) for name in names]

    self._flops += self._factory.create(node, result.node, result.name, names, add, self._routineCache)
  
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
      self._cpp('{} {}[{}] __attribute__((aligned({})));'.format(self._arch.typename, name, size, self._arch.alignment))
      self._tmp[name] = size
    return self.Buffer(name, node)
  
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

class InitializerGenerator(object):
  SHAPE_NAME = 'Shape'
  START_NAME = 'Start'
  STOP_NAME = 'Stop'
  SIZE_NAME = 'Size'
  VALUES_BASENAME = 'Values'
  NAMESPACE = 'init'

  def __init__(self, cpp, arch, namespace=DEFAULT_NAMESPACE):
    self._cpp = cpp
    self._arch = arch
    self._namespace = namespace
  
  def generate(self, matrices):
    collect = dict()
    for matrix in matrices:
      baseName = matrix.baseName()
      group = matrix.group()
      if baseName not in collect:
        collect[baseName] = {group: matrix}
      elif baseName in collect and group not in collect[baseName]:
        collect[baseName][group] = matrix
      else:
        assert collect[baseName][group] == matrix
    with self._cpp.Namespace(self._namespace):
      with self._cpp.Namespace(self.NAMESPACE):
        for baseName,matrixGroup in collect.items():
          with self._cpp.Class(baseName):
            self.visit(matrixGroup)

  def visit(self, group):
    self._cpp.label('public')

    maxIndex = max(group.keys())

    numberType = '{} {}'.format(MODIFIERS, self._arch.uintTypename)
    self._array(numberType, self.SHAPE_NAME, {k: v.shape() for k,v in group.items()}, maxIndex)
    self._array(numberType, self.START_NAME, {k: [r.start for r in v.memoryLayout().bbox()] for k,v in group.items()}, maxIndex)
    self._array(numberType, self.STOP_NAME, {k: [r.stop for r in v.memoryLayout().bbox()] for k,v in group.items()}, maxIndex)
    self._array(numberType, self.SIZE_NAME, {k: [v.memoryLayout().requiredReals()] for k,v in group.items()}, maxIndex)
    
    realType = '{} {}'.format(MODIFIERS, self._arch.typename)
    realPtrType = realType + '*'
    valueNames = dict()
    if maxIndex is not None:
      for k,v in group.items():
        values = v.values()
        memLayout = v.memoryLayout()
        if values is not None:
          memory = ['0.']*memLayout.requiredReals()
          for idx,x in values.items():
            memory[memLayout.address(idx)] = x
          name = '{}{}'.format(self.VALUES_BASENAME, k if k is not None else '')
          valueNames[k] = ['&{}[0]'.format(name)]
          self._cpp('{} {} = {{{}}};'.format(realType, name, ', '.join(memory)))
      if len(valueNames) > 0:
        self._array(realPtrType, self.VALUES_BASENAME, valueNames, maxIndex)
  
  def _array(self, typ, name, group, maxIndex):
    dummy = [0]
    formatArray = lambda L: ', '.join([str(x) for x in L])
    maxLen = max(map(len, group.values())) if len(group.values()) > 0 else 0
    if maxIndex is None:
      init = [formatArray(next(iter(group.values())) if len(group.values()) > 0 else dummy)]
    else:
      groupSize = maxIndex+1
      init = [None]*groupSize
      for idx in range(groupSize):
        init[idx] = formatArray(group[idx] if idx in group else dummy)

    arrayIndices = ''
    if maxLen > 1:
      arrayIndices = '[{}]'.format(maxLen)
      init = ['{{{}}}'.format(i) for i in init]
    
    initStr = ', '.join(init)
    groupIndices = ''
    if maxIndex is not None:
      groupIndices = '[]'
      initStr = '{{{}}}'.format(initStr)

    self._cpp('{} {}{}{} = {};'.format(typ, name, groupIndices, arrayIndices, initStr))
