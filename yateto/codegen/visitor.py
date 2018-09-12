import collections
import copy
import sys
from io import StringIO
from ..memory import DenseMemoryLayout
from ..ast.node import Add, IndexedTensor
from ..ast.visitor import Visitor, ComputeOptimalFlopCount
from ..controlflow.visitor import SortedGlobalsList
from .code import Cpp
from .factory import *

SUPPORT_LIBRARY_NAMESPACE = 'yateto'
MODIFIERS = 'static constexpr'

class KernelGenerator(Visitor):
  TEMPORARY_RESULT = '_tmp'
  NAMESPACE = 'kernel'
  EXECUTE_NAME = 'execute'
  NONZEROFLOPS_NAME = 'NonZeroFlops'
  HARDWAREFLOPS_NAME = 'HardwareFlops'
  
  class Buffer(object):
    def __init__(self, name, node):
      self.name = name
      self.node = node
  
  def __init__(self, arch, routineCache):
    self._cpp = None
    self._arch = arch
    self._routineCache = routineCache
    self._tmp = collections.OrderedDict()
    self._freeTmp = list()
    self._tensors = collections.OrderedDict()
    self._factory = None
    self._flops = 0
  
  def generate(self, cpp, header, name, node):
    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as self._cpp:
      self._factory = KernelFactory(self._cpp, self._arch)
      self.visit(node)
      function = functionIO.getvalue()
    with header.Namespace(self.NAMESPACE):
      with header.Struct(name):
        nonZeroFlops = ComputeOptimalFlopCount().visit(node)
        header('{} {} {} = {};'.format(MODIFIERS, self._arch.uintTypename, self.NONZEROFLOPS_NAME, nonZeroFlops))
        header('{} {} {} = {};'.format(MODIFIERS, self._arch.uintTypename, self.HARDWAREFLOPS_NAME, self._flops))
        header.emptyline()
        for baseName, groups in self._tensors.items():
          if groups:
            size = max(groups)+1
            header('{}* {}[{}] = {{{}}};'.format(self._arch.typename, baseName, size, ', '.join(['nullptr']*size)))
          else:
            header('{}* {} = nullptr;'.format(self._arch.typename, baseName))
        header.emptyline()
        header.functionDeclaration(self.EXECUTE_NAME)

      with cpp.Function('{}::{}::{}'.format(self.NAMESPACE, name, self.EXECUTE_NAME)):
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

  def _callFactory(self, node, result, names, add):
    self._flops += self._factory.create(node, result.node, result.name, names, add, self._routineCache)
  
  def _getTemporary(self, node):
    size = node.memoryLayout().requiredReals()
    name = None
    minSize = sys.maxsize
    index = -1
    for i,n in enumerate(self._freeTmp):
      if size <= self._tmp[n] and size <= minSize:
        name = n
        minSize = size
        index = i
    if index >= 0:
      self._freeTmp.pop(index)

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

class UnitTestGenerator(Visitor):
  TEMPORARY_RESULT = '_tmp'
  KERNEL_VAR = 'krnl'
  CXXTEST_PREFIX = 'test'
  
  class Variable(object):
    def __init__(self, tensor):
      self.baseName = tensor.baseName()
      group = tensor.group()
      self.name = '_{}_{}'.format(self.baseName, group) if group is not None else self.baseName
      self.utName = '_ut_' + self.name
      self.viewName = '_view_' + self.utName
      self.tensor = tensor
    
    def groupTemplate(self):
      group = self.tensor.group()
      return '<{}>'.format(group) if group is not None else ''

    def groupIndex(self):
      group = self.tensor.group()
      return '[{}]'.format(group) if group is not None else ''
  
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch
    self._tmp = 0
    self._tensors = collections.OrderedDict()
    self._factory = None
  
  def generate(self, kernelName, cfg):
    print(SortedGlobalsList().visit(cfg))
    cpp = self._cpp
    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as self._cpp:
      self._factory = UnitTestFactory(self._cpp, self._arch)
      self.visit(node)
      function = functionIO.getvalue()
    with cpp.Function(self.CXXTEST_PREFIX + kernelName):
      for var in self._tensors.values():
        self._factory = UnitTestFactory(cpp, self._arch)
        self._factory.create(var.tensor, var.name, [])
        
        shape = var.tensor.shape()
        size = 1
        for s in var.tensor.shape():
          size *= s
        cpp('{} {}[{}] __attribute__((aligned({}))) = {{}};'.format(self._arch.typename, var.utName, size, self._arch.alignment))
        
        cpp('{supportNS}::DenseTensorView<{dim},{arch.typename},{arch.uintTypename}> {var.viewName}({var.utName}, {{{shape}}}, {{{start}}}, {{{shape}}});'.format(
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            dim=len(shape),
            arch = self._arch,
            var=var,
            shape=', '.join([str(s) for s in shape]),
            start=', '.join(['0']*len(shape))
          )
        )
        cpp( '{initNS}::{var.baseName}::view{groupTemplate}({var.name}).copyToView({var.viewName});'.format(
            initNS = InitializerGenerator.NAMESPACE,
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            groupTemplate=var.groupTemplate(),
            var=var
          )
        )
        cpp.emptyline()

      cpp( '{}::{} {};'.format(KernelGenerator.NAMESPACE, kernelName, self.KERNEL_VAR) )
      for var in self._tensors.values():
        cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.baseName, var.groupIndex(), var.name) )
      cpp( '{}.{}();'.format(self.KERNEL_VAR, KernelGenerator.EXECUTE_NAME) )
      cpp.emptyline()

      cpp(function)
    
  def generic_visit(self, node):
    names = [self.visit(child) for child in node]
    result = self._getTemporary(node)
    self._factory.create(node, result, names)
    return result
  
  def visit_Assign(self, node):
    names = [self.visit(child) for child in node]
    result = self._getTemporary(node)
    self._factory.create(node, self._tensors[ names[0] ].name, names)
    return result
  
  def visit_IndexedTensor(self, node):
    var = self.Variable(node.tensor)
    self._tensors[var.utName] = var
    return var.utName

  def _getTemporary(self, node):
    size = 1
    for s in node.indices.shape():
      size *= s
    name = '{}{}'.format(self.TEMPORARY_RESULT, self._tmp)
    self._cpp('{} {}[{}] __attribute__((aligned({}))) = {{}};'.format(self._arch.typename, name, size, self._arch.alignment))
    self._tmp += 1
    return name

class InitializerGenerator(object):
  SHAPE_NAME = 'Shape'
  START_NAME = 'Start'
  STOP_NAME = 'Stop'
  SIZE_NAME = 'Size'
  VALUES_BASENAME = 'Values'
  NAMESPACE = 'init'
  
  class TensorView(object):
    ARGUMENT_NAME = 'values'

    def typename(self, dim, arch):
      return '::{}::{}<{},{},{}>'.format(SUPPORT_LIBRARY_NAMESPACE, type(self).__name__, dim, arch.typename, arch.uintTypename)
    
    @classmethod
    def arguments(cls, arch):
      return '{}* {}'.format(arch.typename, cls.ARGUMENT_NAME)
    
    def generate(cpp, group, memLayout):
      raise NotImplementedError
  
  class DenseTensorView(TensorView):
    def generate(self, cpp, memLayout, arch, group):
      index = '[{}]'.format(group) if group is not None else ''
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch),
          self.ARGUMENT_NAME,
          InitializerGenerator.SHAPE_NAME + index,
          InitializerGenerator.START_NAME + index,
          InitializerGenerator.STOP_NAME + index
        )
      )

  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch
  
  def _tensorViewGenerator(self, memoryLayout):
    memLayoutMap = {
      'DenseMemoryLayout': self.DenseTensorView
    }
    return memLayoutMap[type(memoryLayout).__name__]()
  
  def generate(self, tensors):
    collect = dict()
    for tensor in tensors:
      baseName = tensor.baseName()
      group = tensor.group()
      if baseName not in collect:
        collect[baseName] = {group: tensor}
      elif baseName in collect and group not in collect[baseName]:
        collect[baseName][group] = tensor
      else:
        assert collect[baseName][group] == tensor
    with self._cpp.Namespace(self.NAMESPACE):
      for baseName,tensorGroup in collect.items():
        with self._cpp.Namespace(baseName):
          self.visit(tensorGroup)

  def visit(self, group):
    maxIndex = max(group.keys())

    numberType = '{} {} const'.format(MODIFIERS, self._arch.uintTypename)
    self._array(numberType, self.SHAPE_NAME, {k: v.shape() for k,v in group.items()}, maxIndex)
    self._array(numberType, self.START_NAME, {k: [r.start for r in v.memoryLayout().bbox()] for k,v in group.items()}, maxIndex)
    self._array(numberType, self.STOP_NAME, {k: [r.stop for r in v.memoryLayout().bbox()] for k,v in group.items()}, maxIndex)
    self._array(numberType, self.SIZE_NAME, {k: [v.memoryLayout().requiredReals()] for k,v in group.items()}, maxIndex)
    
    realType = '{} {} const'.format(MODIFIERS, self._arch.typename)
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
          self._cpp('{} {}[] = {{{}}};'.format(realType, name, ', '.join(memory)))
      if len(valueNames) > 0:
        self._array(realPtrType, self.VALUES_BASENAME, valueNames, maxIndex)

    viewArgs = self.TensorView.arguments(self._arch)
    if maxIndex is None:
      ml = next(iter(group.values())).memoryLayout()
      tv = self._tensorViewGenerator(ml)
      with self._cpp.Function('view', arguments=viewArgs, returnType=tv.typename(len(ml.shape()), self._arch)):
        tv.generate(self._cpp, ml, self._arch, None)
    else:
      self._cpp('template<int n> struct view_type {};')
      self._cpp('template<int n> static typename view_type<n>::type view({});'.format(viewArgs))
      for k,v in group.items():
        ml = v.memoryLayout()
        tv = self._tensorViewGenerator(ml)
        typename = tv.typename(len(ml.shape()), self._arch)
        self._cpp( 'template<> struct view_type<{}> {{ typedef {} type; }};'.format(k, typename) )
        with self._cpp.Function('view<{}>'.format(k), arguments=viewArgs, returnType='template<> {}'.format(typename)):
          tv.generate(self._cpp, ml, self._arch, k)
  
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
