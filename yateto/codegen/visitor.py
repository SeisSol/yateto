import collections
import operator
from functools import reduce
from io import StringIO
from ..memory import DenseMemoryLayout
from ..controlflow.visitor import ScalarsSet, SortedGlobalsList, SortedPrefetchList
from ..controlflow.transformer import DetermineLocalInitialization
from .code import Cpp
from .factory import *

SUPPORT_LIBRARY_NAMESPACE = 'yateto'
CONSTEXPR = 'constexpr'
STATIC = 'static'
INLINE = 'inline'
MODIFIERS = '{} {}'.format(CONSTEXPR, STATIC)
STATIC_INLINE = '{} {}'.format(STATIC, INLINE)

def groupSizeToStride(groupSize):
  if len(groupSize) == 0:
    return tuple()
  stride = [1]
  for i in range(len(groupSize)-1):
    stride.append(stride[i] * groupSize[i])
  return tuple(stride)

def address(group, stride):
  return sum(map(operator.mul, group, stride))

def ndargs(d):
  return ['i' + str(i) for i in range(d)]

def typedNdArgs(d, uintTypename):
  typedArgs = ['{} {}'.format(uintTypename, arg) for arg in ndargs(d)]
  return ', '.join(typedArgs)

def indexFun(stride):
  if len(stride) == 0:
    return '0'
  args = ndargs(len(stride))
  return ' + '.join(['{}*{}'.format(stride, arg) for stride,arg in zip(stride,args)])

class KernelGenerator(object):
  PREFETCHSTRUCT_NAME = 'Prefetch'
  PREFETCHVAR_NAME = '_prefetch'

  def __init__(self, arch):
    self._arch = arch

  def _name(self, var):
    raise NotImplementedError

  def _memoryLayout(self, term):
    raise NotImplementedError

  def _sizeFun(self, term):
    return self._memoryLayout(term).requiredReals()
  
  def generate(self, cpp, cfg, factory, routineCache=None):
    hwFlops = 0
    cfg = DetermineLocalInitialization().visit(cfg, self._sizeFun)
    localNodes = dict()
    for pp in cfg:
      for name, size in pp.initLocal.items():
        cpp('{} {}[{}] __attribute__((aligned({})));'.format(self._arch.typename, name, size, self._arch.alignment))
      action = pp.action
      if action:
        if action.isRHSExpression() or action.term.isGlobal():
          termNode = action.term.node
        else:
          termNode = localNodes[action.term]
        if action.result.isLocal():
          localNodes[action.result] = termNode
          resultNode = termNode
        else:
          resultNode = action.result.node

        scalar = 1.0 if action.scalar is None else action.scalar
        if action.isRHSExpression():
          prefetchName = '{}.{}'.format(self.PREFETCHVAR_NAME, action.term.node.prefetch.name()) if action.term.node.prefetch is not None else None
          hwFlops += factory.create(action.term.node, resultNode, self._name(action.result), [self._name(var) for var in action.term.variableList()], action.add, scalar, prefetchName, routineCache)
        else:
          hwFlops += factory.simple(self._name(action.result), resultNode, self._name(action.term), termNode, action.add, scalar, routineCache)
    return hwFlops

class OptimisedKernelGenerator(KernelGenerator):
  NAMESPACE = 'kernel'
  EXECUTE_NAME = 'execute'
  FIND_EXECUTE_NAME = 'findExecute'
  EXECUTE_ARRAY_NAME = 'ExecutePtrs'
  NONZEROFLOPS_NAME = 'NonZeroFlops'
  HARDWAREFLOPS_NAME = 'HardwareFlops'
  MEMBER_FUNCTION_PTR_NAME = 'member_function_ptr'
  
  def __init__(self, arch, routineCache):
    super().__init__(arch)
    self._routineCache = routineCache

  def _name(self, var):
    return str(var)

  def _memoryLayout(self, term):
    return term.memoryLayout()
  
  class KernelOutline(object):
    def __init__(self, nonZeroFlops, hwFlops, tensors, writable, prefetch, scalars, function):
      self.nonZeroFlops = nonZeroFlops
      self.hwFlops = hwFlops
      self.tensors = tensors
      self.writable = writable
      self.prefetch = prefetch
      self.scalars = scalars
      self.function = function

    @classmethod
    def _addTensor(cls, tensor, tensors):
      bn = tensor.baseName()
      g = tensor.group()
      if bn in tensors:
        p = next(iter(tensors[bn]))
        if len(p) != len(g):
          raise ValueError('Group size mismatch ({} vs {}) for {}.'.format(p, g, bn))
        tensors[bn] = tensors[bn] | {g}
      else:
        tensors[bn] = {g}
  
  def generateKernelOutline(self, nonZeroFlops, cfg):
    scalars = ScalarsSet().visit(cfg)
    variables = SortedGlobalsList().visit(cfg)
    tensors = collections.OrderedDict()
    writable = dict()
    for var in variables:
      self.KernelOutline._addTensor(var.node.tensor, tensors)
      bn = var.node.tensor.baseName()
      if bn in writable:
        if var.writable:
          writable[bn] = True
      else:
        writable[bn] = var.writable

    prefetchTensors = SortedPrefetchList().visit(cfg)
    prefetch = collections.OrderedDict()
    for tensor in prefetchTensors:
      self.KernelOutline._addTensor(tensor, prefetch)

    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as fcpp:
      factory = OptimisedKernelFactory(fcpp, self._arch)
      hwFlops = super().generate(fcpp, cfg, factory, self._routineCache)
      function = functionIO.getvalue()    
    return self.KernelOutline(nonZeroFlops, hwFlops, tensors, writable, prefetch, scalars, function)

  @classmethod
  def _addFromKO(cls, koEntries, entries):
    for key, value in koEntries.items():
      if key not in entries:
        entries[key] = value
      else:
        entries[key] = entries[key] | value
    

  def generate(self, cpp, header, name, kernelOutlines, familyStride=None):
    tensors = collections.OrderedDict()
    prefetch = collections.OrderedDict()
    writable = dict()
    scalars = set()
    for ko in kernelOutlines:
      if ko:
        scalars = scalars | ko.scalars
        self._addFromKO(ko.tensors, tensors)
        self._addFromKO(ko.writable, writable)
        self._addFromKO(ko.prefetch, prefetch)

    scalars = sorted(list(scalars), key=str)

    if familyStride is not None:
      executeName = lambda index: self.EXECUTE_NAME + str(index)
      formatArray = lambda lst: '{{{}}}'.format(', '.join([str(l) for l in lst]))
      brackets = '[]'
    else:
      executeName = lambda index: self.EXECUTE_NAME
      formatArray = lambda lst: lst[0]
      brackets = ''

    with header.Namespace(self.NAMESPACE):
      with header.Struct(name):
        header('{} {} const {}{} = {};'.format(
          MODIFIERS,
          self._arch.uintTypename,
          self.NONZEROFLOPS_NAME,
          brackets,
          formatArray([kernelOutline.nonZeroFlops if kernelOutline else 0 for kernelOutline in kernelOutlines])
        ))
        header('{} {} const {}{} = {};'.format(
          MODIFIERS,
          self._arch.uintTypename,
          self.HARDWAREFLOPS_NAME,
          brackets,
          formatArray([kernelOutline.hwFlops if kernelOutline else 0 for kernelOutline in kernelOutlines])
        ))
        header.emptyline()
        
        for scalar in scalars:
          header('{0} {1} = std::numeric_limits<{0}>::signaling_NaN();'.format(self._arch.typename, scalar))
        
        def kernelArgs(baseName, groups, writable):
          typ = self._arch.typename
          if not writable:
            typ += ' const'
          if len(next(iter(groups))) > 0:
            header('{0}::{1}::{2}<{3}*> {1};'.format(
              InitializerGenerator.TENSOR_NAMESPACE,
              baseName,
              InitializerGenerator.CONTAINER_CLASS_NAME,
              typ
            ))
          else:
            header('{}* {}{{}};'.format(typ, baseName))
        
        for baseName, groups in tensors.items():
          kernelArgs(baseName, groups, writable[baseName])
        header.emptyline()

        if len(prefetch) > 0:
          with header.Struct(self.PREFETCHSTRUCT_NAME):
            for baseName, groups in prefetch.items():
              kernelArgs(baseName, groups, False)
          header('{} {};'.format(self.PREFETCHSTRUCT_NAME, self.PREFETCHVAR_NAME))
          header.emptyline()

        for index, kernelOutline in enumerate(kernelOutlines):
          if kernelOutline:
            header.functionDeclaration(executeName(index))

        if familyStride is not None:
          header('typedef void ({}::* const {})(void);'.format(name, self.MEMBER_FUNCTION_PTR_NAME))
          header('{} {} {}[] = {};'.format(
            MODIFIERS,
            self.MEMBER_FUNCTION_PTR_NAME,
            self.EXECUTE_ARRAY_NAME,
            formatArray(['&{}::{}'.format(name, executeName(index)) if kernelOutline else 'nullptr' for index, kernelOutline in enumerate(kernelOutlines)])
          ))
          args = typedNdArgs(len(familyStride), self._arch.uintTypename)
          indexF = indexFun(familyStride)
          with header.Function(self.FIND_EXECUTE_NAME, args, '{} {}'.format(MODIFIERS, self.MEMBER_FUNCTION_PTR_NAME)):
            header('return {}[{}];'.format(self.EXECUTE_ARRAY_NAME, indexF))
          with header.Function(self.EXECUTE_NAME, args, '{} void'.format(INLINE)):
            header('(this->*{}({}))();'.format(self.FIND_EXECUTE_NAME, ', '.join(ndargs(len(familyStride)))))

          flopFuns = [self.NONZEROFLOPS_NAME, self.HARDWAREFLOPS_NAME]
          for flopFun in flopFuns:
            funName = flopFun[:1].lower() + flopFun[1:]
            with header.Function(funName, args, '{} {}'.format(MODIFIERS, self._arch.uintTypename)):
              header('return {}[{}];'.format(flopFun, indexF))

    flopCounters = [self.NONZEROFLOPS_NAME, self.HARDWAREFLOPS_NAME]
    for fc in flopCounters:
      cpp('{} {} const {}::{}::{}{};'.format(
        CONSTEXPR,
        self._arch.uintTypename,
        self.NAMESPACE,
        name,
        fc,
        brackets
      ))
    if familyStride is not None:
      cpp('{0} {1}::{2}::{3} {1}::{2}::{4}[];'.format(
        CONSTEXPR,
        self.NAMESPACE,
        name,
        self.MEMBER_FUNCTION_PTR_NAME,
        self.EXECUTE_ARRAY_NAME
      ))
    for index, kernelOutline in enumerate(kernelOutlines):
      if kernelOutline is None:
        continue

      with cpp.Function('{}::{}::{}'.format(self.NAMESPACE, name, executeName(index))):
        sclrs = sorted(list(kernelOutline.scalars), key=str)
        for scalar in sclrs:
          cpp('assert(!std::isnan({}));'.format(scalar))
        for baseName, groups in kernelOutline.tensors.items():
          if len(next(iter(groups))) > 0:
            for gis in groups:
              cpp('assert({}({}) != nullptr);'.format(baseName, ','.join(str(gi) for gi in gis)))
          else:
            cpp('assert({} != nullptr);'.format(baseName))
        cpp(kernelOutline.function)

class UnitTestGenerator(KernelGenerator):
  KERNEL_VAR = 'krnl'
  CXXTEST_PREFIX = 'test'
  
  def __init__(self, arch):
    super().__init__(arch)
  
  def _tensorName(self, var):
    if var.isLocal():
      return str(var)
    baseName = var.node.tensor.baseName()
    group = var.node.tensor.group()
    terms = [baseName] + [str(g) for g in group]
    return '_'.join(terms)

  def _name(self, var):
    if var.isLocal():
      return str(var)
    return '_ut_' + self._tensorName(var)

  def _viewName(self, var):
    return '_view_' + self._name(var)
  
  def _groupStr(self, var):
    group = var.node.tensor.group()
    return ','.join([str(g) for g in group])

  def _groupTemplate(self, var):
    gstr = self._groupStr(var)
    return '<{}>'.format(gstr) if gstr else ''

  def _groupIndex(self, var):
    gstr = self._groupStr(var)
    return '({})'.format(gstr) if gstr else ''

  def _memoryLayout(self, term):
    return DenseMemoryLayout(term.shape())

  def _sizeFun(self, term):
    return self._memoryLayout(term).requiredReals()
  
  def generate(self, cpp, testName, kernelClass, cfg, index=None):
    scalars = ScalarsSet().visit(cfg)
    scalars = sorted(scalars, key=str)
    variables = SortedGlobalsList().visit(cfg)
    with cpp.Function(self.CXXTEST_PREFIX + testName):
      for i,scalar in enumerate(scalars):
        cpp('{} {} = {};'.format(self._arch.typename, scalar, float(i+2)))
        
      for var in variables:
        self._factory = UnitTestFactory(cpp, self._arch)
        self._factory.tensor(var.node.tensor, self._tensorName(var))
        
        cpp('{} {}[{}] __attribute__((aligned({}))) = {{}};'.format(self._arch.typename, self._name(var), self._sizeFun(var.node), self._arch.alignment))
        
        shape = var.node.shape()
        cpp('{supportNS}::DenseTensorView<{dim},{arch.typename},{arch.uintTypename}> {viewName}({utName}, {{{shape}}}, {{{start}}}, {{{shape}}});'.format(
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            dim=len(shape),
            arch = self._arch,
            utName=self._name(var),
            viewName=self._viewName(var),
            shape=', '.join([str(s) for s in shape]),
            start=', '.join(['0']*len(shape))
          )
        )
        cpp( '{initNS}::{baseName}::{viewStruct}{groupTemplate}::{createFun}({name}).copyToView({viewName});'.format(
            initNS = InitializerGenerator.INIT_NAMESPACE,
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            groupTemplate=self._groupTemplate(var),
            baseName=var.node.tensor.baseName(),
            name=self._tensorName(var),
            viewName=self._viewName(var),
            viewStruct=InitializerGenerator.VIEW_STRUCT_NAME,
            createFun=InitializerGenerator.VIEW_FUN_NAME
          )
        )
        cpp.emptyline()

      cpp( '{}::{} {};'.format(OptimisedKernelGenerator.NAMESPACE, kernelClass, self.KERNEL_VAR) )
      for scalar in scalars:
        cpp( '{0}.{1} = {1};'.format(self.KERNEL_VAR, scalar) )
      for var in variables:
        cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.node.tensor.baseName(), self._groupIndex(var), self._tensorName(var)) )
      cpp( '{}.{}();'.format(self.KERNEL_VAR, OptimisedKernelGenerator.EXECUTE_NAME + (str(index) if index is not None else '')) )
      cpp.emptyline()

      factory = UnitTestFactory(cpp, self._arch)
      super().generate(cpp, cfg, factory)

      for var in variables:
        if var.writable:
          factory.compare(self._name(var), self._memoryLayout(var.node), self._tensorName(var), var.node.memoryLayout())

class InitializerGenerator(object):
  SHAPE_NAME = 'Shape'
  SIZE_NAME = 'Size'
  SIZE_FUN_NAME = 'size'
  INDEX_FUN_NAME = 'index'
  VALUES_BASENAME = 'Values'
  CONTAINER_CLASS_NAME = 'Container'
  CONTAINER_DATA_NAME = 'data'
  TENSOR_NAMESPACE = 'tensor'
  INIT_NAMESPACE = 'init'
  VIEW_STRUCT_NAME = 'view'
  VIEW_FUN_NAME = 'create'
  VIEW_TYPE_NAME = 'type'
  
  class TensorView(object):
    ARGUMENT_NAME = 'values'

    def typename(self, dim, arch):
      return '::{}::{}<{},{},{}>'.format(SUPPORT_LIBRARY_NAMESPACE, type(self).__name__, dim, arch.typename, arch.uintTypename)
    
    @classmethod
    def arguments(cls, arch):
      return '{}* {}'.format(arch.typename, cls.ARGUMENT_NAME)
    
    def generate(cpp, group, memLayout):
      raise NotImplementedError
    
    def listToInitializerList(self, lst):
      return '{{{}}}'.format(', '.join([str(l) for l in lst]))
    
    def formatArray(self, numberType, name, values, declarationOnly):
      lhs = '{} {}[]'.format(numberType, name)
      if declarationOnly:
        return '{} {};'.format(CONSTEXPR, lhs)
      return '{} {} = {};'.format(MODIFIERS, lhs, self.listToInitializerList(values))
  
  class DenseTensorView(TensorView):
    START_NAME = 'Start'
    STOP_NAME = 'Stop'

    def generate(self, cpp, memLayout, arch, index):
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch),
          self.ARGUMENT_NAME,
          self.listToInitializerList(memLayout.shape()),
          self.listToInitializerList([r.start for r in memLayout.bbox()]),
          self.listToInitializerList([r.stop for r in memLayout.bbox()])
        )
      )
    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      cpp(self.formatArray(numberType, namespace + self.START_NAME + index, [r.start for r in memLayout.bbox()], declarationOnly))
      cpp(self.formatArray(numberType, namespace + self.STOP_NAME + index, [r.stop for r in memLayout.bbox()], declarationOnly))

  class CSCMatrixView(TensorView):
    ROWIND_NAME = 'RowInd'
    COLPTR_NAME = 'ColPtr'
    
    def typename(self, dim, arch):
      return '::{}::{}<{},{}>'.format(SUPPORT_LIBRARY_NAMESPACE, type(self).__name__, arch.typename, arch.uintTypename)

    def generate(self, cpp, memLayout, arch, index):
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch),
          self.ARGUMENT_NAME,
          self.listToInitializerList(memLayout.shape()),
          self.ROWIND_NAME + index,
          self.COLPTR_NAME + index
        )
      )
    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      cpp(self.formatArray(numberType, namespace + self.ROWIND_NAME + index, memLayout.rowIndex(), declarationOnly))
      cpp(self.formatArray(numberType, namespace + self.COLPTR_NAME + index, memLayout.colPointer(), declarationOnly))

  def __init__(self, arch, tensors):
    self._arch = arch
    self._numberType = '{} const'.format(self._arch.uintTypename)
    self._realType = '{} const'.format(self._arch.typename)
    self._realPtrType = self._realType + '*'
    self._collect = collections.OrderedDict()
    for tensor in tensors:
      baseName = tensor.baseName()
      group = tensor.group()
      if baseName not in self._collect:
        self._collect[baseName] = {group: tensor}
      elif group not in self._collect[baseName]:
        groupRef = next(iter(self._collect[baseName].keys()))
        if len(group) != len(groupRef):
          raise ValueError('Mixed group dimensions are not allowed. ({} and {} for {}.)'.format(group, groupRef, baseName))
        self._collect[baseName][group] = tensor
      else:
        assert self._collect[baseName][group] == tensor
    maxIndex = {baseName: tuple(map(max, *groups.keys())) for baseName, groups in self._collect.items()}
    self._groupSize = {baseName: tuple(map(lambda x: x+1, mi)) for baseName, mi in maxIndex.items()}
  
  def _tensorViewGenerator(self, memoryLayout):
    memLayoutMap = {
      'DenseMemoryLayout': self.DenseTensorView,
      'CSCMemoryLayout': self.CSCMatrixView
    }
    return memLayoutMap[type(memoryLayout).__name__]()
  
  def generateTensorsH(self, header):
    with header.Namespace(self.TENSOR_NAMESPACE):
      for baseName,tensors in self._collect.items():        
        with header.Struct(baseName):
          groupSize = self._groupSize[baseName]
          self._tensor(header, '', tensors, groupSize, False)
          args = ndargs(len(groupSize))
          typedArgs = typedNdArgs(len(groupSize), self._arch.uintTypename)
          returnType = '{} {}'.format(MODIFIERS, self._arch.uintTypename)
          if len(groupSize) > 0:
            with header.Function(self.INDEX_FUN_NAME, typedArgs, returnType):
              header('return {};'.format(indexFun(groupSizeToStride(groupSize))))
          with header.Function(self.SIZE_FUN_NAME, typedArgs, returnType):
            if len(groupSize) == 0:
              header('return {};'.format(self.SIZE_NAME))
            else:
              header('return {}[{}({})];'.format(self.SIZE_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
          if len(groupSize) > 0:
            header('template<typename T>')
            with header.Struct(self.CONTAINER_CLASS_NAME):
              header('T {}[{}];'.format(self.CONTAINER_DATA_NAME, reduce(operator.mul, groupSize)))
              header('{}() : {}{{}} {{}}'.format(self.CONTAINER_CLASS_NAME, self.CONTAINER_DATA_NAME))
              with header.Function('operator()', typedArgs, '{} T&'.format(INLINE)):
                header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
  
  def generateTensorsCpp(self, cpp):
    for baseName,tensors in self._collect.items():
      self._tensor(cpp, '::'.join([self.TENSOR_NAMESPACE, baseName, '']), tensors, self._groupSize[baseName], True)
  
  def generateInitH(self, header):
    with header.Namespace(self.INIT_NAMESPACE):
      for baseName,tensors in self._collect.items():
        self._init(header, baseName, '', tensors, False)

  def generateInitCpp(self, header):
    for baseName,tensors in self._collect.items():
      self._init(header, baseName, '::'.join([self.INIT_NAMESPACE, baseName, '']), tensors, True)
  
  def _tensor(self, cpp, name, tensors, groupSize, declarationOnly):
    shape = {group: tensor.shape() for group,tensor in tensors.items()}
    size = {group: [tensor.memoryLayout().requiredReals()] for group,tensor in tensors.items()}
    self._array(cpp, self._numberType, name + self.SHAPE_NAME, shape, groupSize, declarationOnly)
    self._array(cpp, self._numberType, name + self.SIZE_NAME, size, groupSize, declarationOnly, alwaysArray=False)

  def _init(self, cpp, baseName, name, tensors, declarationOnly):
    groupSize = self._groupSize[baseName]
    stride = groupSizeToStride(groupSize)
    index = lambda group: str(address(group, stride)) if len(group) > 0 else ''

    if declarationOnly:
      for group,tensor in tensors.items():
        ml = tensor.memoryLayout()
        tv = self._tensorViewGenerator(ml)
        tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, True)
      nValueArrays = 0
      for group,tensor in tensors.items():
        values = tensor.values()
        memLayout = tensor.memoryLayout()
        if values is not None:
          memory = ['0.']*memLayout.requiredReals()
          for idx,x in values.items():
            memory[memLayout.address(idx)] = x
          valuesName = '{}{}{}'.format(name, self.VALUES_BASENAME, index(group))
          cpp('{} {}[] = {{{}}};'.format(self._realType, valuesName, ', '.join(memory)))
          nValueArrays += 1
      if nValueArrays > 1:
        cpp('{} {} {}{}[];'.format(CONSTEXPR, self._realPtrType, name, self.VALUES_BASENAME))
    else:
      with cpp.Struct('{0} : {1}::{0}'.format(baseName, self.TENSOR_NAMESPACE)):
        for group,tensor in tensors.items():
          ml = tensor.memoryLayout()
          tv = self._tensorViewGenerator(ml)
          tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, False)

        valueNames = dict()
        for group,tensor in tensors.items():
          values = tensor.values()
          if values is not None:
            name = '{}{}'.format(self.VALUES_BASENAME, index(group))
            valueNames[group] = ['&{}[0]'.format(name)]
            cpp('{} {} {}[];'.format(STATIC, self._realType, name))
        if len(valueNames) > 1:
          self._array(cpp, self._realPtrType, self.VALUES_BASENAME, valueNames, groupSize, alwaysArray=False)

        cpp.emptyline()
        viewArgs = self.TensorView.arguments(self._arch)
        if len(groupSize) == 0:
          ml = next(iter(tensors.values())).memoryLayout()
          tv = self._tensorViewGenerator(ml)
          with cpp.Struct(self.VIEW_STRUCT_NAME):
            cpp('typedef {} {};'.format(tv.typename(len(ml.shape()), self._arch), self.VIEW_TYPE_NAME))
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, None)
        else:
          typedArgs = typedNdArgs(len(groupSize), self._arch.uintTypename)
          cpp('template<{}> struct {} {{}};'.format(typedArgs, self.VIEW_STRUCT_NAME))

      if len(groupSize) > 0:
        for group,tensor in tensors.items():
          ml = tensor.memoryLayout()
          tv = self._tensorViewGenerator(ml)
          typename = tv.typename(len(ml.shape()), self._arch)
          special = ','.join(str(g) for g in group)
          cpp('template<>')
          with cpp.Struct('{}::{}<{}>'.format(baseName, self.VIEW_STRUCT_NAME, special)):
            cpp('typedef {} {};'.format(typename, self.VIEW_TYPE_NAME))
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, index(group))
  
  def _array(self, cpp, typ, name, content, groupSize, declarationOnly=False, alwaysArray=True):
    maxLen = max(map(len, content.values())) if len(content.values()) > 0 else 0

    isGroup = len(groupSize) > 0
    groupIndices = '[]' if isGroup else ''

    isArray = alwaysArray or maxLen > 1
    arrayIndices = '[{}]'.format(maxLen) if isArray else ''
    
    if declarationOnly:
      cpp('{} {} {}{}{};'.format(CONSTEXPR, typ, name, groupIndices, arrayIndices))
    else:
      formatArray = lambda L: ', '.join([str(x) for x in L])
      if isGroup:
        stride = groupSizeToStride(groupSize)
        size = reduce(operator.mul, groupSize, 1)
        init = [0]*size
        for key, value in content.items():
          idx = address(key, stride)
          init[idx] = formatArray(value)
      else:
        init = [formatArray(next(iter(content.values())))]

      if isArray:
        init = ['{{{}}}'.format(i) for i in init]
      
      initStr = ', '.join(init)
      if isGroup:
        initStr = '{{{}}}'.format(initStr)
      
      cpp('{} static {} {}{}{} = {};'.format(CONSTEXPR, typ, name, groupIndices, arrayIndices, initStr))
