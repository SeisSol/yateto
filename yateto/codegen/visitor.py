import collections
from io import StringIO
from ..memory import DenseMemoryLayout
from ..controlflow.visitor import SortedGlobalsList
from ..controlflow.transformer import DetermineLocalInitialization
from .code import Cpp
from .factory import *

SUPPORT_LIBRARY_NAMESPACE = 'yateto'
CONSTEXPR = 'constexpr'
MODIFIERS = '{} static'.format(CONSTEXPR)

class KernelGenerator(object):  
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

        if action.isRHSExpression():
          hwFlops += factory.create(action.term.node, resultNode, self._name(action.result), [self._name(var) for var in action.term.variableList()], action.add, routineCache)
        else:
          hwFlops += factory.simple(self._name(action.result), resultNode, self._name(action.term), termNode, action.add, routineCache)
    return hwFlops

class OptimisedKernelGenerator(KernelGenerator):
  NAMESPACE = 'kernel'
  EXECUTE_NAME = 'execute'
  FIND_EXECUTE_NAME = 'findExecute'
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
    def __init__(self, nonZeroFlops, hwFlops, tensors, writable, function):
      self.nonZeroFlops = nonZeroFlops
      self.hwFlops = hwFlops
      self.tensors = tensors
      self.writable = writable
      self.function = function
  
  def generateKernelOutline(self, nonZeroFlops, cfg):
    variables = SortedGlobalsList().visit(cfg)
    tensors = collections.OrderedDict()
    writable = dict()
    for var in variables:
      bn = var.node.tensor.baseName()
      g = var.node.tensor.group()
      if bn in tensors:
        p = tensors[bn]
        if p is not None and g is not None:
          tensors[bn] = p | {g}
        elif not (p is None and g is None):
          raise ValueError('Grouped tensors ({}) and single tensors ({}) may not appear mixed in a kernel.'.format(node.name(), bn))
        if var.writable:
          writable[bn] = True
      else:
        tensors[bn] = {g} if g is not None else None
        writable[bn] = var.writable

    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as fcpp:
      factory = OptimisedKernelFactory(fcpp, self._arch)
      hwFlops = super().generate(fcpp, cfg, factory, self._routineCache)
      function = functionIO.getvalue()    
    return self.KernelOutline(nonZeroFlops, hwFlops, tensors, writable, function)

  def generate(self, cpp, header, name, kernelOutlines, familyStride=None):
    tensors = dict()
    writable = dict()
    for ko in kernelOutlines:
      if ko:
        for key,groups in ko.tensors.items():
          if key not in tensors:
            tensors[key] = groups
            writable[key] = ko.writable[key]
          else:
            if tensors[key] is not None or groups is not None:
              tensors[key] = tensors[key] | groups
            if ko.writable[key]:
              writable[key] = True

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
        
        for baseName, groups in tensors.items():
          typ = self._arch.typename
          if not writable[baseName]:
            typ += ' const'
          if groups:
            size = max(groups)+1
            header('{}* {}[{}] = {{{}}};'.format(typ, baseName, size, ', '.join(['nullptr']*size)))
          else:
            header('{}* {} = nullptr;'.format(typ, baseName))
        header.emptyline()
        for index, kernelOutline in enumerate(kernelOutlines):
          if kernelOutline:
            header.functionDeclaration(executeName(index))
        
        if familyStride is not None:
          header('typedef void ({}::* const {})(void);'.format(name, self.MEMBER_FUNCTION_PTR_NAME))
          header('{} {} {}[] = {};'.format(
            MODIFIERS,
            self.MEMBER_FUNCTION_PTR_NAME,
            self.EXECUTE_NAME,
            formatArray(['&{}::{}'.format(name, executeName(index)) if kernelOutline else 'nullptr' for index, kernelOutline in enumerate(kernelOutlines)])
          ))
          args = ['i' + str(i) for i,v in enumerate(familyStride)]
          argsStr = ' + '.join(['{}*{}'.format(familyStride[i], arg) for i,arg in enumerate(args)])
          typedArgs = ['{} {}'.format(self._arch.uintTypename, arg) for arg in args]
          typedArgsStr = ', '.join(typedArgs)
          with header.Function(self.FIND_EXECUTE_NAME, typedArgsStr, '{} {}'.format(MODIFIERS, self.MEMBER_FUNCTION_PTR_NAME)):
            header('return {}[{}];'.format(self.EXECUTE_NAME, argsStr))

          flopFuns = [self.NONZEROFLOPS_NAME, self.HARDWAREFLOPS_NAME]
          for flopFun in flopFuns:
            funName = flopFun[:1].lower() + flopFun[1:]
            with header.Function(funName, typedArgsStr, '{} {}'.format(MODIFIERS, self._arch.uintTypename)):
              header('return {}[{}];'.format(flopFun, argsStr))

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
        self.EXECUTE_NAME
      ))
    for index, kernelOutline in enumerate(kernelOutlines):
      if kernelOutline is None:
        continue

      with cpp.Function('{}::{}::{}'.format(self.NAMESPACE, name, executeName(index))):
        for baseName, groups in kernelOutline.tensors.items():
          if groups:
            for g in groups:
              cpp('assert({}[{}] != nullptr);'.format(baseName, g))
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
    return '_{}_{}'.format(baseName, group) if group is not None else baseName

  def _name(self, var):
    if var.isLocal():
      return str(var)
    return '_ut_' + self._tensorName(var)

  def _viewName(self, var):
    return '_view_' + self._name(var)

  def _groupTemplate(self, var):
    group = var.node.tensor.group()
    return '<{}>'.format(group) if group is not None else ''

  def _groupIndex(self, var):
    group = var.node.tensor.group()
    return '[{}]'.format(group) if group is not None else ''

  def _memoryLayout(self, term):
    return DenseMemoryLayout(term.shape())

  def _sizeFun(self, term):
    return self._memoryLayout(term).requiredReals()
  
  def generate(self, cpp, testName, kernelClass, cfg, index=None):
    variables = SortedGlobalsList().visit(cfg)
    with cpp.Function(self.CXXTEST_PREFIX + testName):
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
        cpp( '{initNS}::{baseName}::view{groupTemplate}({name}).copyToView({viewName});'.format(
            initNS = InitializerGenerator.INIT_NAMESPACE,
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            groupTemplate=self._groupTemplate(var),
            baseName=var.node.tensor.baseName(),
            name=self._tensorName(var),
            viewName=self._viewName(var)
          )
        )
        cpp.emptyline()

      cpp( '{}::{} {};'.format(OptimisedKernelGenerator.NAMESPACE, kernelClass, self.KERNEL_VAR) )
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
  START_NAME = 'Start'
  STOP_NAME = 'Stop'
  SIZE_NAME = 'Size'
  VALUES_BASENAME = 'Values'
  TENSOR_NAMESPACE = 'tensor'
  INIT_NAMESPACE = 'init'
  
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
      elif baseName in self._collect and group not in self._collect[baseName]:
        self._collect[baseName][group] = tensor
      else:
        assert self._collect[baseName][group] == tensor
  
  def _tensorViewGenerator(self, memoryLayout):
    memLayoutMap = {
      'DenseMemoryLayout': self.DenseTensorView
    }
    return memLayoutMap[type(memoryLayout).__name__]()
  
  def generateTensorsH(self, header):
    with header.Namespace(self.TENSOR_NAMESPACE):
      for baseName,tensorGroup in self._collect.items():        
        with header.Struct(baseName):
          self._tensor(header, '', tensorGroup, False)
  
  def generateTensorsCpp(self, cpp):
    for baseName,tensorGroup in self._collect.items():
      self._tensor(cpp, '::'.join([self.TENSOR_NAMESPACE, baseName, '']), tensorGroup, True)
  
  def generateInitH(self, header):
    with header.Namespace(self.INIT_NAMESPACE):
      for baseName,tensorGroup in self._collect.items():
        self._init(header, baseName, '', tensorGroup, False)

  def generateInitCpp(self, header):
    for baseName,tensorGroup in self._collect.items():
      self._init(header, baseName, '::'.join([self.INIT_NAMESPACE, baseName, '']), tensorGroup, True)
  
  def _tensor(self, cpp, name, group, declarationOnly):
    maxIndex = max(group.keys())
    self._array(cpp, self._numberType, name + self.SHAPE_NAME, {k: v.shape() for k,v in group.items()}, maxIndex, declarationOnly)
    self._array(cpp, self._numberType, name + self.SIZE_NAME, {k: [v.memoryLayout().requiredReals()] for k,v in group.items()}, maxIndex, declarationOnly, alwaysArray=False)

  def _init(self, cpp, baseName, name, group, declarationOnly):
    maxIndex = max(group.keys())
    
    def arrays():
      self._array(cpp, self._numberType, name + self.START_NAME, {k: [r.start for r in v.memoryLayout().bbox()] for k,v in group.items()}, maxIndex, declarationOnly)
      self._array(cpp, self._numberType, name + self.STOP_NAME, {k: [r.stop for r in v.memoryLayout().bbox()] for k,v in group.items()}, maxIndex, declarationOnly)
    
    if declarationOnly:
      arrays()
      if maxIndex is not None:
        nValueArrays = 0
        for k,v in group.items():
          if v.values() is not None:
            valuesName = '{}{}{}'.format(name, self.VALUES_BASENAME, k if k is not None else '')
            cpp('{} {} {}[];'.format(CONSTEXPR, self._realType, valuesName))
            nValueArrays += 1
        if nValueArrays > 0:
          cpp('{} {} {}{}[];'.format(CONSTEXPR, self._realPtrType, name, self.VALUES_BASENAME))
    else:
      with cpp.Struct('{0} : {1}::{0}'.format(baseName, self.TENSOR_NAMESPACE)):
        arrays()
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
              cpp('{} {} {}[] = {{{}}};'.format(MODIFIERS, self._realType, name, ', '.join(memory)))
          if len(valueNames) > 0:
            self._array(cpp, self._realPtrType, self.VALUES_BASENAME, valueNames, maxIndex, alwaysArray=False)

        viewArgs = self.TensorView.arguments(self._arch)
        if maxIndex is None:
          ml = next(iter(group.values())).memoryLayout()
          tv = self._tensorViewGenerator(ml)
          with cpp.Function('view', arguments=viewArgs, returnType='static {}'.format(tv.typename(len(ml.shape()), self._arch))):
            tv.generate(cpp, ml, self._arch, None)
        else:        
          cpp('template<int n> struct view_type {};')
          cpp('template<int n> static typename view_type<n>::type view({});'.format(viewArgs))

      if maxIndex is not None:
        for k,v in group.items():
          ml = v.memoryLayout()
          tv = self._tensorViewGenerator(ml)
          typename = tv.typename(len(ml.shape()), self._arch)
          cpp( 'template<> struct {}::view_type<{}> {{ typedef {} type; }};'.format(baseName, k, typename) )
          with cpp.Function('{}::view<{}>'.format(baseName, k), arguments=viewArgs, returnType='template<> inline {}'.format(typename)):
            tv.generate(cpp, ml, self._arch, k)
  
  def _array(self, cpp, typ, name, group, maxIndex, declarationOnly=False, alwaysArray=True):
    maxLen = max(map(len, group.values())) if len(group.values()) > 0 else 0

    isGroup = maxIndex is not None
    groupIndices = ''
    if isGroup:
      groupIndices = '[]'

    isArray = alwaysArray or maxLen > 1
    arrayIndices = ''
    if isArray:
      arrayIndices = '[{}]'.format(maxLen)
    
    if declarationOnly:
      cpp('{} {} {}{}{};'.format(CONSTEXPR, typ, name, groupIndices, arrayIndices))
    else:
      dummy = [0]
      formatArray = lambda L: ', '.join([str(x) for x in L])
      if isGroup:
        groupSize = maxIndex+1
        init = [None]*groupSize
        for idx in range(groupSize):
          init[idx] = formatArray(group[idx] if idx in group else dummy)
      else:
        init = [formatArray(next(iter(group.values())) if len(group.values()) > 0 else dummy)]

      if isArray:
        init = ['{{{}}}'.format(i) for i in init]
      
      initStr = ', '.join(init)
      if isGroup:
        initStr = '{{{}}}'.format(initStr)
      
      cpp('{} static {} {}{}{} = {};'.format(CONSTEXPR, typ, name, groupIndices, arrayIndices, initStr))
