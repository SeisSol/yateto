import collections
import operator
from functools import reduce
from io import StringIO
from ..memory import DenseMemoryLayout
from ..controlflow.visitor import ScalarsSet, SortedGlobalsList, SortedPrefetchList
from ..controlflow.transformer import DetermineLocalInitialization
from ..controlflow.graph import Variable
from ..type import Tensor
from .code import Cpp
from .factory import *
from .common import BatchedOperationsAux
from ..type import Scalar

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
  BUFFER_NAME = '_buffer'

  def __init__(self, arch):
    self._arch = arch

  @classmethod
  def _bufferName(cls, buf):
    return cls.BUFFER_NAME + str(buf)

  def deduce_single_scalar(self, scalar):
    return 1.0 if scalar is None else scalar

  def deduce_scalar_list(self, action):
    return [self.deduce_single_scalar(scalar) for scalar in action.scalar]

  def deduce_scalar(self, action):
    if isinstance(action.scalar, list):
      return self.deduce_scalar_list(action)
    else:
      return self.deduce_single_scalar(action.scalar)

  def generate(self, cpp, cfg, factory,  routineCache, gemm_cfg):
    hwFlops = 0
    # temporary memory required (per element in case of gpu)
    # NOTE: it is required to know in case if the memory is allocated on the heap
    #       an provided by the user
    required_tmp_mem = 0
    cfg = DetermineLocalInitialization().visit(cfg)
    localPtrs = set()
    for pp in cfg:
      localPtrs.update(pp.bufferMap.keys())
    if localPtrs:
      cpp( '{}{};'.format(self._arch.typename, ','.join(map(lambda x: ' *' + str(x), localPtrs))) )
    for pp in cfg:
      for buf, size in pp.initBuffer.items():
        required_tmp_mem += size * self._arch.bytesPerReal
        bufname = self._bufferName(buf)
        factory.temporary(bufname, size)
      for local, buf in pp.bufferMap.items():
        cpp('{} = {};'.format(local, self._bufferName(buf)))
      action = pp.action
      if action:
        scalar = self.deduce_scalar(action)
        if action.isRHSExpression():
          prefetchName = '{}.{}'.format(self.PREFETCHVAR_NAME, action.term.node.prefetch.name()) if action.term.node.prefetch is not None else None
          hwFlops += factory.create(action.term.node, action.result, action.term.variableList(), action.add, scalar, prefetchName, routineCache, gemm_cfg)
        else:
          hwFlops += factory.simple(action.result, action.term, action.add, scalar, routineCache)
    return hwFlops, required_tmp_mem

class OptimisedKernelGenerator(KernelGenerator):
  NAMESPACE = 'kernel'
  EXECUTE_NAME = 'execute'
  FIND_EXECUTE_NAME = 'findExecute'
  EXECUTE_ARRAY_NAME = 'ExecutePtrs'
  NONZEROFLOPS_NAME = 'NonZeroFlops'
  HARDWAREFLOPS_NAME = 'HardwareFlops'
  MEMBER_FUNCTION_PTR_NAME = 'member_function_ptr'
  TEMP_MEM_REQUIRED_NAME = 'TmpMemRequiredInBytes'
  TEMP_MAX_MEM_REQUIRED_NAME = 'TmpMaxMemRequiredInBytes'

  
  def __init__(self, arch, routineCache):
    super().__init__(arch)
    self._routineCache = routineCache
  
  class KernelOutline(object):
    def __init__(self,
                 nonZeroFlops,
                 hwFlops,
                 tensors,
                 writable,
                 prefetch,
                 scalars,
                 function,
                 tmp_mem_size,
                 is_compute_constant_tensors,
                 target):

      self.nonZeroFlops = nonZeroFlops
      self.hwFlops = hwFlops
      self.tensors = tensors
      self.writable = writable
      self.prefetch = prefetch
      self.scalars = scalars
      self.function = function
      self.tmp_mem_size = tmp_mem_size
      self.is_compute_constant_tensors = is_compute_constant_tensors
      self.target = target

    @classmethod
    def _addTensor(cls, tensor, tensors):
      base_name = tensor.baseNameWithNamespace()
      group = tensor.group()
      if base_name in tensors:
        p = next(iter(tensors[base_name]))
        if len(p) != len(group):
          raise ValueError('Group size mismatch ({} vs {}) for {}.'.format(p, group, base_name))
        tensors[base_name] = tensors[base_name] | {group}
      else:
        tensors[base_name] = {group}
  
  def generateKernelOutline(self, nonZeroFlops, cfg, gemm_cfg, target):
    scalarsP = ScalarsSet().visit(cfg)
    variables = SortedGlobalsList().visit(cfg)
    tensors = collections.OrderedDict()
    writable = dict()
    is_compute_constant_tensors = dict()
    scalars = collections.OrderedDict()
    for scalar in scalarsP:
      self.KernelOutline._addTensor(scalar, scalars)
    for var in variables:
      self.KernelOutline._addTensor(var.tensor, tensors)
      bn = var.tensor.baseNameWithNamespace()

      if bn in writable:
        if var.writable:
          writable[bn] = True
      else:
        writable[bn] = var.writable

      is_compute_constant_tensors[bn] = var.tensor.is_compute_constant()

    prefetchTensors = SortedPrefetchList().visit(cfg)
    prefetch = collections.OrderedDict()
    for tensor in prefetchTensors:
      self.KernelOutline._addTensor(tensor, prefetch)

    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as fcpp:
      factory = OptimisedKernelFactory(fcpp, self._arch, target)
      hwFlops, tmp_memory = super().generate(fcpp, cfg, factory, self._routineCache, gemm_cfg)
      factory.freeTmp()
      factory.reset_stream()
      factory.reset_flags()
      function = functionIO.getvalue()    
    return self.KernelOutline(nonZeroFlops,
                              hwFlops,
                              tensors,
                              writable,
                              prefetch,
                              scalars,
                              function,
                              tmp_memory,
                              is_compute_constant_tensors,
                              target)

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
    scalars = collections.OrderedDict()
    is_compute_constant_tensors = dict()
    for ko in kernelOutlines:
      if ko:
        self._addFromKO(ko.scalars, scalars)
        self._addFromKO(ko.tensors, tensors)
        self._addFromKO(ko.writable, writable)
        self._addFromKO(ko.prefetch, prefetch)
        self._addFromKO(ko.is_compute_constant_tensors, is_compute_constant_tensors)

    target = kernelOutlines[-1].target
    is_same_target = True
    for outline in kernelOutlines:
      if outline:
        is_same_target = True if outline.target == target else False

    if not is_same_target:
      raise RuntimeError("kernels with the same family belong to different compute target.")

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
          self._arch.ulongTypename,
          self.NONZEROFLOPS_NAME,
          brackets,
          formatArray([kernelOutline.nonZeroFlops if kernelOutline else 0 for kernelOutline in kernelOutlines])
        ))
        header('{} {} const {}{} = {};'.format(
          MODIFIERS,
          self._arch.ulongTypename,
          self.HARDWAREFLOPS_NAME,
          brackets,
          formatArray([kernelOutline.hwFlops if kernelOutline else 0 for kernelOutline in kernelOutlines])
        ))

        # tmp mem required by a kernel(s)
        tmp_mem_list = [kernelOutline.tmp_mem_size if kernelOutline else 0 for kernelOutline in kernelOutlines]
        header('{} {} const {}{} = {};'.format(MODIFIERS,
                                               self._arch.ulongTypename,
                                               self.TEMP_MEM_REQUIRED_NAME,
                                               brackets,
                                               formatArray(tmp_mem_list)))

        header('{} {} const {} = {};'.format(MODIFIERS,
                                             self._arch.ulongTypename,
                                             self.TEMP_MAX_MEM_REQUIRED_NAME,
                                             max(tmp_mem_list)))

        if target == 'gpu':
          # LinearAllocatorT controls external extra mem. allocated on gpu for tmp. variables
          header(f'yateto::LinearAllocatorT<{self._arch.typename}> linearAllocator;')

        header.emptyline()

        def kernelArgs(base_name_with_namespace, groups, writable, is_constant, target):
          prefix, base_name = Tensor.splitBasename(base_name_with_namespace)
          typ = self._arch.typename
          ptr_type = '**' if not is_constant and target == 'gpu' else '*'
          if not writable:
            typ += ' const'
          if len(next(iter(groups))) > 0:
            class_name = f'{prefix}{InitializerGenerator.TENSOR_NAMESPACE}::{base_name}'
            container_type = f'{InitializerGenerator.CONTAINER_CLASS_NAME}<{typ}{ptr_type}>'
            header(f'{class_name}::{container_type} {base_name};')
          else:
            header(f'{typ}{ptr_type} {base_name}{{}};')
        
        def scalarArgs(base_name_with_namespace, groups):
          prefix, base_name = Tensor.splitBasename(base_name_with_namespace)
          typ = self._arch.typename
          if len(next(iter(groups))) > 0:
            class_name = f'{prefix}{InitializerGenerator.TENSOR_NAMESPACE}::{base_name}'
            container_type = f'{InitializerGenerator.CONTAINER_CLASS_NAME}<{typ}>'
            header(f'{class_name}::{container_type} {base_name};')
          else:
            header(f'{typ} {base_name} = std::numeric_limits<{typ}>::signaling_NaN();')

        for baseName, groups in scalars.items():
          scalarArgs(baseName,
                     groups)
        for baseName, groups in tensors.items():
          kernelArgs(baseName,
                     groups,
                     writable[baseName],
                     is_compute_constant_tensors[baseName],
                     target)
        header.emptyline()

        # containers with extra offsets for GPU-like computations
        if target == 'gpu':
          header(f'unsigned {BatchedOperationsAux.NUM_ELEMENTS_NAME} = 0;')
          header(f'void *{BatchedOperationsAux.STREAM_PTR_NAME} = {BatchedOperationsAux.FORBIDDEN_STREAM_PTR};')
          header(f'unsigned *{BatchedOperationsAux.FLAGS_NAME} = nullptr;')

          def generate_extra_offset_args(base_name_with_namespace, groups):
            prefix, base_name = Tensor.splitBasename(base_name_with_namespace)
            offset_type = 'int'
            offset_name = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{base_name}'
            if len(next(iter(groups))) > 0:
              class_name = f'{prefix}{InitializerGenerator.TENSOR_NAMESPACE}::{base_name}'
              container_type = f'{InitializerGenerator.CONTAINER_CLASS_NAME}<{offset_type}>'
              header(f'{class_name}::{container_type} {offset_name};')
            else:
              header(f'{offset_type} {offset_name}{{}};')

          for base_name, groups in tensors.items():
            generate_extra_offset_args(base_name, groups)
        header.emptyline()

        if len(prefetch) > 0:
          with header.Struct(self.PREFETCHSTRUCT_NAME):
            for baseName, groups in prefetch.items():
              kernelArgs(baseName, groups, writable=False, is_constant=False, target='any')
          header('{} {};'.format(self.PREFETCHSTRUCT_NAME, self.PREFETCHVAR_NAME))
          header.emptyline()

        for index, kernelOutline in enumerate(kernelOutlines):
          if kernelOutline:
            header.functionDeclaration(executeName(index))

        if familyStride is not None:
          header('using {} = void ({}::*)();'.format(self.MEMBER_FUNCTION_PTR_NAME, name))
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

          aux_functions = [self.NONZEROFLOPS_NAME, self.HARDWAREFLOPS_NAME, self.TEMP_MEM_REQUIRED_NAME]
          for function in aux_functions:
            funName = function[:1].lower() + function[1:]
            with header.Function(funName, args, '{} {}'.format(MODIFIERS, self._arch.ulongTypename)):
              header('return {}[{}];'.format(function, indexF))

    flopCounters = [self.NONZEROFLOPS_NAME, self.HARDWAREFLOPS_NAME]
    for fc in flopCounters:
      cpp('{} {} const {}::{}::{}{};'.format(
        CONSTEXPR,
        self._arch.ulongTypename,
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
        for base_name_with_namespace, groups in kernelOutline.tensors.items():
          base_name = Tensor.splitBasename(base_name_with_namespace)[-1]
          if len(next(iter(groups))) > 0:
            for gis in groups:
              cpp('assert({}({}) != nullptr);'.format(base_name, ','.join(str(gi) for gi in gis)))
          else:
            cpp(f'assert({base_name} != nullptr);')

        if target == 'gpu':
          cpp(f'assert({BatchedOperationsAux.NUM_ELEMENTS_NAME} != 0);')
          cpp(f'assert({BatchedOperationsAux.STREAM_PTR_NAME} != {BatchedOperationsAux.FORBIDDEN_STREAM_PTR});')

        cpp(kernelOutline.function)

class UnitTestGenerator(KernelGenerator):
  KERNEL_VAR = 'krnl'
  
  def __init__(self, arch):
    super().__init__(arch)

  def deduce_single_scalar(self, scalar):
    if scalar is None:
      return 1.0
    elif isinstance(scalar, Scalar):
      return self._tensorNameS(scalar)
    else:
      return scalar

  @classmethod
  def _tensorName(cls, var):
    if var.isLocal():
      return str(var)
    baseName = var.tensor.baseName()
    group = var.tensor.group()
    terms = [baseName] + [str(g) for g in group]
    return '_'.join(terms)

  @classmethod
  def _nameS(cls, var):
    return '_ut_' + cls._tensorNameS(var)

  @classmethod
  def _tensorNameS(cls, var):
    baseName = var.baseName()
    group = var.group()
    terms = [baseName] + [str(g) for g in group]
    return '_'.join(terms)

  @classmethod
  def _name(cls, var):
    if var.isLocal():
      return str(var)
    return '_ut_' + cls._tensorName(var)

  def _viewName(self, var):
    return '_view_' + self._name(var)
  
  def _groupStr(self, var):
    group = var.group()
    return ','.join([str(g) for g in group])

  def _groupTemplate(self, var):
    gstr = self._groupStr(var)
    return '<{}>'.format(gstr) if gstr else ''

  def _groupIndex(self, var):
    gstr = self._groupStr(var)
    return '({})'.format(gstr) if gstr else ''
  
  def generate(self, cpp, namespace, testName, kernelClass, cfg, gemm_cfg, testFramework, index=None):
    scalars = ScalarsSet().visit(cfg)
    scalars = sorted(scalars, key=str)
    variables = SortedGlobalsList().visit(cfg)
    kernel_prefix = '{}::'.format(namespace) if namespace else ''
    with cpp.Function(**testFramework.functionArgs(testName)):
      factory = UnitTestFactory(cpp, self._arch, self._name, testFramework)

      for i,scalar in enumerate(scalars):
        cpp('{} {} = {};'.format(self._arch.typename, self._tensorNameS(scalar), float(i+2)))
        
      for var in variables:
        factory.tensor(var.tensor, self._tensorName(var))
        factory.temporary(self._name(var), var.memoryLayout().requiredReals(), iniZero=True)
        
        shape = var.memoryLayout().shape()
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
        prefix = '{}::'.format(var.tensor.namespace) if var.tensor.namespace else ''
        cpp( '{prefix}{initNS}::{baseName}::{viewStruct}{groupTemplate}::{createFun}({name}).copyToView({viewName});'.format(
            initNS = InitializerGenerator.INIT_NAMESPACE,
            supportNS = SUPPORT_LIBRARY_NAMESPACE,
            groupTemplate=self._groupTemplate(var.tensor),
            prefix=prefix,
            baseName=var.tensor.baseName(),
            name=self._tensorName(var),
            viewName=self._viewName(var),
            viewStruct=InitializerGenerator.VIEW_STRUCT_NAME,
            createFun=InitializerGenerator.VIEW_FUN_NAME
          )
        )
        cpp.emptyline()

      cpp( '{}{}::{} {};'.format(kernel_prefix, OptimisedKernelGenerator.NAMESPACE, kernelClass, self.KERNEL_VAR) )
      for var in scalars:
        cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.baseName(), self._groupIndex(var), self._tensorNameS(var)) )
      for var in variables:
        cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.tensor.baseName(), self._groupIndex(var.tensor), self._tensorName(var)) )

      cpp( '{}.{}();'.format(self.KERNEL_VAR, OptimisedKernelGenerator.EXECUTE_NAME + (str(index) if index is not None else '')) )
      cpp.emptyline()

      super().generate(cpp, cfg, factory, None, gemm_cfg)

      for var in variables:
        if var.writable:
          factory.compare(var, Variable(self._tensorName(var), False, var.tensor.memoryLayout()))

      factory.freeTmp()

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
      if memLayout.shape():
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
          self.ROWIND_NAME + (index if index is not None else ''),
          self.COLPTR_NAME + (index if index is not None else '')
        )
      )
    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      cpp(self.formatArray(numberType, namespace + self.ROWIND_NAME + index, memLayout.rowIndex(), declarationOnly))
      cpp(self.formatArray(numberType, namespace + self.COLPTR_NAME + index, memLayout.colPointer(), declarationOnly))

  def __init__(self, arch, tensors, scalars):
    self._arch = arch
    self._numberType = '{} const'.format(self._arch.uintTypename)
    self._realType = '{} const'.format(self._arch.typename)
    self._realPtrType = self._realType + '*'
    self._scalarCollect = collections.OrderedDict()
    self._collect = collections.OrderedDict()
    for tensor in tensors:
      baseName = tensor.baseNameWithNamespace()
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
    for scalar in scalars:
      baseName = scalar.baseNameWithNamespace()
      group = scalar.group()
      if baseName not in self._scalarCollect:
        self._scalarCollect[baseName] = {group: scalar}
      elif group not in self._scalarCollect[baseName]:
        groupRef = next(iter(self._scalarCollect[baseName].keys()))
        if len(group) != len(groupRef):
          raise ValueError('Mixed group dimensions are not allowed. ({} and {} for {}.)'.format(group, groupRef, baseName))
        self._scalarCollect[baseName][group] = scalar
      else:
        assert self._scalarCollect[baseName][group] == scalar
    maxIndex = {baseName: tuple(map(max, *groups.keys())) if len(groups) > 1 else next(iter(groups.keys())) for baseName, groups in self._collect.items()}
    self._groupSize = {baseName: tuple(map(lambda x: x+1, mi)) for baseName, mi in maxIndex.items()}
    maxIndexScalar = {baseName: tuple(map(max, *groups.keys())) if len(groups) > 1 else next(iter(groups.keys())) for baseName, groups in self._scalarCollect.items()}
    self._groupSizeScalar = {baseName: tuple(map(lambda x: x+1, mi)) for baseName, mi in maxIndexScalar.items()}
  
  def _tensorViewGenerator(self, memoryLayout):
    memLayoutMap = {
      'DenseMemoryLayout': self.DenseTensorView,
      'CSCMemoryLayout': self.CSCMatrixView
    }
    return memLayoutMap[type(memoryLayout).__name__]()
  
  def iterate_collect(self):
    cur_namespace = ''
    cur_dict = collections.OrderedDict()
    for base_name, tensors in self._collect.items():
      splitName = base_name.rsplit('::', 1)
      if len(splitName) == 1:
        namespace = ''
        base_name_without_ns = splitName[0]
      else:
        namespace, base_name_without_ns = splitName
      if namespace != cur_namespace:
        yield cur_namespace, cur_dict
        cur_namespace = namespace
        cur_dict = {}
      cur_dict[base_name, base_name_without_ns] = tensors
    # Don't forget last namespace
    yield cur_namespace, cur_dict

  def iterate_collect_scalar(self):
    cur_namespace = ''
    cur_dict = collections.OrderedDict()
    for base_name, scalars in self._scalarCollect.items():
      splitName = base_name.rsplit('::', 1)
      if len(splitName) == 1:
        namespace = ''
        base_name_without_ns = splitName[0]
      else:
        namespace, base_name_without_ns = splitName
      if namespace != cur_namespace:
        yield cur_namespace, cur_dict
        cur_namespace = namespace
        cur_dict = {}
      cur_dict[base_name, base_name_without_ns] = scalars
    # Don't forget last namespace
    yield cur_namespace, cur_dict

  def generateTensorsH(self, header):
    for namespace, tensor_dict in self.iterate_collect():
      with header.Namespace(namespace), header.Namespace(self.TENSOR_NAMESPACE):
        for (baseName, baseNameWithoutNamespace), tensors in tensor_dict.items():        
          with header.Struct(baseNameWithoutNamespace):
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
                with header.Function('operator()', typedArgs, '{} T const&'.format(INLINE), const=True):
                  header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
    for namespace, scalar_dict in self.iterate_collect_scalar():
      with header.Namespace(namespace), header.Namespace(self.TENSOR_NAMESPACE):
        for (baseName, baseNameWithoutNamespace), scalars in scalar_dict.items():        
          with header.Struct(baseNameWithoutNamespace):
            groupSize = self._groupSizeScalar[baseName]
            args = ndargs(len(groupSize))
            typedArgs = typedNdArgs(len(groupSize), self._arch.uintTypename)
            if len(groupSize) > 0:
              with header.Function(self.INDEX_FUN_NAME, typedArgs, returnType):
                header('return {};'.format(indexFun(groupSizeToStride(groupSize))))
            if len(groupSize) > 0:
              header('template<typename T>')
              with header.Struct(self.CONTAINER_CLASS_NAME):
                header('T {}[{}];'.format(self.CONTAINER_DATA_NAME, reduce(operator.mul, groupSize)))
                with header.Function(self.CONTAINER_CLASS_NAME, '', ''):
                  pass
                with header.Function('operator()', typedArgs, '{} T&'.format(INLINE)):
                  header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
                with header.Function('operator()', typedArgs, '{} T const&'.format(INLINE), const=True):
                  header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
  
  def generateTensorsCpp(self, cpp):
    for namespace, tensor_dict in self.iterate_collect():
      with cpp.Namespace(namespace):
        for (base_name, base_name_without_namespace), tensors in tensor_dict.items():
          self._tensor(cpp, '::'.join([self.TENSOR_NAMESPACE, base_name_without_namespace, '']), tensors, self._groupSize[base_name], True)
  
  def generateInitH(self, header):
    for namespace, tensor_dict in self.iterate_collect():
      with header.Namespace(namespace), header.Namespace(self.INIT_NAMESPACE):
        for (base_name, base_name_without_namespace), tensors in tensor_dict.items():
          self._init(header, base_name, base_name_without_namespace, '', tensors, False)

  def generateInitCpp(self, cpp):
    for namespace, tensor_dict in self.iterate_collect():
      for (base_name, base_name_without_namespace), tensors in tensor_dict.items():
        prefix_parts = []
        if len(namespace) > 0:
          prefix_parts.append(namespace)
        prefix_parts +=  [self.INIT_NAMESPACE, base_name_without_namespace, '']
        prefix = '::'.join(prefix_parts)
        self._init(cpp, base_name, base_name_without_namespace, prefix, tensors, True)

  def _tensor(self, cpp, name, tensors, groupSize, declarationOnly):
    shape = {group: tensor.shape() for group,tensor in tensors.items()}
    size = {group: [tensor.memoryLayout().requiredReals()] for group,tensor in tensors.items()}
    self._array(cpp, self._numberType, name + self.SHAPE_NAME, shape, groupSize, declarationOnly)
    self._array(cpp, self._numberType, name + self.SIZE_NAME, size, groupSize, declarationOnly, alwaysArray=False)

  def _init(self, cpp, baseName, baseNameWithoutNamespace, name, tensors, declarationOnly):
    groupSize = self._groupSize[baseName]
    stride = groupSizeToStride(groupSize)
    index = lambda group: str(address(group, stride)) if len(group) > 0 else ''

    if declarationOnly:
      for group,tensor in tensors.items():
        ml = tensor.memoryLayout()
        tv = self._tensorViewGenerator(ml)
        tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, True)
      valueNames = dict()
      for group,tensor in tensors.items():
        values = tensor.values()
        memLayout = tensor.memoryLayout()
        if values is not None:
          memory = ['0.']*memLayout.requiredReals()
          for idx,x in values.items():
            memory[memLayout.address(idx)] = x
          valuesName = '{}{}{}'.format(name, self.VALUES_BASENAME, index(group))
          valueNames[group] = ['&{}[0]'.format(valuesName)]
          cpp('{} {}[] = {{{}}};'.format(self._realType, valuesName, ', '.join(memory)))
      if len(valueNames) > 1:
        self._array(cpp, self._realPtrType, name + self.VALUES_BASENAME, valueNames, groupSize, alwaysArray=False, constexpr=False, static=False)
    else:
      with cpp.Struct('{0} : {1}::{0}'.format(baseNameWithoutNamespace, self.TENSOR_NAMESPACE)):
        for group,tensor in tensors.items():
          ml = tensor.memoryLayout()
          tv = self._tensorViewGenerator(ml)
          tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, False)

        nValueArrays = 0
        for group,tensor in tensors.items():
          values = tensor.values()
          if values is not None:
            name = '{}{}'.format(self.VALUES_BASENAME, index(group))
            aligned = ''
            if tensor.memoryLayout().alignedStride():
              aligned = ' __attribute__((aligned({})))'.format(self._arch.alignment)
            cpp('{} {} {}[]{};'.format(STATIC, self._realType, name, aligned))
            nValueArrays += 1
        if nValueArrays > 1:
          cpp('{} {} {}[];'.format(STATIC, self._realPtrType, self.VALUES_BASENAME))

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
          with cpp.Struct('{}::{}<{}>'.format(baseNameWithoutNamespace, self.VIEW_STRUCT_NAME, special)):
            cpp('typedef {} {};'.format(typename, self.VIEW_TYPE_NAME))
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, index(group))
  
  def _array(self, cpp, typ, name, content, groupSize, declarationOnly=False, alwaysArray=True, constexpr=True, static=True):
    cexpr = CONSTEXPR + ' ' if constexpr else ''
    stat = STATIC + ' ' if static else ''
    maxLen = max(map(len, content.values())) if len(content.values()) > 0 else 0

    isGroup = len(groupSize) > 0
    groupIndices = '[]' if isGroup else ''

    isArray = alwaysArray or maxLen > 1
    arrayIndices = '[{}]'.format(maxLen) if isArray else ''
    if maxLen == 0:
      return
    
    if declarationOnly:
      cpp('{}{} {}{}{};'.format(cexpr, typ, name, groupIndices, arrayIndices))
    else:
      formatArray = lambda L: ', '.join([str(x) for x in L])
      if isGroup:
        stride = groupSizeToStride(groupSize)
        size = reduce(operator.mul, groupSize, 1)
        init = ['0']*size
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
      
      cpp('{}{}{} {}{}{} = {};'.format(cexpr, stat, typ, name, groupIndices, arrayIndices, initStr))


