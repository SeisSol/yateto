import collections
import contextlib
import operator
from functools import reduce
from io import StringIO
from ..memory import DenseMemoryLayout
from .. import aspp
from ..controlflow.visitor import DerivedScalarsList, ScalarsSet, SortedGlobalsList, SortedPrefetchList
from ..controlflow.transformer import DetermineLocalInitialization
from ..controlflow.graph import Guard
from ..controlflow.graph import Variable
from .code import Cpp
from .factory import *
from .common import BatchedOperationsAux, KernelAttributes
from ..type import Scalar, Tensor, Datatype

import numpy as np

SUPPORT_LIBRARY_NAMESPACE = 'yateto'
CONSTEXPR = 'constexpr'
STATIC = 'static'
INLINE = 'inline'
MODIFIERS = '{} {}'.format(CONSTEXPR, STATIC)
STATIC_INLINE = '{} {}'.format(STATIC, INLINE)
#: Alignment of the constant pool as a whole. Every entry's place inside the
#: image is an offset from its base, so the alignment an entry was given only
#: survives a copy if the destination is aligned at least this far. This is
#: the floor; a stricter entry raises it, which is why consumers are told the
#: number by poolAlignment() rather than being expected to know it.
POOL_ALIGNMENT = 128

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

  def _generateScalarPrologue(self, cpp, cfg):
    """Compute every scalar-only expression up front.

    They read nothing the kernel produces, so hoisting them here means they are
    evaluated once, before any kernel or external routine call, rather than per
    statement. A guarded statement gets its factor computed regardless of the
    guard, which only matters for an expression that can trap.
    """
    for scalar in DerivedScalarsList().visit(cfg):
      datatype = scalar.getDatatype(self._arch)
      cpp(f'{datatype.ctype()} const {scalar.name()} = {scalar.expression.ccode(self._arch)};')

  def generate(self, cpp, cfg, factory,  routineCache, gemm_cfg):
    hwFlops = 0
    # temporary memory required (per element in case of gpu)
    # NOTE: it is required to know in case if the memory is allocated on the heap
    #       an provided by the user
    required_tmp_mem = 0
    cfg = DetermineLocalInitialization().visit(cfg)
    self._generateScalarPrologue(cpp, cfg)
    if factory.allocateTemporary():
      localPtrs = set()
      for pp in cfg:
        localPtrs.update(pp.bufferMap.keys())
      for localPtr in sorted(localPtrs, key=str):
        cpp(f'{localPtr.datatype.ctype()}* {localPtr};')
    for pp in cfg:
      if factory.allocateTemporary():
        for buf, size in pp.initBuffer.items():
          required_tmp_mem += size
          bufname = self._bufferName(buf)
          # NOTE: size is in bytes here, hence the untyped (int8_t) buffer
          factory.temporary(bufname, size, None)
        for local, buf in pp.bufferMap.items():
          # buffers are untyped storage; each pointer is cast to its own type
          cpp(f'{local} = reinterpret_cast<{local.datatype.ctype()}*>({self._bufferName(buf)});')
      action = pp.action
      if action:
        scalar = self.deduce_scalar(action)
        if action.isRHSExpression():
          prefetchName = '{}.{}'.format(self.PREFETCHVAR_NAME, action.term.node.prefetch.name()) if action.term.node.prefetch is not None else None
          hwFlops += factory.create(action.term.node, action.result, action.term.variableList(), action.condition, action.add, scalar, prefetchName, routineCache, gemm_cfg)
        else:
          hwFlops += factory.simple(action.result, action.term, action.condition, action.add, scalar, routineCache, gemm_cfg)
    return hwFlops, required_tmp_mem

class OptimizedKernelGenerator(KernelGenerator):
  NAMESPACE = 'kernel'
  EXECUTE_NAME = 'execute'
  FIND_EXECUTE_NAME = 'findExecute'
  EXECUTE_ARRAY_NAME = 'ExecutePtrs'
  NONZEROFLOPS_NAME = 'NonZeroFlops'
  HARDWAREFLOPS_NAME = 'HardwareFlops'
  OUTBOUND_BYTES_NAME = 'OutboundBytes'
  INBOUND_CONST_BYTES_NAME = 'InboundConstBytes'
  INBOUND_BYTES_NAME = 'InboundBytes'
  MEMBER_FUNCTION_PTR_NAME = 'member_function_ptr'
  TEMP_MEM_REQUIRED_NAME = 'TmpMemRequiredInBytes'
  TEMP_MAX_MEM_REQUIRED_NAME = 'TmpMaxMemRequiredInBytes'


  def __init__(self, arch, routineCache, routine_exporters):
    super().__init__(arch)
    self._routineCache = routineCache
    self._routine_exporters = routine_exporters

    self._routine_factories = {
      'cpu': OptimizedKernelFactory,
      'gpu': OptimizedKernelFactory
    }

    for entry in routine_exporters:
      self._routine_factories[entry] = ExportFactory.makeFactory(routine_exporters[entry])

  class KernelOutline(object):
    def __init__(self,
                 nonZeroFlops,
                 hwFlops,
                 inConstBytes,
                 inBytes,
                 outBytes,
                 tensors,
                 writable,
                 prefetch,
                 scalars,
                 function,
                 tmp_mem_size,
                 is_compute_constant_tensors,
                 datatype,
                 target,
                 attrs):

      self.nonZeroFlops = nonZeroFlops
      self.hwFlops = hwFlops
      self.inConstBytes = inConstBytes
      self.inBytes = inBytes
      self.outBytes = outBytes
      self.tensors = tensors
      self.writable = writable
      self.prefetch = prefetch
      self.scalars = scalars
      self.function = function
      self.tmp_mem_size = tmp_mem_size
      self.is_compute_constant_tensors = is_compute_constant_tensors
      self.datatype = datatype
      self.target = target
      self.attrs = attrs

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

  def generateKernelOutline(self, nonZeroFlops, cfg, gemm_cfg, target, attrs=None):
    # NOTE: sorted, because these become the kernel's scalar members and a set
    #       enumerates in an order PYTHONHASHSEED varies between runs.
    scalarsP = sorted(ScalarsSet().visit(cfg), key=str)
    variables = SortedGlobalsList().visit(cfg)
    tensors = collections.OrderedDict()
    writable = dict()
    is_compute_constant_tensors = dict()
    scalars = collections.OrderedDict()
    datatype = dict()

    inConstTensors = {}
    inTensors = {}
    outTensors = {}

    # A by-value operand is a scalar wherever it turns up, not only in the
    # scaling slot of an action. Collected here rather than only from there,
    # because a kernel that uses one both ways would otherwise declare the name
    # twice -- once by value and once as a pointer -- and not compile.
    byValue = [var.tensor for var in variables if var.tensor.isPassedByValue()]
    variables = [var for var in variables if not var.tensor.isPassedByValue()]
    for scalar in sorted(set(scalarsP) | set(byValue), key=str):
      self.KernelOutline._addTensor(scalar, scalars)
      datatype[scalar.baseNameWithNamespace()] = scalar.getDatatype(self._arch)
    for var in variables:
      self.KernelOutline._addTensor(var.tensor, tensors)
      bn = var.tensor.baseNameWithNamespace()

      datatype[bn] = var.datatype

      if bn in writable:
        if var.writable:
          writable[bn] = True
      else:
        writable[bn] = var.writable

      is_compute_constant_tensors[bn] = var.tensor.is_compute_constant()

      nm = var.tensor.nameWithNamespace()

      size = var.tensor.memoryLayout().storage().requiredReals() * self._arch.bytesPerReal
      if var.tensor.is_compute_constant():
        inConstTensors[nm] = size
      else:
        if var.writable:
          outTensors[nm] = size
        else:
          inTensors[nm] = size

    inConstBytes = sum(size for size in inConstTensors.values())
    inBytes = sum(size for size in inTensors.values())
    outBytes = sum(size for size in outTensors.values())

    prefetchTensors = SortedPrefetchList().visit(cfg)
    prefetch = collections.OrderedDict()
    for tensor in prefetchTensors:
      self.KernelOutline._addTensor(tensor, prefetch)

    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as fcpp:
      attrs = attrs if attrs is not None else KernelAttributes()
      factory = self._routine_factories[target](fcpp, self._arch, target, attrs)
      hwFlops, tmp_memory = super().generate(fcpp, cfg, factory, self._routineCache, gemm_cfg)
      factory.post_generate(self._routineCache)
      factory.freeTmp()
      factory.reset_stream()
      factory.reset_flags()
      function = functionIO.getvalue()
    return self.KernelOutline(nonZeroFlops,
                              hwFlops,
                              inConstBytes,
                              inBytes,
                              outBytes,
                              tensors,
                              writable,
                              prefetch,
                              scalars,
                              function,
                              tmp_memory,
                              is_compute_constant_tensors,
                              datatype,
                              target,
                              attrs)

  @classmethod
  def _addFromKO(cls, koEntries, entries):
    for key, value in koEntries.items():
      if key not in entries:
        entries[key] = value
      elif entries[key] != value:
        if isinstance(value, Datatype) or isinstance(entries[key], Datatype):
          # NOTE: Datatype is a plain Enum; `|` would raise an opaque TypeError
          raise ValueError(
            f'Conflicting datatypes for "{key}" across the kernels of a family: '
            f'{entries[key]} vs. {value}.')
        entries[key] = entries[key] | value


  def generate(self, cpp, header, name, kernelOutlines, familyStride=None):
    tensors = collections.OrderedDict()
    prefetch = collections.OrderedDict()
    writable = dict()
    scalars = collections.OrderedDict()
    is_compute_constant_tensors = dict()
    datatype = dict()
    for ko in kernelOutlines:
      if ko:
        self._addFromKO(ko.scalars, scalars)
        self._addFromKO(ko.tensors, tensors)
        self._addFromKO(ko.writable, writable)
        self._addFromKO(ko.prefetch, prefetch)
        self._addFromKO(ko.is_compute_constant_tensors, is_compute_constant_tensors)
        self._addFromKO(ko.datatype, datatype)

    target = kernelOutlines[-1].target
    is_same_target = True
    for outline in kernelOutlines:
      if outline:
        is_same_target = True if outline.target == target else False

    if not is_same_target:
      raise RuntimeError("kernels with the same family belong to different compute target.")

    # One struct carries the whole family, so one set of attributes has to
    # describe every member of it: the flags member is either there for all of
    # them or for none.
    attrs = kernelOutlines[-1].attrs
    for outline in kernelOutlines:
      if outline and outline.attrs != attrs:
        raise RuntimeError("kernels within the same family were given different "
                           "attributes.")

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
        def addConst(name, attrcall):
          header('{} {} const {}{} = {};'.format(
            MODIFIERS,
            self._arch.ulongTypename,
            name,
            brackets,
            formatArray([attrcall(kernelOutline) if kernelOutline else 0 for kernelOutline in kernelOutlines])
          ))

        addConst(self.NONZEROFLOPS_NAME, lambda ko: ko.nonZeroFlops)
        addConst(self.HARDWAREFLOPS_NAME, lambda ko: ko.hwFlops)
        addConst(self.INBOUND_CONST_BYTES_NAME, lambda ko: ko.inConstBytes)
        addConst(self.INBOUND_BYTES_NAME, lambda ko: ko.inBytes)
        addConst(self.OUTBOUND_BYTES_NAME, lambda ko: ko.outBytes)

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
          # the buffers are declared as int8_t*, and char and int8_t are
          # distinct types, so the allocator has to hand out int8_t* as well
          header(f'yateto::LinearAllocatorT<{Datatype.I8.ctype()}> linearAllocator;')

        header.emptyline()

        def kernelArgs(base_name_with_namespace, groups, writable, is_constant, datatype, target):
          prefix, base_name = Tensor.splitBasename(base_name_with_namespace)
          typ = datatype.ctype()
          ptr_type = '**' if not is_constant and target == 'gpu' else '*'
          if not writable:
            typ += ' const'
          if len(next(iter(groups))) > 0:
            class_name = f'{prefix}{InitializerGenerator.TENSOR_NAMESPACE}::{base_name}'
            container_type = f'{InitializerGenerator.CONTAINER_CLASS_NAME}<{typ}{ptr_type}>'
            header(f'{class_name}::{container_type} {base_name};')
          else:
            header(f'{typ}{ptr_type} {base_name}{"{"}nullptr{"}"};')

        def scalarArgs(base_name_with_namespace, datatype, groups):
          prefix, base_name = Tensor.splitBasename(base_name_with_namespace)
          typ = datatype.ctype()
          if len(next(iter(groups))) > 0:
            class_name = f'{prefix}{InitializerGenerator.TENSOR_NAMESPACE}::{base_name}'
            container_type = f'{InitializerGenerator.CONTAINER_CLASS_NAME}<{typ}>'
            header(f'{class_name}::{container_type} {base_name};')
          else:
            header(f'{typ} {base_name} = std::numeric_limits<{typ}>::signaling_NaN();')

        for baseName, groups in scalars.items():
          scalarArgs(baseName,
                     datatype[baseName],
                     groups)
        for baseName, groups in tensors.items():
          kernelArgs(baseName,
                     groups,
                     writable[baseName],
                     is_compute_constant_tensors[baseName],
                     datatype[baseName],
                     target)
        header.emptyline()

        # containers with extra offsets for GPU-like computations
        if target == 'gpu':
          header(f'unsigned {BatchedOperationsAux.NUM_ELEMENTS_NAME} = 0;')
          header(f'void *{BatchedOperationsAux.STREAM_PTR_NAME} = {BatchedOperationsAux.FORBIDDEN_STREAM_PTR};')
          # Only where the kernel asked for it: without the member, a caller
          # that means to skip elements fails to compile instead of getting a
          # kernel that computes all of them.
          if attrs.flags:
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
              kernelArgs(baseName, groups, writable=False, is_constant=False, datatype=self._arch.datatype, target='any')
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

          indexer = f'[{indexF}]'
        else:
          args = ''
          indexer = ''

        aux_functions = [self.NONZEROFLOPS_NAME,
                          self.HARDWAREFLOPS_NAME,
                          self.INBOUND_CONST_BYTES_NAME,
                          self.INBOUND_BYTES_NAME,
                          self.OUTBOUND_BYTES_NAME,
                          self.TEMP_MEM_REQUIRED_NAME]

        for function in aux_functions:
          funName = function[:1].lower() + function[1:]
          with header.Function(funName, args, f'{MODIFIERS} {self._arch.ulongTypename}'):
            header(f'return {function}{indexer};')

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
        for base_name_with_namespace, groups in kernelOutline.scalars.items():
          base_name = Tensor.splitBasename(base_name_with_namespace)[-1]
          if len(next(iter(groups))) > 0:
            for gis in groups:
              cpp('assert(!std::isnan({}({})));'.format(base_name, ','.join(str(gi) for gi in gis)))
          else:
            cpp(f'assert(!std::isnan({base_name}));')
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
  CASE_VAR = '_case'
  #: How many condition assignments to enumerate at most. Every one costs a
  #: full run of the kernel and of the reference, and a kernel guarded by
  #: more conditions than this is rare enough to be covered by hand.
  MAX_CASES = 16
  STREAM = '_stream'
  TMP_MEM = '_tmpMem'
  TMP_SIZE = 128 * 8

  def __init__(self, arch):
    super().__init__(arch)

  def deduce_single_scalar(self, scalar):
    if scalar is None:
      return 1.0
    elif isinstance(scalar, Tensor):
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
  def _devTensorName(cls, var):
      return f'_dev_{cls._tensorName(var)}'

  @classmethod
  def _devPtrTensorName(cls, var):
      return f'_dev_ptr_{cls._tensorName(var)}'

  def _devTensorKernelArgument(self, var, writable):
    if var.tensor.is_compute_constant():
      return self._devTensorName(var)
    elif writable[var.tensor.baseNameWithNamespace()]:
      return self._devPtrTensorName(var)
    else:
      return f'const_cast<{var.datatype.ctype()} const**>({self._devPtrTensorName(var)})'

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
    if not var.isGlobal():
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

  @staticmethod
  def _conditionVariables(cfg):
    """The values the kernel's guards read, in a stable order.

    A condition may be written inside the kernel, in which case its later
    reads are a different value under the same variable; the variable is
    what gets filled, so it is what is enumerated.
    """
    seen = {}
    for pp in cfg:
      action = pp.action
      if action is None:
        continue
      guard = Guard.coerce(action.condition)
      if guard.isAlways() or guard.isNever():
        continue
      for var in guard.variables():
        seen.setdefault(str(var), var)
    return [seen[name] for name in sorted(seen)]

  @classmethod
  def _conditionCases(cls, conditions):
    if not conditions:
      return 1
    return min(2 ** len(conditions), cls.MAX_CASES)

  def generate(self, cpp, namespace, testName, kernelClass, cfg, target, gemm_cfg, testFramework, index=None):
    if target == 'gpu':
      if self._arch.backend in ['oneapi', 'acpp', 'hipsycl']:
        # (name queue_op "stream_op" for consistency with the existing C++ interface)

        stream_new = lambda name: cpp(f'auto {name} = new sycl::queue{{sycl::property::queue::in_order()}};')
        stream_delete = lambda name: cpp(f'delete {name};')
        stream_wait = lambda name: cpp(f'{name}->wait_and_throw();')

        data_malloc = lambda name, size, datatype, stream: cpp(f'auto {name} = reinterpret_cast<{datatype}>(sycl::malloc_device({size}, *{stream}));')
        data_free = lambda name, stream: cpp(f'sycl::free({name}, *{stream});')
        data_memcpy = lambda dest, src, size, stream: cpp(f'{stream}->memcpy({dest}, {src}, {size});')

        device_test = True
      elif self._arch.backend in ['cuda', 'hip']:
        backendprefix = self._arch.backend

        stream_new = lambda name: cpp(f'{backendprefix}Stream_t {name};\n{backendprefix}StreamCreateWithFlags(&{name}, {backendprefix}StreamNonBlocking);')
        stream_delete = lambda name: cpp(f'{backendprefix}StreamDestroy({name});')
        stream_wait = lambda name: cpp(f'{backendprefix}StreamSynchronize({name});')

        data_malloc = lambda name, size, datatype, stream: cpp(f'{datatype} {name} = nullptr;\n{backendprefix}Malloc(&{name}, {size});')
        data_free = lambda name, stream: cpp(f'{backendprefix}Free({name});')
        data_memcpy = lambda dest, src, size, stream: cpp(f'{backendprefix}MemcpyAsync({dest}, {src}, {size}, {backendprefix}MemcpyDefault, {stream});')

        device_test = True
      else:
        # NYI
        raise NotImplementedError(f'Device testing is not implemented for {self._arch.backend}')
        device_test = False
    else:
      device_test = False

    scalars = ScalarsSet().visit(cfg)
    scalars = sorted(scalars, key=str)
    variables = SortedGlobalsList().visit(cfg)
    conditions = self._conditionVariables(cfg)
    kernel_prefix = '{}::'.format(namespace) if namespace else ''
    with cpp.Function(**testFramework.functionArgs(testName)):
      # A guarded kernel is several kernels: which statements run depends on
      # the conditions, and filling them from the usual pattern picks one
      # assignment out of the 2^n there are. The whole body -- fill, run,
      # reference, compare -- is repeated once per assignment instead, so a
      # branch that is never taken in one case is taken in another. Every
      # declaration is inside the loop, so each case starts from nothing.
      cases = self._conditionCases(conditions)
      # nullcontext, not an anonymous scope: a kernel with no guard gets the
      # test it always got, down to the indentation
      loop = cpp.For(f'int {self.CASE_VAR} = 0; {self.CASE_VAR} < {cases}; ++{self.CASE_VAR}') \
             if cases > 1 else contextlib.nullcontext()
      with loop:
       factory = UnitTestFactory(cpp, self._arch, self._name, testFramework)

       for i,scalar in enumerate(scalars):
         cpp('{} {} = {};'.format(scalar.getDatatype(self._arch).ctype(), self._tensorNameS(scalar), float(i+2)))

       conditionBit = {str(var): i for i, var in enumerate(conditions)}
       for var in variables:
         bit = conditionBit.get(str(var))
         factory.tensor(var.tensor, self._tensorName(var),
                        caseVar=self.CASE_VAR if (bit is not None and cases > 1) else None,
                        caseBit=bit)
         factory.temporary(self._name(var), var.memoryLayout().requiredReals(), var.datatype, iniZero=True)

         shape = var.memoryLayout().shape()
         cpp('{supportNS}::DenseTensorView<{dim},{datatype},{arch.uintTypename}> {viewName}({utName}, {{{shape}}}, {{{start}}}, {{{stop}}});'.format(
             supportNS = SUPPORT_LIBRARY_NAMESPACE,
             dim=len(shape),
             datatype=var.datatype.ctype(),
             arch = self._arch,
             utName=self._name(var),
             viewName=self._viewName(var),
             shape=', '.join([str(s) for s in shape]),
             start=', '.join([str(s.start) for s in var.memoryLayout().bbox()]),
             stop=', '.join([str(s.stop) for s in var.memoryLayout().bbox()])
           )
         )
         prefix = '{}::'.format(var.tensor.namespace) if var.tensor.namespace else ''
         cpp( '{prefix}{initNS}::{baseName}::{viewStruct}{groupTemplate}::{createFun}({name}).copyToView({viewName});'.format(
             initNS = InitializerGenerator.INIT_NAMESPACE,
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

       kernelTensorName = self._tensorName
       if device_test:
         writable = dict()
         for var in variables:
           bn = var.tensor.baseNameWithNamespace()
           writable[bn] = writable.get(bn, False) or var.writable

         kernelTensorName = lambda var: self._devTensorKernelArgument(var, writable)

         stream_new(self.STREAM)
         data_malloc(self.TMP_MEM, self.TMP_SIZE, f'{Datatype.I8.ctype()}*', self.STREAM)
         for var in variables:
           data_malloc(self._devTensorName(var), f'sizeof({self._tensorName(var)})', f'{var.datatype.ctype()}*', self.STREAM)
           data_malloc(self._devPtrTensorName(var), f'sizeof({var.datatype.ctype()}*)', f'{var.datatype.ctype()}**', self.STREAM)
         for var in variables:
           data_memcpy(self._devTensorName(var), self._tensorName(var), f'sizeof({self._tensorName(var)})', self.STREAM)
           data_memcpy(self._devPtrTensorName(var), f'&{self._devTensorName(var)}', f'sizeof({var.datatype.ctype()}*)', self.STREAM)
         stream_wait(self.STREAM)
         cpp.emptyline()

       cpp( '{}{}::{} {};'.format(kernel_prefix, OptimizedKernelGenerator.NAMESPACE, kernelClass, self.KERNEL_VAR) )
       for var in scalars:
         cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.baseName(), self._groupIndex(var), self._tensorNameS(var)) )
       for var in variables:
         cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.tensor.baseName(), self._groupIndex(var.tensor), kernelTensorName(var)) )

       if device_test:
         cpp( f'{self.KERNEL_VAR}.numElements = 1;' )
         cpp( f'{self.KERNEL_VAR}.linearAllocator.initialize({self.TMP_MEM});' )
         cpp( f'{self.KERNEL_VAR}.streamPtr = reinterpret_cast<void*>({self.STREAM});' )

       cpp( '{}.{}();'.format(self.KERNEL_VAR, OptimizedKernelGenerator.EXECUTE_NAME + (str(index) if index is not None else '')) )
       cpp.emptyline()

       if device_test:
         stream_wait(self.STREAM)
         for var in variables:
           if var.writable:
             data_memcpy(self._tensorName(var), self._devTensorName(var), f'sizeof({self._tensorName(var)})', self.STREAM)
         stream_wait(self.STREAM)
         data_free(self.TMP_MEM, self.STREAM)
         for var in variables:
           data_free(self._devPtrTensorName(var), self.STREAM)
           data_free(self._devTensorName(var), self.STREAM)
         stream_wait(self.STREAM)
         stream_delete(self.STREAM)
         cpp.emptyline()

       super().generate(cpp, cfg, factory, None, gemm_cfg)

       for var in variables:
         if var.writable:
           factory.compare(var, Variable(self._tensorName(var), False, var.tensor.memoryLayout(), datatype=var.datatype))

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
  VIEW_TYPE_NAME_CONST = 'type_const'

  class TensorView(object):
    ARGUMENT_NAME = 'values'

    def __init__(self, datatype):
      self._datatype = datatype

    def typename(self, dim, arch, const):
      constStr = 'true' if const else 'false'
      return f'::{SUPPORT_LIBRARY_NAMESPACE}::{type(self).__name__}<{dim},{self._datatype.ctype()},{arch.uintTypename},{constStr}>'

    def arguments(self, const):
      conststr = ' const*' if const else '*'
      return f'{self._datatype.ctype()}{conststr} {self.ARGUMENT_NAME}'

    def generate(cpp, group, memLayout):
      raise NotImplementedError

    def listToInitializerList(self, lst):
      if isinstance(lst, np.ndarray):
        lst = lst.flatten(order='K')
      return '{{{}}}'.format(', '.join([str(l) for l in lst]))

    def formatArray(self, numberType, name, values, declarationOnly):
      lhs = f'{numberType} {name}[]'
      if declarationOnly:
        return ''
      return f'{MODIFIERS} {lhs} = {self.listToInitializerList(values)};'

  class DenseTensorView(TensorView):
    START_NAME = 'Start'
    STOP_NAME = 'Stop'

    def generate(self, cpp, memLayout, arch, index, const):
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch, const),
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

    def typename(self, dim, arch, const):
      constStr = 'true' if const else 'false'
      return f'::{SUPPORT_LIBRARY_NAMESPACE}::{type(self).__name__}<{self._datatype.ctype()},{arch.uintTypename},{constStr}>'

    def generate(self, cpp, memLayout, arch, index, const):
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch, const),
          self.ARGUMENT_NAME,
          self.listToInitializerList(memLayout.shape()),
          self.ROWIND_NAME + (index if index is not None else ''),
          self.COLPTR_NAME + (index if index is not None else '')
        )
      )
    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      cpp(self.formatArray(numberType, namespace + self.ROWIND_NAME + index, memLayout.rowIndex(), declarationOnly))
      cpp(self.formatArray(numberType, namespace + self.COLPTR_NAME + index, memLayout.colPointer(), declarationOnly))

  class PatternTensorView(TensorView):
    PATTERN_NAME = 'Pattern'

    def typename(self, dim, arch, const):
      constStr = 'true' if const else 'false'
      return f'::{SUPPORT_LIBRARY_NAMESPACE}::{type(self).__name__}<{dim}, {arch.typename}, {arch.uintTypename}, {constStr}>'

    def generate(self, cpp, memLayout, arch, index, const):
      cpp( 'return {}({}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch, const),
          self.ARGUMENT_NAME,
          self.listToInitializerList(memLayout.shape()),
          self.PATTERN_NAME + (index if index is not None else '')
        )
      )

    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      cpp(self.formatArray(numberType, namespace + self.PATTERN_NAME + index, memLayout.pattern(), declarationOnly))

  #: What the pool needs to know about one tensor group: how large the group
  #: is, which element type its entries were stored as, and where each member
  #: of it ended up.
  PoolEntry = collections.namedtuple('PoolEntry', ['groupSize', 'datatype', 'symbols'])

  def __init__(self, arch, tensors, scalars):
    self._arch = arch
    self._numberType = f'{self._arch.uintTypename} const'
    self._pool = dict()
    self._realType = lambda datatype: f'{datatype.ctype()} const'
    self._realPtrType = lambda datatype: self._realType(datatype) + '*'
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
        print(f'The tensor {baseName}{group} has been defined twice. Error.')
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
        print(f'The scalar {baseName}{group} has been defined twice.')
        # having a scalar (family) with the same name should not cause problems
        pass
    maxIndex = {baseName: tuple(map(max, *groups.keys())) if len(groups) > 1 else next(iter(groups.keys())) for baseName, groups in self._collect.items()}
    self._groupSize = {baseName: tuple(map(lambda x: x+1, mi)) for baseName, mi in maxIndex.items()}
    maxIndexScalar = {baseName: tuple(map(max, *groups.keys())) if len(groups) > 1 else next(iter(groups.keys())) for baseName, groups in self._scalarCollect.items()}
    self._groupSizeScalar = {baseName: tuple(map(lambda x: x+1, mi)) for baseName, mi in maxIndexScalar.items()}

  def _tensorViewGenerator(self, tensor):
    memoryLayout = tensor.memoryLayout()
    memLayoutMap = {
      'DenseMemoryLayout': self.DenseTensorView,
      'CSCMemoryLayout': self.CSCMatrixView,
      'PatternMemoryLayout': self.PatternTensorView
    }
    return memLayoutMap[type(memoryLayout).__name__](tensor.getDatatype(self._arch))

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

  def collectPool(self, dataCache):
    """Registers every constant tensor in `dataCache` and reports the symbols.

    The result maps a tensor's base name (namespace included) to its group
    size, its element type and the pool symbol each group ended up under.
    Groups without values are absent: they have nothing to store, and the
    pool leaves the corresponding pointer null.

    The element type is the tensor's own, not the architecture's. A tensor
    may carry a datatype of its own, and an entry stored under the wrong one
    is not a matter of spelling: init binds a reference of the tensor's type
    to it, and that reference does not bind.

    Registration happens here rather than while writing init.cpp so that the
    two stay independent -- the pool is built from the tensors, not from the
    text that was printed for them.
    """
    pool = collections.OrderedDict()
    for baseName, tensors in self._collect.items():
      groupSize = self._groupSize[baseName]
      stride = groupSizeToStride(groupSize)
      symbols = collections.OrderedDict()
      datatype = None
      for group, tensor in tensors.items():
        values = tensor.values()
        if values is None:
          continue
        memLayout = tensor.memoryLayout()
        hint = baseName if len(group) == 0 else '{}_{}'.format(baseName, address(group, stride))
        groupDatatype = tensor.getDatatype(self._arch)
        if datatype is None:
          datatype = groupDatatype
        elif datatype != groupDatatype:
          # One member of a group, one element type: the group is handed out
          # as a Container<T>, and there is no T that fits both.
          raise ValueError('Mixed datatypes are not allowed within a tensor group. '
                           '({} and {} for {}.)'.format(datatype, groupDatatype, baseName))
        # The layout already says whether anything reads this array with
        # aligned loads; asking for a cache line unconditionally would pad
        # every three-by-three matrix out to one.
        alignment = self._arch.cacheline if memLayout.alignedStride() else 1
        symbols[group] = dataCache.add(hint,
                                       [groupDatatype.literal(value) for value in memLayout.pack(values)],
                                       groupDatatype.ctype(),
                                       alignment)
      if symbols:
        pool[baseName] = self.PoolEntry(groupSize, datatype, symbols)
    self._pool = pool
    return pool

  def poolSymbol(self, baseName, group):
    """Pool entry holding exactly what init would otherwise print, if there is one.

    There is one as long as a tensor has a single realisation, which is why
    the initialiser can bind a reference instead of materialising the values a
    second time. Once a tensor is realised in more than one layout, only the
    one that matches what init promises can be bound this way, and the rest of
    them answer None here and get their own array.
    """
    entry = self._pool.get(baseName)
    return entry.symbols.get(group) if entry is not None else None

  def generateInitH(self, header):
    for namespace, tensor_dict in self.iterate_collect():
      with header.Namespace(namespace), header.Namespace(self.INIT_NAMESPACE):
        for (base_name, base_name_without_namespace), tensors in tensor_dict.items():
          self._init(header, base_name, base_name_without_namespace, '', tensors, False)
    for namespace, scalar_dict in self.iterate_collect_scalar():
      with header.Namespace(namespace), header.Namespace(self.INIT_NAMESPACE):
        for (baseName, baseNameWithoutNamespace), scalars in scalar_dict.items():
          with header.Struct('{0} : {1}::{0}'.format(baseNameWithoutNamespace, self.TENSOR_NAMESPACE)):
            # empty forward declaration
            pass

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
        tv = self._tensorViewGenerator(tensor)
        tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, True)
      valueNames = dict()
      for group,tensor in tensors.items():
        values = tensor.values()
        memLayout = tensor.memoryLayout()
        datatype = tensor.getDatatype(self._arch)
        if values is not None:
          valuesName = f'{name}{self.VALUES_BASENAME}{index(group)}'
          valueNames[group] = [f'&{valuesName}[0]']
          if self.poolSymbol(baseName, group) is None:
            memory = [datatype.literal(value) for value in memLayout.pack(values)]
            cpp('{} {}[] = {{{}}};'.format(self._realType(datatype), valuesName, ', '.join(memory)))
          # Otherwise the header has already bound it, and a constexpr
          # reference needs no definition outside the class.
      if len(valueNames) > 1:
        _,prototensor = next(iter(tensors.items()))
        datatype = prototensor.getDatatype(self._arch)
        self._array(cpp, self._realPtrType(datatype), name + self.VALUES_BASENAME, valueNames, groupSize, alwaysArray=False, constexpr=False, static=False)
    else:
      with cpp.Struct('{0} : {1}::{0}'.format(baseNameWithoutNamespace, self.TENSOR_NAMESPACE)):
        for group,tensor in tensors.items():
          ml = tensor.memoryLayout()
          tv = self._tensorViewGenerator(tensor)
          tv.arrays(cpp, ml, self._arch, name, index(group), self._numberType, False)

        nValueArrays = 0
        for group,tensor in tensors.items():
          values = tensor.values()
          datatype = tensor.getDatatype(self._arch)
          if values is not None:
            name = f'{self.VALUES_BASENAME}{index(group)}'
            symbol = self.poolSymbol(baseName, group)
            if symbol is None:
              aligned = ''
              if tensor.memoryLayout().alignedStride():
                aligned = f' __attribute__((aligned({self._arch.cacheline})))'
              cpp('{} {} {}[]{};'.format(STATIC, self._realType(datatype), name, aligned))
            else:
              # Bound here and as a constant expression, for two reasons. A
              # reference whose initialiser lives in another translation unit
              # has to be read before it can be followed, which costs a load
              # and a dynamic relocation at every use; one that is a constant
              # expression is folded to the address instead. And the entry's
              # alignment comes along with the address, where an out-of-line
              # reference would have hidden it.
              #
              # No alignment attribute either: the entry was registered with
              # this layout's own requirement, so the pool declaration already
              # carries it.
              cpp('{} {} {} (&{})[{}] = {}::{}.{};'.format(CONSTEXPR,
                                                           STATIC,
                                                           self._realType(datatype),
                                                           name,
                                                           tensor.memoryLayout().requiredReals(),
                                                           PoolGenerator.STORAGE_NAMESPACE,
                                                           PoolGenerator.STORAGE_VAR_NAME,
                                                           symbol))
            nValueArrays += 1
        if nValueArrays > 1:
          cpp(f'{STATIC} {self._realPtrType(datatype)} {self.VALUES_BASENAME}[];')

        cpp.emptyline()
        if len(groupSize) == 0:
          prototensor = next(iter(tensors.values()))
          ml = prototensor.memoryLayout()
          tv = self._tensorViewGenerator(prototensor)
          viewArgs = tv.arguments(False)
          viewArgsConst = tv.arguments(True)
          with cpp.Struct(self.VIEW_STRUCT_NAME):
            cpp(f'using {self.VIEW_TYPE_NAME} = {tv.typename(len(ml.shape()), self._arch, False)};')
            cpp(f'using {self.VIEW_TYPE_NAME_CONST} = {tv.typename(len(ml.shape()), self._arch, True)};')
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, None, False)
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgsConst, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME_CONST)):
              tv.generate(cpp, ml, self._arch, None, True)
        else:
          typedArgs = typedNdArgs(len(groupSize), self._arch.uintTypename)
          cpp('template<{}> struct {} {{}};'.format(typedArgs, self.VIEW_STRUCT_NAME))

      if len(groupSize) > 0:
        for group,tensor in tensors.items():
          ml = tensor.memoryLayout()
          tv = self._tensorViewGenerator(tensor)
          viewArgs = tv.arguments(False)
          viewArgsConst = tv.arguments(True)
          typename = tv.typename(len(ml.shape()), self._arch, False)
          typenameConst = tv.typename(len(ml.shape()), self._arch, True)
          special = ','.join(str(g) for g in group)
          cpp('template<>')
          with cpp.Struct('{}::{}<{}>'.format(baseNameWithoutNamespace, self.VIEW_STRUCT_NAME, special)):
            cpp(f'using {self.VIEW_TYPE_NAME} = {typename};')
            cpp(f'using {self.VIEW_TYPE_NAME_CONST} = {typenameConst};')
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, index(group), False)
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgsConst, returnType='{} {}'.format(STATIC_INLINE, self.VIEW_TYPE_NAME_CONST)):
              tv.generate(cpp, ml, self._arch, index(group), True)

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

class PoolGenerator(object):
  """Emits the constant pool: one image, one allocation, one copy.

  The image is a struct of named arrays rather than a flat byte blob, so that
  every entry keeps a real array type -- sizeof works, a debugger shows
  something useful, and the offsets come out of offsetof instead of being
  computed here and written down a second time.

  What a consumer binds against is not the image but `Pool`, a table of
  pointers derived from a base address. Pointing it at the image gives the
  host view with no allocation and no copy; pointing it at a device
  allocation gives the device view. Entries that the cache merged are one
  member of the image and two pointers in the table.

  This makes every entry position-independent, which is the condition for
  copying the image in one piece: an entry may not contain an address into
  another one.
  """

  STORAGE_NAMESPACE = 'poolstorage'
  STORAGE_STRUCT_NAME = 'Storage'
  STORAGE_VAR_NAME = 'image'
  POOL_STRUCT_NAME = 'Pool'
  CREATE_FUN_NAME = 'create'
  HOST_FUN_NAME = 'host'
  BYTES_FUN_NAME = 'poolBytes'
  DATA_FUN_NAME = 'poolData'
  ALIGN_FUN_NAME = 'poolAlignment'
  SIZE_TYPE = 'std::size_t'

  def __init__(self, arch, dataCache, pool):
    self._arch = arch
    self._dataCache = dataCache
    self._pool = pool
    self._members = self.assignMembers(pool)

  @classmethod
  def memberName(cls, baseNameWithNamespace):
    """Name a tensor goes by inside `Pool`.

    Flattened rather than nested by namespace: a kernel may read constants
    from several namespaces at once, so one flat table is the only shape that
    lets it bind all of them against a single object.
    """
    return baseNameWithNamespace.replace('::', '_')

  @classmethod
  def assignMembers(cls, pool):
    """Member name per tensor, with the collisions flattening can cause refused.

    Two tensors that differ only in where the namespace separator sat --
    `a::b` and `a_b` -- flatten to the same identifier. Declaring the member
    twice would not compile, and were the name to come from a hint instead
    one of them would quietly write into the other's slot. Say which two, and
    let the caller rename one.
    """
    members = collections.OrderedDict()
    taken = dict()
    for baseName in pool:
      member = cls.memberName(baseName)
      if member in taken:
        raise ValueError('The tensors {} and {} share the pool member {}. '
                         'Rename one of them.'.format(taken[member], baseName, member))
      taken[member] = baseName
      members[baseName] = member
    return members

  @staticmethod
  def _elementPtrType(datatype):
    return '{} const*'.format(datatype.ctype())

  def _memberType(self, baseName, entry):
    elementPtr = self._elementPtrType(entry.datatype)
    if len(entry.groupSize) == 0:
      return elementPtr
    prefix, name = Tensor.splitBasename(baseName)
    return '{}{}::{}::{}<{}>'.format(prefix,
                                     InitializerGenerator.TENSOR_NAMESPACE,
                                     name,
                                     InitializerGenerator.CONTAINER_CLASS_NAME,
                                     elementPtr)

  def _storageType(self):
    return '{}::{}'.format(self.STORAGE_NAMESPACE, self.STORAGE_STRUCT_NAME)

  def generateH(self, header):
    with header.Namespace(self.STORAGE_NAMESPACE):
      with header.Struct('alignas({}) {}'.format(POOL_ALIGNMENT, self.STORAGE_STRUCT_NAME)):
        for entry in self._dataCache.entries():
          alignment = 'alignas({}) '.format(entry.alignment()) if entry.alignment() > 1 else ''
          header('{}{} const {}[{}];'.format(alignment,
                                             entry.typename(),
                                             entry.name(),
                                             entry.elements()))
      header.emptyline()
      header('//! The image itself, so that init can bind references into it.')
      header('extern {} const {};'.format(self.STORAGE_STRUCT_NAME, self.STORAGE_VAR_NAME))
    header.emptyline()

    with header.Struct(self.POOL_STRUCT_NAME):
      for baseName, entry in self._pool.items():
        header('{} {}{{}};'.format(self._memberType(baseName, entry),
                                   self._members[baseName]))
      header.emptyline()
      header('//! Table for an image that lives at `base`, host or device.')
      header.functionDeclaration(self.CREATE_FUN_NAME,
                                 'void const* base',
                                 '{} {}'.format(STATIC, self.POOL_STRUCT_NAME))
      header('//! Table for the image in this binary. No allocation, no copy.')
      header.functionDeclaration(self.HOST_FUN_NAME,
                                 '',
                                 '{} {}'.format(STATIC, self.POOL_STRUCT_NAME))
    header.emptyline()

    header('//! Size of the image, for one allocation of one block.')
    header.functionDeclaration(self.BYTES_FUN_NAME, '', self.SIZE_TYPE)
    header('//! Alignment the allocation has to meet for create() to hold.')
    header.functionDeclaration(self.ALIGN_FUN_NAME, '', self.SIZE_TYPE)
    header('//! Address of the image, for one copy of one block.')
    header.functionDeclaration(self.DATA_FUN_NAME, '', 'void const*')

  def generateCpp(self, cpp):
    with cpp.Namespace(self.STORAGE_NAMESPACE):
      entries = self._dataCache.entries()
      cpp('extern {} const {} = {{'.format(self.STORAGE_STRUCT_NAME, self.STORAGE_VAR_NAME))
      for i, entry in enumerate(entries):
        separator = ',' if i + 1 < len(entries) else ''
        cpp('  {{{}}}{}'.format(', '.join(str(value) for value in entry.values()), separator))
      cpp('};')
    cpp.emptyline()

    returnType = self.POOL_STRUCT_NAME
    with cpp.Function('{}::{}'.format(self.POOL_STRUCT_NAME, self.CREATE_FUN_NAME),
                      'void const* base',
                      returnType):
      cpp('auto const* origin = static_cast<char const*>(base);')
      cpp('{} result;'.format(self.POOL_STRUCT_NAME))
      for baseName, entry in self._pool.items():
        member = self._members[baseName]
        stride = groupSizeToStride(entry.groupSize)
        for group, symbol in entry.symbols.items():
          target = member if len(group) == 0 else '{}.{}[{}]'.format(
            member, InitializerGenerator.CONTAINER_DATA_NAME, address(group, stride))
          cpp('result.{} = reinterpret_cast<{}>(origin + offsetof({}, {}));'.format(
            target, self._elementPtrType(entry.datatype), self._storageType(), symbol))
      cpp('return result;')
    cpp.emptyline()

    with cpp.Function('{}::{}'.format(self.POOL_STRUCT_NAME, self.HOST_FUN_NAME), '', returnType):
      cpp('return {}({}());'.format(self.CREATE_FUN_NAME, self.DATA_FUN_NAME))
    cpp.emptyline()

    with cpp.Function(self.BYTES_FUN_NAME, '', self.SIZE_TYPE):
      cpp('return sizeof({});'.format(self._storageType()))
    cpp.emptyline()

    with cpp.Function(self.ALIGN_FUN_NAME, '', self.SIZE_TYPE):
      # Read off the image rather than repeated from POOL_ALIGNMENT: an entry
      # asking for more than the floor raises the struct, and a consumer that
      # allocated for the floor would then be one entry short.
      cpp('return alignof({});'.format(self._storageType()))
    cpp.emptyline()

    with cpp.Function(self.DATA_FUN_NAME, '', 'void const*'):
      cpp('return &{}::{};'.format(self.STORAGE_NAMESPACE, self.STORAGE_VAR_NAME))
