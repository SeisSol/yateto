import collections
import contextlib
import hashlib
import operator
import re
from functools import reduce
from io import StringIO
from ..memory import DenseMemoryLayout, PreparedImage
from .. import aspp
from ..controlflow.visitor import DerivedScalarsList, ScalarsSet, SortedGlobalsList, SortedPrefetchList
from ..controlflow.transformer import DetermineLocalInitialization
from ..controlflow.graph import Guard
from ..controlflow.graph import Variable
from .arrangement import Arrangement, layoutTag
from .code import Block, Cpp
from .factory import *
from .flops import FlopCount
from .common import BatchedOperationsAux, KernelAttributes
from ..type import Scalar, Tensor, Datatype

import numpy as np

SUPPORT_LIBRARY_NAMESPACE = 'yateto'
# Whatever a consumer may call from device code carries the marker from
# yateto/Marker.h. constexpr alone is not enough: clang treats a constexpr
# function as callable from both sides, but nvcc only does so with
# --expt-relaxed-constexpr, and neither covers a function that is not one.
HOSTDEVICE = 'YATETO_HOSTDEVICE'
CONSTEXPR = 'constexpr'
STATIC = 'static'
INLINE = 'inline'
MODIFIERS = '{} {} {}'.format(HOSTDEVICE, CONSTEXPR, STATIC)
# For data, not functions. An execution space is a property of code: nvcc
# refuses one on a variable ("memory qualifier on data member is not
# allowed"), and a constexpr datum needs none to be read on either side.
DATA_MODIFIERS = '{} {}'.format(CONSTEXPR, STATIC)
STATIC_INLINE = '{} {} {}'.format(HOSTDEVICE, STATIC, INLINE)
HOSTDEVICE_INLINE = '{} {}'.format(HOSTDEVICE, INLINE)
#: Alignment floor, for the image and for every entry in it.
#:
#: For the image, because an entry's place inside it is an offset from its
#: base: the alignment an entry was given only survives a copy if the
#: destination is aligned at least this far.
#:
#: For each entry, because entries are not only read one at a time. Constant
#: operands that a merged kernel picks between at runtime have to be reachable
#: with one stride, and a floor every entry meets is what makes the stride the
#: same for all of them regardless of how large each one happens to be. An
#: entry whose layout asks for more than the floor gets more; nothing gets
#: less, so consumers are told the number by poolAlignment() rather than being
#: expected to know it.
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
    hwFlops = FlopCount()
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
          hwFlops += factory.assign(action.result, action.term, action.condition, action.add, scalar, routineCache, gemm_cfg)
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
  BIND_GLOBALS_NAME = 'bindGlobals'
  BIND_GLOBALS_ARGUMENT = 'pool'


  def __init__(self, arch, routineCache, dataCache, routine_exporters, namespace='',
               families=None):
    super().__init__(arch)
    self._routineCache = routineCache
    #: Reachable while the kernels are written out, and not only afterwards,
    #: so that a generator deciding how it wants an operand laid out can put
    #: that arrangement into the pool at the moment it decides.
    self._dataCache = dataCache
    #: Which members each tensor family has, so that a kernel reading one of
    #: them can still speak about the whole of it. A family is handed out as
    #: one table of pointers, so what it is asked about is how all of it is
    #: held, not the part this kernel happens to touch.
    self._families = families or {}
    #: Every arrangement a kernel was generated against, per family. The pool
    #: is filled from this rather than from the tensors alone: an arrangement
    #: nobody reads need not be stored, and one that two kernels disagree
    #: about has to be stored twice.
    self._arrangements = collections.OrderedDict()
    self._routine_exporters = routine_exporters
    self._poolType = '::{}{}'.format('{}::'.format(namespace) if namespace else '',
                                     PoolGenerator.POOL_STRUCT_NAME)

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
                 layouts,
                 target,
                 attrs,
                 inMemory=None,
                 reads=None,
                 offered=None):

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
      #: The arrangement each operand's family is generated against. Which
      #: entry in the pool a kernel reads follows from it, so it has to be
      #: known here, where the kernel is written, and not only where the pool
      #: is filled.
      self.layouts = layouts
      self.target = target
      self.attrs = attrs
      #: Immediate operands this kernel reads from memory, by name, with the
      #: operations that could not take them as they are addressed.
      self.inMemory = inMemory if inMemory is not None else {}
      #: Per family, the groups this kernel reads. What the arrangement says
      #: about the others is the family's own layout, not a claim of this
      #: kernel's, which is what lets its variants be put together.
      self.reads = reads if reads is not None else {}
      #: Per family, the members this kernel reads in an arrangement its
      #: generator asked for, which only the pool holds.
      self.offered = offered if offered is not None else {}

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
    members = dict()

    inConstTensors = {}
    inTensors = {}
    outTensors = {}

    # The body first: which immediate operands a generator could not take, and
    # so reads from memory after all, is only known once it has been asked.
    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as fcpp:
      attrs = attrs if attrs is not None else KernelAttributes()
      factory = self._routine_factories[target](fcpp, self._arch, target, attrs,
                                                self._dataCache)
      hwFlops, tmp_memory = super().generate(fcpp, cfg, factory, self._routineCache, gemm_cfg)
      factory.post_generate(self._routineCache)
      factory.freeTmp()
      factory.reset_stream()
      factory.reset_flags()
      function = functionIO.getvalue()
    inMemory = factory.inMemory

    # A by-value operand is a scalar wherever it turns up, not only in the
    # scaling slot of an action. Collected here rather than only from there,
    # because a kernel that uses one both ways would otherwise declare the name
    # twice -- once by value and once as a pointer -- and not compile.
    byValue = [var.tensor for var in variables if var.tensor.isPassedByValue()]
    # An operand whose data is in the generated code has nothing to pass and
    # nothing to bind: neither a member nor a scalar. It stays in the
    # initializer namespace, where it describes a tensor rather than an
    # argument -- unless one of the generators here read it from memory, in
    # which case it is a constant member like any other.
    variables = [var for var in variables
                 if not var.tensor.isPassedByValue()
                 and (var.tensor.isPassedAsArgument()
                      or var.tensor.nameWithNamespace() in inMemory)]
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

      # Noted per member. The family is what is arranged, so the layouts are
      # put together into one once every operand has been seen.
      members.setdefault(bn, collections.OrderedDict())[var.tensor.group()] = \
        var.tensor.memoryLayout()

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

    # Over the whole family where the family is known, so that two kernels
    # reading different members of it still name the same arrangement of it.
    # A member this kernel never reads is in the table it is handed all the
    # same, and a family of tensors laid out differently is that family held
    # one way rather than two arrangements in conflict.
    layouts = {baseName: Arrangement.of(self._families[baseName])
               if baseName in self._families else Arrangement(byGroup)
               for baseName, byGroup in members.items()}

    prefetchTensors = SortedPrefetchList().visit(cfg)
    prefetch = collections.OrderedDict()
    for tensor in prefetchTensors:
      self.KernelOutline._addTensor(tensor, prefetch)

    # Counted by whoever issued it: a generator working in a precision other
    # than the operation's is the only one that can say what it issued, and
    # in what.
    hwFlops = hwFlops + FlopCount(factory.flopReport())

    # Asked once the kernel is built, because only then does the generator
    # know how it wants to read what it reads, and once the operands are
    # known, because that is what an offering is checked against. Constants
    # only: an operand the caller fills in is arranged by whoever fills it.
    # The description names a tensor as the kernel does, without its
    # namespace, and an offering answers in those names.
    described = collections.defaultdict(set)
    offeredMembers = collections.defaultdict(set)
    for bn in members:
      described[Tensor.splitBasename(bn)[1]].add(bn)
    for offeredName, offered in factory.layoutOfferings().items():
      # Offered under whatever name the tensor was described by, so a member
      # of a family names that member, and only that member is rearranged.
      named = Tensor.isValidName(offeredName)
      baseName = Tensor.getBaseName(offeredName) if named else offeredName
      group = Tensor.getGroup(offeredName) if named else tuple()
      if baseName not in layouts:
        candidates = described.get(baseName, set())
        if len(candidates) > 1:
          raise ValueError('{} offered an arrangement for {}, which this kernel reads '
                           'from more than one namespace ({}).'.format(
                             type(factory).__name__, offeredName, ', '.join(sorted(candidates))))
        if candidates:
          baseName = next(iter(candidates))
      if baseName not in layouts:
        raise ValueError('{} offered an arrangement for {}, which it does not '
                         'read.'.format(type(factory).__name__, baseName))
      if not is_compute_constant_tensors[baseName] or writable[baseName]:
        raise ValueError('{} offered an arrangement for {}, which is not a '
                         'constant it only reads.'.format(type(factory).__name__, baseName))
      layout = layouts[baseName].layoutOf(group)
      if layout is None:
        raise ValueError('{} offered an arrangement for {}, which it does not '
                         'read.'.format(type(factory).__name__, offeredName))
      layouts[baseName] = layouts[baseName].withMember(
        group, self._grantOffering(offeredName, layout, offered))
      offeredMembers[baseName].add(group)

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
                              layouts,
                              target,
                              attrs,
                              inMemory,
                              {baseName: frozenset(byGroup) for baseName, byGroup in members.items()},
                              {baseName: frozenset(groups) for baseName, groups in offeredMembers.items()})

  def arrangements(self):
    """Per tensor, the arrangements the kernels actually read it in."""
    return self._arrangements

  @staticmethod
  def _grantOffering(baseName, layout, offering):
    """The arrangement an offering asks for, checked against the tensor.

    An offering names what it wants, not how to get it: the layout is built
    here, from the one the tensor already has, so that what comes back is
    something this yateto can address and pack. A field it does not know is
    refused by name rather than ignored, because an arrangement granted in
    part is an arrangement nobody asked for.
    """
    unknown = set(offering) - {'order', 'data', 'parts', 'planar'}
    if unknown:
      raise ValueError('Offering for {} asks for {}, which this yateto cannot '
                       'grant.'.format(baseName, ', '.join(sorted(unknown))))
    order = offering.get('order')
    data = offering.get('data')
    if data is not None:
      if order is not None:
        raise ValueError('Offering for {} asks both for an order and for numbers '
                         'of its own; the numbers already carry one.'.format(baseName))
      images = data if isinstance(data, dict) else {baseName: data}
      return PreparedImage(images,
                           parts=int(offering.get('parts', 1)),
                           planar=bool(offering.get('planar', False)))
    if offering.get('parts', 1) != 1 or offering.get('planar', False):
      raise ValueError('Offering for {} asks for a prepared shape without the '
                       'numbers to fill it.'.format(baseName))
    if order is None:
      return layout
    if not hasattr(layout, 'reordered'):
      raise ValueError('Offering for {} asks for an axis order, which a {} '
                       'cannot be given.'.format(baseName, type(layout).__name__))
    return layout.reordered(list(order))

  @classmethod
  def interface(cls, name, namespace, kernelOutlines, families, familyStride=None):
    """What code outside the kernel sets on it, as plain data.

    For the metagen, which writes a way into the kernel for operands whose
    layout is only known at run time. Put together over the variants of a
    family the way `generate` puts together the members of its struct, so that
    both agree on which operand is a constant `bindGlobals` fills. Every
    operand is listed with the groups a caller hands over: all the kernel
    reads of it where the pool does not bind it, and where it does, the
    members the pool holds no values for. `kernels` says, per variant, which
    of those it uses and which operands it writes, so that a caller only
    hands over what the variant it runs uses.
    """
    tensors = collections.OrderedDict()
    writable = dict()
    constant = dict()
    datatype = dict()
    scalars = collections.OrderedDict()
    for ko in kernelOutlines:
      if ko:
        cls._addFromKO(ko.tensors, tensors)
        cls._addFromKO(ko.writable, writable)
        cls._addFromKO(ko.is_compute_constant_tensors, constant)
        cls._addFromKO(ko.datatype, datatype)
        cls._addFromKO(ko.scalars, scalars)

    def rankOf(groups):
      return len(next(iter(groups)))

    def listed(groups):
      return sorted(list(group) for group in groups)

    operands = []
    handed = dict()
    for baseName, groups in tensors.items():
      bound = bool(constant.get(baseName)) and not writable.get(baseName)
      family = families.get(baseName, {})
      handed[baseName] = {group for group in groups
                          if not bound or family.get(group) is None or family[group].values() is None}
      operands.append({'name': baseName,
                       'member': Tensor.splitBasename(baseName)[1],
                       'rank': rankOf(groups),
                       'groups': listed(handed[baseName]),
                       'writable': bool(writable.get(baseName)),
                       'bound': bound,
                       'datatype': str(datatype[baseName])})
    kernels = []
    for position, ko in enumerate(kernelOutlines):
      if ko:
        uses = {baseName: listed(groups & handed[baseName]) for baseName, groups in ko.tensors.items()}
        kernels.append({'position': position,
                        'uses': {baseName: groups for baseName, groups in uses.items() if groups},
                        'writes': sorted(baseName for baseName, written in ko.writable.items() if written)})
    return {
      'name': name,
      'namespace': namespace,
      'family': None if familyStride is None else {
        'stride': list(familyStride),
        'size': len(kernelOutlines),
      },
      'operands': operands,
      'scalars': [{'name': baseName,
                   'member': Tensor.splitBasename(baseName)[1],
                   'rank': rankOf(groups),
                   'groups': listed(groups),
                   'datatype': str(datatype[baseName])}
                  for baseName, groups in scalars.items()],
      'kernels': kernels,
    }

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
    #: Per family, the arrangements the variants read it in, each with the
    #: members read that way and the variant that read each first. One, as a
    #: rule; the first is bound to the member named after the family.
    bindings = dict()
    #: Per variant, which of those arrangements it reads each family in.
    positions = collections.defaultdict(dict)
    for index, ko in enumerate(kernelOutlines):
      if ko:
        self._addFromKO(ko.scalars, scalars)
        self._addFromKO(ko.tensors, tensors)
        self._addFromKO(ko.writable, writable)
        self._addFromKO(ko.prefetch, prefetch)
        self._addFromKO(ko.is_compute_constant_tensors, is_compute_constant_tensors)
        self._addFromKO(ko.datatype, datatype)
        # The variants of one kernel share its members, so a constant is bound
        # in one arrangement for all of them wherever that is possible. Each
        # variant speaks only for the members it reads -- one reading
        # plusFluxMatrices(0) rearranged and another plusFluxMatrices(1) is one
        # arrangement with both rearranged. Two that read one member
        # differently cannot share it, and the one that disagrees with every
        # arrangement so far gets one of its own, bound to a member of its own.
        for baseName, arrangement in ko.layouts.items():
          constant = is_compute_constant_tensors.get(baseName) and not writable.get(baseName)
          reads = ko.reads.get(baseName, ())
          if baseName not in bindings or not constant:
            bindings[baseName] = [(arrangement, {group: index for group in reads})]
            for which in positions.values():
              which.pop(baseName, None)
            positions[index][baseName] = 0
            continue
          held = bindings[baseName]
          for position, (merged, readers) in enumerate(held):
            if all(group not in readers
                   or layoutTag(merged.layoutOf(group)) == layoutTag(arrangement.layoutOf(group))
                   for group in reads):
              break
          else:
            position = len(held)
            held.append((arrangement, {group: index for group in reads}))
          merged, readers = held[position]
          for group in reads:
            if group not in readers:
              merged = merged.withMember(group, arrangement.layoutOf(group))
              readers[group] = index
          held[position] = (merged, readers)
          positions[index][baseName] = position

    #: Whether bindGlobals binds the family, for the kernel as a whole.
    isBound = lambda baseName: is_compute_constant_tensors.get(baseName) and not writable.get(baseName)

    # Only bindGlobals can hand a variant an arrangement its generator asked
    # for, and it binds constants only. A family another variant writes, or
    # one that is not a constant for the kernel as a whole, is filled by the
    # caller as it lays itself out -- which is not what that variant reads.
    for index, ko in enumerate(kernelOutlines):
      if ko:
        for baseName, groups in ko.offered.items():
          if groups and not isBound(baseName):
            raise ValueError(
              'Variant {} of {} reads {} in an arrangement its generator asked for, '
              'but the kernel does not bind {} from the pool: another variant writes '
              'it, or it is not a constant for all of them.'.format(
                index, name, baseName, Tensor.splitBasename(baseName)[1]))

    # The member named after the family keeps the family's own arrangement
    # wherever some variant reads it that way, so that it means what
    # init::X::Values means to whoever fills it by hand; the arrangements a
    # generator asked for are the ones that get members of their own.
    for baseName, held in bindings.items():
      if len(held) < 2 or baseName not in self._families:
        continue
      own = Arrangement.of(self._families[baseName]).tag()
      first = next((position for position, (arrangement, _) in enumerate(held)
                    if arrangement.tag() == own), 0)
      if first:
        order = [first] + [position for position in range(len(held)) if position != first]
        bindings[baseName] = [held[position] for position in order]
        for which in positions.values():
          if baseName in which:
            which[baseName] = order.index(which[baseName])

    #: Per variant, the families it reads in an arrangement other than the
    #: first, and which one.
    alternates = collections.defaultdict(collections.OrderedDict)
    for index, which in positions.items():
      for baseName, position in which.items():
        if position > 0 and isBound(baseName):
          alternates[index][baseName] = position

    # Recorded once the variants are put together, so that the pool holds what
    # the kernel reads and not what one variant of it was going to read on the
    # way there.
    for baseName, held in bindings.items():
      for arrangement, _ in held:
        self._arrangements.setdefault(baseName, collections.OrderedDict())[
          arrangement.tag()] = arrangement

    # The member a further arrangement is bound to. Named after the pool entry
    # it points at, so that the two can be found from one another.
    def alternateName(baseName, position):
      _, memberName = Tensor.splitBasename(baseName)
      return '{}_{}'.format(memberName, bindings[baseName][position][0].tag())
    taken = {Tensor.splitBasename(baseName)[1] for baseName in list(tensors) + list(scalars)}
    for baseName, held in bindings.items():
      for position in range(1, len(held)):
        if alternateName(baseName, position) in taken:
          raise ValueError('{} needs a member {} for the variants that read it laid out '
                           'differently, and a tensor of that name is already one.'.format(
                             name, alternateName(baseName, position)))

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
            DATA_MODIFIERS,
            self._arch.ulongTypename,
            name,
            brackets,
            formatArray([attrcall(kernelOutline) if kernelOutline else 0 for kernelOutline in kernelOutlines])
          ))

        addConst(self.NONZEROFLOPS_NAME, lambda ko: ko.nonZeroFlops)
        # What the total is made of, where it is made of more than one kind of
        # arithmetic. Said here rather than summed silently: a kernel that
        # reaches its result in a narrower precision issues a count in a
        # currency of its own, and dividing it by one peak gives a figure that
        # means nothing.
        for index, ko in enumerate(kernelOutlines):
          counted = FlopCount(ko.hwFlops) if ko is not None else FlopCount()
          if counted.isPlain():
            continue
          header('// {}{}: {}'.format(
            self.HARDWAREFLOPS_NAME,
            '[{}]'.format(index) if brackets else '',
            ', '.join('{} {}'.format(count, kind)
                      for kind, count in counted.kinds().items())))
        addConst(self.HARDWAREFLOPS_NAME, lambda ko: ko.hwFlops)
        addConst(self.INBOUND_CONST_BYTES_NAME, lambda ko: ko.inConstBytes)
        addConst(self.INBOUND_BYTES_NAME, lambda ko: ko.inBytes)
        addConst(self.OUTBOUND_BYTES_NAME, lambda ko: ko.outBytes)

        # tmp mem required by a kernel(s)
        tmp_mem_list = [kernelOutline.tmp_mem_size if kernelOutline else 0 for kernelOutline in kernelOutlines]
        header('{} {} const {}{} = {};'.format(DATA_MODIFIERS,
                                               self._arch.ulongTypename,
                                               self.TEMP_MEM_REQUIRED_NAME,
                                               brackets,
                                               formatArray(tmp_mem_list)))

        header('{} {} const {} = {};'.format(DATA_MODIFIERS,
                                             self._arch.ulongTypename,
                                             self.TEMP_MAX_MEM_REQUIRED_NAME,
                                             max(tmp_mem_list)))

        if target == 'gpu':
          # LinearAllocatorT controls external extra mem. allocated on gpu for tmp. variables
          # the buffers are declared as int8_t*, and char and int8_t are
          # distinct types, so the allocator has to hand out int8_t* as well
          header(f'yateto::LinearAllocatorT<{Datatype.I8.ctype()}> linearAllocator;')

        header.emptyline()

        def kernelArgs(base_name_with_namespace, groups, writable, is_constant, datatype, target,
                       member=None):
          prefix, base_name = Tensor.splitBasename(base_name_with_namespace)
          member = member if member is not None else base_name
          typ = datatype.ctype()
          ptr_type = '**' if not is_constant and target == 'gpu' else '*'
          if not writable:
            typ += ' const'
          if len(next(iter(groups))) > 0:
            class_name = f'{prefix}{InitializerGenerator.TENSOR_NAMESPACE}::{base_name}'
            container_type = f'{InitializerGenerator.CONTAINER_CLASS_NAME}<{typ}{ptr_type}>'
            header(f'{class_name}::{container_type} {member};')
          else:
            header(f'{typ}{ptr_type} {member}{"{"}nullptr{"}"};')

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
        # Which variants read a constant in which of its arrangements, where
        # they do not all read it in one.
        def readersOf(baseName, position):
          return [index for index, ko in enumerate(kernelOutlines)
                  if ko and baseName in ko.layouts
                  and alternates.get(index, {}).get(baseName, 0) == position]

        def variantList(indices):
          return 'variant{} {}'.format('s' if len(indices) > 1 else '', ', '.join(map(str, indices)))

        for baseName, groups in tensors.items():
          if isBound(baseName) and len(bindings[baseName]) > 1:
            header('//! {} as {} read{} it; the others read the members further down.'.format(
              Tensor.splitBasename(baseName)[1],
              variantList(readersOf(baseName, 0)),
              '' if len(readersOf(baseName, 0)) > 1 else 's'))
          kernelArgs(baseName,
                     groups,
                     writable[baseName],
                     is_compute_constant_tensors[baseName],
                     datatype[baseName],
                     target)
        header.emptyline()

        # Which of the members the caller does not have to fill in itself.
        # Writable ones are excluded even when they carry values: the member
        # is a pointer to mutable memory and the pool hands out const.
        constants = [baseName for baseName in tensors
                     if is_compute_constant_tensors[baseName] and not writable[baseName]]

        # A constant that variants read laid out differently, once more for
        # each further arrangement. Only bindGlobals fills these: a variant
        # reading one of them reads it in place of the member named after the
        # family, so assigning that member by hand does not reach it.
        for baseName in constants:
          for position in range(1, len(bindings[baseName])):
            readBy = readersOf(baseName, position)
            header('//! {} as {} read{} it; bound by {} only.'.format(
              Tensor.splitBasename(baseName)[1],
              variantList(readBy),
              '' if len(readBy) > 1 else 's',
              self.BIND_GLOBALS_NAME))
            kernelArgs(baseName,
                       tensors[baseName],
                       writable[baseName],
                       is_compute_constant_tensors[baseName],
                       datatype[baseName],
                       target,
                       member=alternateName(baseName, position))
        if any(len(bindings[baseName]) > 1 for baseName in constants):
          header.emptyline()
        # Emitted even when there is nothing to bind, so that "bind the globals
        # of every kernel" is a rule a caller can follow without knowing which
        # operands a kernel happens to have. The failure modes are not
        # symmetric: a missing call is a null pointer at run time, a redundant
        # one is nothing. It also keeps the interface stable when a kernel
        # gains or loses its last constant operand.
        header('//! Points every constant operand at its entry in `{}`.'.format(
          self.BIND_GLOBALS_ARGUMENT))
        with header.Function(self.BIND_GLOBALS_NAME,
                             '{} const& {}'.format(self._poolType, self.BIND_GLOBALS_ARGUMENT)):
          if not constants:
            header('static_cast<void>({});'.format(self.BIND_GLOBALS_ARGUMENT))
          for baseName in constants:
            _, memberName = Tensor.splitBasename(baseName)
            for position, (arrangement, _) in enumerate(bindings[baseName]):
              header('{} = {}.{};'.format(alternateName(baseName, position) if position else memberName,
                                          self.BIND_GLOBALS_ARGUMENT,
                                          PoolGenerator.memberName(baseName, arrangement)))
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
            DATA_MODIFIERS,
            self.MEMBER_FUNCTION_PTR_NAME,
            self.EXECUTE_ARRAY_NAME,
            formatArray(['&{}::{}'.format(name, executeName(index)) if kernelOutline else 'nullptr' for index, kernelOutline in enumerate(kernelOutlines)])
          ))
          args = typedNdArgs(len(familyStride), self._arch.uintTypename)
          indexF = indexFun(familyStride)
          familySize = len(kernelOutlines)
          boundsCheck = 'assert({} < {});'.format(indexF, familySize)
          with header.Function(self.FIND_EXECUTE_NAME, args, '{} {}'.format(MODIFIERS, self.MEMBER_FUNCTION_PTR_NAME)):
            header(boundsCheck)
            header('return {}[{}];'.format(self.EXECUTE_ARRAY_NAME, indexF))
          with header.Function(self.EXECUTE_NAME, args, '{} void'.format(INLINE)):
            ndArgList = ', '.join(ndargs(len(familyStride)))
            header('assert({}({}) != nullptr);'.format(self.FIND_EXECUTE_NAME, ndArgList))
            header('(this->*{}({}))();'.format(self.FIND_EXECUTE_NAME, ndArgList))

          indexer = f'[{indexF}]'
        else:
          args = ''
          indexer = ''
          boundsCheck = None

        aux_functions = [self.NONZEROFLOPS_NAME,
                          self.HARDWAREFLOPS_NAME,
                          self.INBOUND_CONST_BYTES_NAME,
                          self.INBOUND_BYTES_NAME,
                          self.OUTBOUND_BYTES_NAME,
                          self.TEMP_MEM_REQUIRED_NAME]

        for function in aux_functions:
          funName = function[:1].lower() + function[1:]
          with header.Function(funName, args, f'{MODIFIERS} {self._arch.ulongTypename}'):
            if boundsCheck is not None:
              header(boundsCheck)
            header(f'return {function}{indexer};')

    for index, kernelOutline in enumerate(kernelOutlines):
      if kernelOutline is None:
        continue

      with cpp.Function('{}::{}::{}'.format(self.NAMESPACE, name, executeName(index))):
        # This variant reads these constants laid out differently than the
        # member named after them holds them, so it reads them from the member
        # bound to its own arrangement -- under the name its code already
        # uses, which is why the declaration shadows the member on purpose.
        aliases = [(baseName, position) for baseName, position in alternates.get(index, {}).items()
                   if baseName in constants]
        if aliases:
          cpp('#if defined(__GNUC__)')
          cpp('#pragma GCC diagnostic push')
          cpp('#pragma GCC diagnostic ignored "-Wshadow"')
          cpp('#endif')
          for baseName, position in aliases:
            memberName = Tensor.splitBasename(baseName)[1]
            alternate = alternateName(baseName, position)
            family = self._families.get(baseName, {})
            withValues = [group for group, tensor in family.items() if tensor.values() is not None]
            if len(withValues) < len(family):
              # The pool holds nothing for a member without values, so the
              # caller fills that one in, and in the member named after the
              # family: this variant takes it from there and the rest from
              # its own arrangement.
              cpp('[[maybe_unused]] auto {0} = this->{0};'.format(memberName))
              for group in withValues:
                cpp('{0}({1}) = this->{2}({1});'.format(memberName, ','.join(map(str, group)), alternate))
            else:
              cpp('[[maybe_unused]] auto const& {} = this->{};'.format(memberName, alternate))
          cpp('#if defined(__GNUC__)')
          cpp('#pragma GCC diagnostic pop')
          cpp('#endif')
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
  TMP_SIZE = '_tmpMemSize'
  FLAGS_MEM = '_flags'
  DEV_FLAGS_MEM = '_dev_flags'
  DEV_POOL_MEM = '_dev_pool'

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

  def _cmpName(self, var):
    return '_cmp_' + self._tensorName(var)

  def _cmpViewName(self, var):
    return '_view_' + self._cmpName(var)

  def _emitUnpack(self, cpp, var, packedName, denseName, viewName):
    """Copies a tensor out of the layout the kernel stores it in into a dense
    buffer, through the view the initializer generates for it."""
    shape = var.memoryLayout().shape()
    cpp('{supportNS}::DenseTensorView<{dim},{datatype},{arch.uintTypename}> {viewName}({denseName}, {{{shape}}}, {{{start}}}, {{{stop}}});'.format(
        supportNS = SUPPORT_LIBRARY_NAMESPACE,
        dim=len(shape),
        datatype=var.datatype.ctype(),
        arch = self._arch,
        denseName=denseName,
        viewName=viewName,
        shape=', '.join([str(s) for s in shape]),
        start=', '.join([str(s.start) for s in var.memoryLayout().bbox()]),
        stop=', '.join([str(s.stop) for s in var.memoryLayout().bbox()])
      )
    )
    prefix = '{}::'.format(var.tensor.namespace) if var.tensor.namespace else ''
    cpp( '{prefix}{initNS}::{baseName}::{viewStruct}{groupTemplate}::{createFun}({packedName}).copyToView({viewName});'.format(
        initNS = InitializerGenerator.INIT_NAMESPACE,
        groupTemplate=self._groupTemplate(var.tensor),
        prefix=prefix,
        baseName=var.tensor.baseName(),
        packedName=packedName,
        viewName=viewName,
        viewStruct=InitializerGenerator.VIEW_STRUCT_NAME,
        createFun=InitializerGenerator.VIEW_FUN_NAME
      )
    )

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

  def generate(self, cpp, namespace, testName, kernelClass, cfg, target, gemm_cfg, testFramework, index=None, attrs=None):
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

    use_flags = device_test and attrs is not None and attrs.flags

    scalars = ScalarsSet().visit(cfg)
    variables = SortedGlobalsList().visit(cfg)
    # A by-value operand is a scalar wherever it turns up, as it is in the
    # kernel's signature: the kernel takes its value, and there is no buffer,
    # no init view and no device copy of it. The reference reads it through a
    # buffer of its own like any other operand, filled from that same value.
    byValue = [var for var in variables if var.tensor.isPassedByValue()]
    variables = [var for var in variables if not var.tensor.isPassedByValue()]
    scalars = {scalar.name(): scalar for scalar in scalars}
    for var in byValue:
      scalars.setdefault(var.tensor.name(), var.tensor)
    scalars = sorted(scalars.values(), key=str)
    conditions = self._conditionVariables(cfg)
    # A constant the kernel binds is read in the arrangement it was generated
    # against, which need not be the tensor's own, and only the pool holds it
    # that way. A condition is not one of them even where it carries values:
    # the test runs every case of it, and the kernel has to see each.
    guards = {str(var) for var in conditions}
    pooled = lambda var: (var.tensor.is_compute_constant() and not var.writable
                          and str(var) not in guards)
    bindPool = any(not var.tensor.isPassedAsArgument() or pooled(var) for var in variables)
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

       conditionBit = {str(var): i for i, var in enumerate(conditions)}
       for i,scalar in enumerate(scalars):
         bit = conditionBit.get(scalar.name())
         value = (f'(({self.CASE_VAR} >> {bit}) & 1) != 0' if bit is not None and cases > 1
                  else float(i+2))
         cpp('{} {} = {};'.format(scalar.getDatatype(self._arch).ctype(), self._tensorNameS(scalar), value))
       for var in byValue:
         cpp('{} {}[1] = {{{}}};'.format(var.datatype.ctype(), self._name(var), self._tensorNameS(var.tensor)))

       for var in variables:
         bit = conditionBit.get(str(var))
         factory.tensor(var.tensor, self._tensorName(var),
                        caseVar=self.CASE_VAR if (bit is not None and cases > 1) else None,
                        caseBit=bit)
         factory.temporary(self._name(var), var.memoryLayout().requiredReals(), var.datatype, iniZero=True)
         self._emitUnpack(cpp, var, self._tensorName(var), self._name(var),
                          self._viewName(var))
         cpp.emptyline()

       kernelTensorName = self._tensorName
       if device_test:
         writable = dict()
         for var in variables:
           bn = var.tensor.baseNameWithNamespace()
           writable[bn] = writable.get(bn, False) or var.writable

         kernelTensorName = lambda var: self._devTensorKernelArgument(var, writable)

         stream_new(self.STREAM)
         # what the kernel needs for one element, which is how many it runs
         # on; at least a byte, so that there is a block to hand it
         required = (f'{kernel_prefix}{OptimizedKernelGenerator.NAMESPACE}::{kernelClass}::'
                     f'{OptimizedKernelGenerator.TEMP_MAX_MEM_REQUIRED_NAME}')
         cpp(f'const std::size_t {self.TMP_SIZE} = {required} > 0 ? {required} : 1;')
         data_malloc(self.TMP_MEM, self.TMP_SIZE, f'{Datatype.I8.ctype()}*', self.STREAM)
         if use_flags:
           # A kernel that declares the batch flags reads one per element and
           # skips the ones that are off. The member defaults to a null
           # pointer, so leaving it unset is not a kernel that computes
           # everything -- it is a null dereference on the device, and the
           # context it kills takes every kernel launched after it with it.
           # The reference computes unconditionally, so the one element is on.
           cpp(f'unsigned {self.FLAGS_MEM}[1] = {{1}};')
           data_malloc(self.DEV_FLAGS_MEM, f'sizeof({self.FLAGS_MEM})', 'unsigned*', self.STREAM)
         for var in variables:
           data_malloc(self._devTensorName(var), f'sizeof({self._tensorName(var)})', f'{var.datatype.ctype()}*', self.STREAM)
           data_malloc(self._devPtrTensorName(var), f'sizeof({var.datatype.ctype()}*)', f'{var.datatype.ctype()}**', self.STREAM)
         for var in variables:
           data_memcpy(self._devTensorName(var), self._tensorName(var), f'sizeof({self._tensorName(var)})', self.STREAM)
           data_memcpy(self._devPtrTensorName(var), f'&{self._devTensorName(var)}', f'sizeof({var.datatype.ctype()}*)', self.STREAM)
         if use_flags:
           data_memcpy(self.DEV_FLAGS_MEM, self.FLAGS_MEM, f'sizeof({self.FLAGS_MEM})', self.STREAM)
         if bindPool:
           data_malloc(self.DEV_POOL_MEM, f'{PoolGenerator.BYTES_FUN_NAME}()', 'char*', self.STREAM)
           data_memcpy(self.DEV_POOL_MEM, f'{PoolGenerator.DATA_FUN_NAME}()',
                       f'{PoolGenerator.BYTES_FUN_NAME}()', self.STREAM)
         stream_wait(self.STREAM)
         cpp.emptyline()

       cpp( '{}{}::{} {};'.format(kernel_prefix, OptimizedKernelGenerator.NAMESPACE, kernelClass, self.KERNEL_VAR) )
       for var in scalars:
         cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.baseName(), self._groupIndex(var), self._tensorNameS(var)) )
       # Constants first, then the pool, then everything else. Which
       # arrangement of a constant the kernel reads, and whether a generator
       # that cannot write an immediate operand into its code reads it from
       # memory after all, are decided when the kernel is generated -- after
       # this test was. The test's buffer holds a constant as the tensor lays
       # itself out and the pool as the kernel reads it, so the pool has the
       # last word on the constants it binds; the buffer stays for those it
       # does not. bindGlobals hands out a family whole, and the pool holds
       # nothing for a member without values, so those members come after it.
       def assign(var):
         if not var.tensor.isPassedAsArgument():
           # The kernel has no member for it. The buffer above stays: the
           # reference implementation reads the tensor from memory, the
           # kernel spells it out, and the comparison between the two is
           # exactly what this test is for.
           return
         cpp( '{}.{}{} = {};'.format(self.KERNEL_VAR, var.tensor.baseName(), self._groupIndex(var.tensor), kernelTensorName(var)) )
       for var in variables:
         if pooled(var):
           assign(var)
       if bindPool:
         pool = (f'{PoolGenerator.POOL_STRUCT_NAME}::{PoolGenerator.CREATE_FUN_NAME}({self.DEV_POOL_MEM})'
                 if device_test else
                 f'{PoolGenerator.POOL_STRUCT_NAME}::{PoolGenerator.HOST_FUN_NAME}()')
         cpp(f'{self.KERNEL_VAR}.{OptimizedKernelGenerator.BIND_GLOBALS_NAME}({pool});')
       for var in variables:
         if not pooled(var):
           assign(var)

       if device_test:
         cpp( f'{self.KERNEL_VAR}.numElements = 1;' )
         cpp( f'{self.KERNEL_VAR}.linearAllocator.initialize({self.TMP_MEM}, {self.TMP_SIZE});' )
         cpp( f'{self.KERNEL_VAR}.streamPtr = reinterpret_cast<void*>({self.STREAM});' )
         if use_flags:
           cpp( f'{self.KERNEL_VAR}.{BatchedOperationsAux.FLAGS_NAME} = {self.DEV_FLAGS_MEM};' )

       cpp( '{}.{}();'.format(self.KERNEL_VAR, OptimizedKernelGenerator.EXECUTE_NAME + (str(index) if index is not None else '')) )
       cpp.emptyline()

       if device_test:
         stream_wait(self.STREAM)
         for var in variables:
           if var.writable:
             data_memcpy(self._tensorName(var), self._devTensorName(var), f'sizeof({self._tensorName(var)})', self.STREAM)
         stream_wait(self.STREAM)
         data_free(self.TMP_MEM, self.STREAM)
         if use_flags:
           data_free(self.DEV_FLAGS_MEM, self.STREAM)
         if bindPool:
           data_free(self.DEV_POOL_MEM, self.STREAM)
         for var in variables:
           data_free(self._devPtrTensorName(var), self.STREAM)
           data_free(self._devTensorName(var), self.STREAM)
         stream_wait(self.STREAM)
         stream_delete(self.STREAM)
         cpp.emptyline()

       super().generate(cpp, cfg, factory, None, gemm_cfg)

       for var in variables:
         if var.writable:
           layout = var.tensor.memoryLayout()
           if isinstance(layout, DenseMemoryLayout):
             factory.compare(var, Variable(self._tensorName(var), False, layout, datatype=var.datatype))
           else:
             # A tensor the kernel stores packed has no address for the entries
             # its pattern leaves out, so the comparison cannot walk it the way
             # it walks the reference. Unpack what the kernel wrote, the same
             # way the inputs are unpacked on the way in, and compare dense
             # against dense -- which also checks that the entries outside the
             # pattern are the zeros the reference computes for them.
             unpacked = self._cmpName(var)
             factory.temporary(unpacked, var.memoryLayout().requiredReals(), var.datatype, iniZero=True)
             self._emitUnpack(cpp, var, self._tensorName(var), unpacked,
                              self._cmpViewName(var))
             factory.compare(var, Variable(unpacked, False, var.memoryLayout(), datatype=var.datatype))

       factory.freeTmp()

class ViewArrayPool(object):
  """The index arrays the views need, each spelled once.

  Tensors that share a sparsity pattern need the same row indices. One array
  per tensor repeats them; naming them by content spells each once and lets
  the tensors refer to it. They stay constant expressions on both sides of
  that, which is what lets a lookup with constant indices fold to a single
  address instead of a search through the pattern. The bounds of a view are
  not among them (see `TensorView.OWN_ARRAYS`).

  Named after the text that gets emitted, the way the constant pool is: two
  arrays are the same array exactly when the generated source cannot tell
  them apart.
  """

  STRUCT_NAME = 'viewdata'
  NAME_SUFFIX_LENGTH = 8

  def __init__(self):
    self._arrays = collections.OrderedDict()
    self._names = dict()

  def intern(self, numberType, values, hint='array'):
    """Registers one array; reports the symbol it is spelled under and its length."""
    text = ViewArrayPool._text(values)
    key = hashlib.sha256('{}|{}'.format(numberType, text).encode('utf-8')).hexdigest()
    known = self._arrays.get(key)
    if known is None:
      known = (self._takeName(hint, key), numberType, text, ViewArrayPool._length(values))
      self._arrays[key] = known
    return known[0], known[3]

  def generate(self, cpp):
    if not self._arrays:
      return
    with cpp.Struct(self.STRUCT_NAME):
      for name, numberType, text, length in self._arrays.values():
        cpp('{} {} {}[{}] = {};'.format(DATA_MODIFIERS, numberType, name, length, text))
    cpp.emptyline()

  def __len__(self):
    return len(self._arrays)

  @staticmethod
  def _text(values):
    if isinstance(values, np.ndarray):
      values = values.flatten(order='K')
    return '{{{}}}'.format(', '.join([str(v) for v in values]))

  @staticmethod
  def _length(values):
    if isinstance(values, np.ndarray):
      return values.size
    return len(values)

  def _takeName(self, hint, key):
    stem = re.sub(r'\W', '_', hint)
    length = self.NAME_SUFFIX_LENGTH
    while True:
      name = '{}_{}'.format(stem, key[:length])
      if self._names.get(name, key) == key:
        self._names[name] = key
        return name
      length += self.NAME_SUFFIX_LENGTH
      if length > len(key):
        raise RuntimeError('Could not find a unique symbol for the view array {}.'.format(hint))


class InitializerGenerator(object):
  SHAPE_NAME = 'Shape'
  SIZE_NAME = 'Size'
  SIZE_FUN_NAME = 'size'
  INDEX_FUN_NAME = 'index'
  VALUES_BASENAME = 'Values'
  POOL_MEMBER_NAME = 'PoolMember'
  CONTAINER_CLASS_NAME = 'Container'
  CONTAINER_DATA_NAME = 'data'
  TENSOR_NAMESPACE = 'tensor'
  INIT_NAMESPACE = 'init'
  VIEW_STRUCT_NAME = 'view'
  VIEW_FUN_NAME = 'create'
  VIEW_TYPE_NAME = 'type'
  VIEW_TYPE_NAME_CONST = 'type_const'
  DESCRIPTOR_NAME = 'Descriptor'
  DESCRIPTORS_NAME = 'Descriptors'
  DESCRIPTOR_FUN_NAME = 'descriptor'
  TABLE_FUN_NAME = 'tensorTable'
  DESCRIPTOR_TYPE = '::{}::TensorDescriptor'.format(SUPPORT_LIBRARY_NAMESPACE)
  ENTRY_TYPE = '::{}::TensorEntry'.format(SUPPORT_LIBRARY_NAMESPACE)
  TABLE_TYPE = '::{}::TensorTable'.format(SUPPORT_LIBRARY_NAMESPACE)
  #: How the support library spells an element type.
  DESCRIPTOR_DATATYPES = {
    Datatype.BOOL: 'Bool',
    Datatype.I8: 'I8',
    Datatype.I16: 'I16',
    Datatype.I32: 'I32',
    Datatype.I64: 'I64',
    Datatype.F32: 'F32',
    Datatype.F64: 'F64',
    Datatype.F16: 'F16',
    Datatype.BF16: 'BF16',
    Datatype.F128: 'F128',
  }

  class TensorView(object):
    ARGUMENT_NAME = 'values'
    #: Whether the factory can be called from device code, which it can as
    #: long as it names nothing but its argument and literals. A sparse view
    #: is built on index arrays that are constexpr data of the host: device
    #: code may not refer to them, and a marked factory that does stops every
    #: CUDA translation unit including init.h, called or not.
    DEVICE_CALLABLE = True
    #: The arrays of `arrayData` each tensor spells as an array of its own
    #: instead of a reference to the shared one: the bounds, which device
    #: code reads. A static data member of reference type is no constant to
    #: SYCL, whose device code may not use it.
    OWN_ARRAYS = ()

    def __init__(self, datatype, arrayPool=None):
      self._datatype = datatype
      self._arrayPool = arrayPool

    def factoryModifiers(self):
      return STATIC_INLINE if self.DEVICE_CALLABLE else f'{STATIC} {INLINE}'

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

    def arrayData(self, memLayout):
      """The arrays this kind of view needs beside the values, as (suffix, values).

      Both the emission and the interning read the view's arrays from here,
      so that what gets spelled once and what gets named once cannot drift
      apart.
      """
      return []

    def arrays(self, cpp, memLayout, arch, namespace, index, numberType, declarationOnly):
      for suffix, values in self.arrayData(memLayout):
        cpp(self.formatArray(numberType, namespace + suffix + index, values,
                             declarationOnly, hint=suffix,
                             shared=suffix not in self.OWN_ARRAYS))

    #: The storage a descriptor names, and which of its arrays this kind of
    #: view fills, by the name of the member of the descriptor.
    STORAGE = None
    DESCRIPTOR_ARRAYS = {}

    def descriptorArrays(self, memLayout, index):
      """The members of a descriptor that point at the arrays of this view.

      Spelled as the names `arrays` emitted them under, so that a descriptor
      and a view read the same numbers. A view without an array of some kind
      -- a tensor without dimensions has no box -- points at nothing there.
      """
      emitted = {suffix for suffix, _ in self.arrayData(memLayout)}
      return {member: suffix + (index if index is not None else '')
              for member, suffix in self.DESCRIPTOR_ARRAYS.items() if suffix in emitted}

    def internArrays(self, memLayout, numberType):
      for suffix, values in self.arrayData(memLayout):
        if suffix not in self.OWN_ARRAYS:
          self._arrayPool.intern(numberType, values, suffix)

    def formatArray(self, numberType, name, values, declarationOnly, hint='array', shared=True):
      if declarationOnly:
        return ''
      if self._arrayPool is None or not shared:
        return f'{DATA_MODIFIERS} {numberType} {name}[] = {self.listToInitializerList(values)};'
      symbol, length = self._arrayPool.intern(numberType, values, hint)
      return '{} {} (&{})[{}] = {}::{};'.format(
        DATA_MODIFIERS, numberType, name, length, ViewArrayPool.STRUCT_NAME, symbol)

  class DenseTensorView(TensorView):
    START_NAME = 'Start'
    STOP_NAME = 'Stop'
    STRIDE_NAME = 'Stride'
    STORAGE = 'Dense'
    DESCRIPTOR_ARRAYS = {'start': START_NAME, 'stop': STOP_NAME, 'stride': STRIDE_NAME}
    OWN_ARRAYS = (START_NAME, STOP_NAME, STRIDE_NAME)

    def generate(self, cpp, memLayout, arch, index, const):
      cpp( 'return {}({}, {}, {}, {});'.format(
          self.typename(len(memLayout.shape()), arch, const),
          self.ARGUMENT_NAME,
          self.listToInitializerList(memLayout.shape()),
          self.listToInitializerList([r.start for r in memLayout.bbox()]),
          self.listToInitializerList([r.stop for r in memLayout.bbox()])
        )
      )
    def arrayData(self, memLayout):
      if not memLayout.shape():
        return []
      # The strides as the layout has them. They follow from the box as long
      # as nobody gave the layout strides of its own, which is what the view
      # assumes; a descriptor says what is true either way.
      return [(self.START_NAME, [r.start for r in memLayout.bbox()]),
              (self.STOP_NAME, [r.stop for r in memLayout.bbox()]),
              (self.STRIDE_NAME, list(memLayout.stride()))]

  class CSCMatrixView(TensorView):
    ROWIND_NAME = 'RowInd'
    COLPTR_NAME = 'ColPtr'
    DEVICE_CALLABLE = False
    STORAGE = 'CSC'
    DESCRIPTOR_ARRAYS = {'rowIndex': ROWIND_NAME, 'columnPointer': COLPTR_NAME}

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
    def arrayData(self, memLayout):
      return [(self.ROWIND_NAME, memLayout.rowIndex()),
              (self.COLPTR_NAME, memLayout.colPointer())]

  class PatternTensorView(TensorView):
    PATTERN_NAME = 'Pattern'
    START_NAME = 'Start'
    STOP_NAME = 'Stop'
    STRIDE_NAME = 'Stride'
    DEVICE_CALLABLE = False
    STORAGE = 'Pattern'
    DESCRIPTOR_ARRAYS = {'start': START_NAME, 'stop': STOP_NAME, 'stride': STRIDE_NAME,
                         'pattern': PATTERN_NAME}
    OWN_ARRAYS = (START_NAME, STOP_NAME, STRIDE_NAME)

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

    def arrayData(self, memLayout):
      # The pattern covers a box of its own, which reaches past the shape where
      # the layout is padded; its entries lie in it by columns.
      extent = memLayout.pattern().shape
      stride = [1]
      for size in extent[:-1]:
        stride.append(stride[-1] * size)
      return [(self.START_NAME, [0] * len(extent)),
              (self.STOP_NAME, list(extent)),
              (self.STRIDE_NAME, stride),
              (self.PATTERN_NAME, memLayout.pattern())]

  #: What the pool needs to know about one tensor group: how large the group
  #: is, which element type its entries were stored as, and where each member
  #: of it ended up.
  PoolEntry = collections.namedtuple(
    'PoolEntry', ['baseName', 'groupSize', 'datatype', 'symbols', 'arrangement'])

  def __init__(self, arch, tensors, scalars, inMemory=frozenset(), namespace=''):
    self._arch = arch
    #: The namespace everything is generated into, for the table, which names
    #: the tensors of every namespace from one place.
    self._namespace = namespace
    #: Immediate tensors some kernel reads from memory after all, by name.
    #: They get a pool entry like any other constant.
    self._inMemory = inMemory
    self._numberType = f'{self._arch.uintTypename} const'
    self._viewArrays = None
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
    return memLayoutMap[type(memoryLayout).__name__](tensor.getDatatype(self._arch),
                                                     self._viewArrays)

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
                header('assert({} < {});'.format(indexFun(groupSizeToStride(groupSize)),
                                                 reduce(operator.mul, groupSize)))
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
                with header.Function('operator()', typedArgs, '{} T&'.format(HOSTDEVICE_INLINE)):
                  header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
                with header.Function('operator()', typedArgs, '{} T const&'.format(HOSTDEVICE_INLINE), const=True):
                  header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
    for namespace, scalar_dict in self.iterate_collect_scalar():
      if len(scalar_dict) == 0:
        continue
      with header.Namespace(namespace), header.Namespace(self.TENSOR_NAMESPACE):
        for (baseName, baseNameWithoutNamespace), scalars in scalar_dict.items():
          with header.Struct(baseNameWithoutNamespace):
            groupSize = self._groupSizeScalar[baseName]
            args = ndargs(len(groupSize))
            typedArgs = typedNdArgs(len(groupSize), self._arch.uintTypename)
            if len(groupSize) > 0:
              with header.Function(self.INDEX_FUN_NAME, typedArgs, returnType):
                header('assert({} < {});'.format(indexFun(groupSizeToStride(groupSize)),
                                                 reduce(operator.mul, groupSize)))
                header('return {};'.format(indexFun(groupSizeToStride(groupSize))))
            if len(groupSize) > 0:
              header('template<typename T>')
              with header.Struct(self.CONTAINER_CLASS_NAME):
                header('T {}[{}];'.format(self.CONTAINER_DATA_NAME, reduce(operator.mul, groupSize)))
                with header.Function(self.CONTAINER_CLASS_NAME, '', ''):
                  pass
                with header.Function('operator()', typedArgs, '{} T&'.format(HOSTDEVICE_INLINE)):
                  header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
                with header.Function('operator()', typedArgs, '{} T const&'.format(HOSTDEVICE_INLINE), const=True):
                  header('return {}[{}({})];'.format(self.CONTAINER_DATA_NAME, self.INDEX_FUN_NAME, ', '.join(args)))

  def generateTensorsCpp(self, cpp):
    # Shape and Size are constexpr static members, and a constexpr static
    # member is implicitly inline, so it needs no definition outside the
    # class. Writing one anyway is deprecated and both GCC and clang say so.
    pass

  def collectPool(self, dataCache, arrangements=None):
    """Registers every constant tensor in `dataCache` and reports the symbols.

    The result maps a tensor and one arrangement of it -- keyed by the pair --
    to its group size, its element type and the pool symbol each group ended
    up under. A tensor read in two arrangements is stored twice, because the
    two are two different sequences of numbers and an address into one is not
    an address into the other. Where `arrangements` says nothing about a
    tensor, it is stored as it lays itself out.

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
      read = (arrangements or {}).get(baseName) \
        or {None: Arrangement.of(tensors)}
      for _, arrangement in read.items():
        entry = self._poolEntry(dataCache, baseName, tensors, groupSize, stride, arrangement)
        if entry is not None:
          pool[(baseName, arrangement.tag())] = entry
    self._pool = pool
    return pool

  def _poolEntry(self, dataCache, baseName, tensors, groupSize, stride, arrangement):
    """One family in one arrangement, registered member by member.

    A member is stored as the arrangement holds it, and as it lays itself out
    where the arrangement says nothing about it -- a member the kernels never
    read still has an entry, because the family is handed out whole.
    """
    symbols = collections.OrderedDict()
    datatype = None
    for group, tensor in tensors.items():
        values = tensor.values()
        if values is None:
          continue
        if not tensor.isPassedAsArgument() \
           and tensor.nameWithNamespace() not in self._inMemory:
          # Its data is in every kernel that reads it, so the pool has nothing
          # to hold and nobody to hand an address to. `init::X::Values` is
          # still written, from the same numbers, for whoever computes with
          # the tensor on the host.
          continue
        memLayout = arrangement.layoutOf(group, tensor.memoryLayout())
        index = address(group, stride)
        hint = baseName if len(group) == 0 else '{}_{}'.format(baseName, index)
        groupDatatype = tensor.getDatatype(self._arch)
        if datatype is None:
          datatype = groupDatatype
        elif datatype != groupDatatype:
          # One member of a group, one element type: the group is handed out
          # as a Container<T>, and there is no T that fits both.
          raise ValueError('Mixed datatypes are not allowed within a tensor group. '
                           '({} and {} for {}.)'.format(datatype, groupDatatype, baseName))
        # The floor is what keeps entries at a common stride; the layout is
        # asked on top of it, because a target whose cache line is wider than
        # the floor -- a64fx -- would otherwise get less than its aligned
        # loads need.
        layoutAlignment = self._arch.cacheline if memLayout.alignedStride() else 1
        alignment = max(POOL_ALIGNMENT, layoutAlignment)
        # A prepared image is stored as it was handed over; anything this
        # side laid out is packed from the tensor's own numbers.
        image = (memLayout.imageFor(tensor.nameWithNamespace(), tensor.name())
                 if isinstance(memLayout, PreparedImage)
                 else memLayout.pack(values))
        symbols[group] = dataCache.add(hint,
                                       [groupDatatype.literal(value) for value in image],
                                       groupDatatype.ctype(),
                                       alignment)
    if not symbols:
      return None
    return self.PoolEntry(baseName, groupSize, datatype, symbols, arrangement)

  def poolSymbol(self, baseName, group):
    """Pool entry holding exactly what init would otherwise print, if there is one.

    There is one as long as a family is held the way it lays itself out, which
    is why the initialiser can bind a reference instead of materialising the
    values a second time. Once the kernels read it some other way, only an
    entry that matches what init promises can be bound like that, and the rest
    of them answer None here and get their own array.
    """
    tensors = self._collect.get(baseName)
    if not tensors or group not in tensors:
      return None
    entry = self._pool.get((baseName, Arrangement.of(tensors).tag()))
    return entry.symbols.get(group) if entry is not None else None

  def generateInitH(self, header):
    for namespace, tensor_dict in self.iterate_collect():
      # The shared arrays have to stand before the structs that name them, so
      # which ones there are is settled before anything is written.
      self._viewArrays = ViewArrayPool()
      for _, tensors in tensor_dict.items():
        for tensor in tensors.values():
          self._tensorViewGenerator(tensor).internArrays(tensor.memoryLayout(),
                                                         self._numberType)
      with header.Namespace(namespace), header.Namespace(self.INIT_NAMESPACE):
        self._viewArrays.generate(header)
        for (base_name, base_name_without_namespace), tensors in tensor_dict.items():
          self._init(header, base_name, base_name_without_namespace, '', tensors, False)
    self._viewArrays = None
    for namespace, scalar_dict in self.iterate_collect_scalar():
      if len(scalar_dict) == 0:
        continue
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
              # No alignment attribute either: the pool aligns the entry to
              # at least POOL_ALIGNMENT, which is not less than this would
              # have asked for.
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

        # Where a pool holds the family as it lays itself out -- the numbers
        # `Values` refers to. The member of `Pool` is named after the
        # arrangement, which is a hash; code that reads the constants itself
        # rather than through a kernel's bindGlobals, from a pool that lives
        # on a device, say, names it through this instead: `pool.*PoolMember`.
        own = self._pool.get((baseName, Arrangement.of(tensors).tag()))
        if own is not None:
          cpp('{} {} auto {} = &{}::{};'.format(CONSTEXPR, STATIC, self.POOL_MEMBER_NAME,
                                                PoolGenerator.POOL_STRUCT_NAME,
                                                PoolGenerator.memberName(baseName, own.arrangement)))

        self._descriptors(cpp, baseName, tensors, groupSize)

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
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(tv.factoryModifiers(), self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, None, False)
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgsConst, returnType='{} {}'.format(tv.factoryModifiers(), self.VIEW_TYPE_NAME_CONST)):
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
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgs, returnType='{} {}'.format(tv.factoryModifiers(), self.VIEW_TYPE_NAME)):
              tv.generate(cpp, ml, self._arch, index(group), False)
            with cpp.Function(self.VIEW_FUN_NAME, arguments=viewArgsConst, returnType='{} {}'.format(tv.factoryModifiers(), self.VIEW_TYPE_NAME_CONST)):
              tv.generate(cpp, ml, self._arch, index(group), True)

  def _descriptors(self, cpp, baseName, tensors, groupSize):
    """Where the values of each member are, as data: a descriptor per member.

    It points at the arrays the views are built from, so that a descriptor
    and a view cannot tell two different stories about one tensor.
    `Descriptors` holds them in the order `tensor::X::index` gives -- a hole
    of the family is null there -- and `descriptor` picks one out.
    """
    stride = groupSizeToStride(groupSize)
    members = dict()
    for group, tensor in tensors.items():
      index = str(address(group, stride)) if len(group) > 0 else ''
      at = '[{}]'.format(index) if len(group) > 0 else ''
      memLayout = tensor.memoryLayout()
      view = self._tensorViewGenerator(tensor)
      arrays = view.descriptorArrays(memLayout, index)
      rank = len(memLayout.shape())
      datatype = tensor.getDatatype(self._arch)
      alignmentArch = memLayout.alignmentArch()
      aligned = memLayout.alignedStride() and alignmentArch is not None
      fields = ['::{}::Datatype::{}'.format(SUPPORT_LIBRARY_NAMESPACE, self.DESCRIPTOR_DATATYPES[datatype]),
                '::{}::Storage::{}'.format(SUPPORT_LIBRARY_NAMESPACE, view.STORAGE),
                str(rank),
                self.SHAPE_NAME + at if rank > 0 else 'nullptr']
      fields += [arrays.get(member, 'nullptr')
                 for member in ('start', 'stop', 'stride', 'rowIndex', 'columnPointer', 'pattern')]
      fields += [self.SIZE_NAME + at,
                 str(alignmentArch.alignment if aligned else datatype.size())]
      name = self.DESCRIPTOR_NAME + index
      cpp('{} {} {}{{{}}};'.format(DATA_MODIFIERS, self.DESCRIPTOR_TYPE, name, ', '.join(fields)))
      members[group] = name

    count = reduce(operator.mul, groupSize, 1)
    table = ['nullptr'] * count
    for group, name in members.items():
      table[address(group, stride)] = '&' + name
    cpp('{} {} const* {}[] = {{{}}};'.format(DATA_MODIFIERS, self.DESCRIPTOR_TYPE,
                                             self.DESCRIPTORS_NAME, ', '.join(table)))
    args = ndargs(len(groupSize))
    with cpp.Function(self.DESCRIPTOR_FUN_NAME,
                      typedNdArgs(len(groupSize), self._arch.uintTypename),
                      '{} {} {} const*'.format(CONSTEXPR, STATIC, self.DESCRIPTOR_TYPE)):
      if len(groupSize) > 0:
        cpp('return {}[{}({})];'.format(self.DESCRIPTORS_NAME, self.INDEX_FUN_NAME, ', '.join(args)))
      else:
        cpp('return &{};'.format(self.DESCRIPTOR_NAME))

  def tableEntries(self):
    """What the table lists: name and number of group indices, ordered by name."""
    return [(baseName, len(self._groupSize[baseName])) for baseName in sorted(self._collect)]

  def generateTableH(self, header):
    if self.TABLE_FUN_NAME in self._collect:
      raise ValueError('The tensor {0} has the name of the function that lists every tensor, '
                       '{1}::{0}(). Rename the tensor.'.format(self.TABLE_FUN_NAME, self.INIT_NAMESPACE))
    with header.Namespace(self.INIT_NAMESPACE):
      header('//! Every tensor of this generation, ordered by name.')
      header.functionDeclaration(self.TABLE_FUN_NAME, '', '{} const&'.format(self.TABLE_TYPE))

  def generateTableCpp(self, cpp):
    """The table of every tensor, for code that learns at run time which one it wants.

    Ordered by name, the namespace included, which is the order a lookup in
    it relies on.
    """
    root = '::{}'.format(self._namespace) if self._namespace else ''
    with cpp.Namespace(self.INIT_NAMESPACE):
      with cpp.Function(self.TABLE_FUN_NAME, '', '{} const&'.format(self.TABLE_TYPE)):
        entries = []
        for position, (baseName, rank) in enumerate(self.tableEntries()):
          prefix, name = Tensor.splitBasename(baseName)
          qualified = '{}::{}{}::{}::{}'.format(root, prefix, self.INIT_NAMESPACE, name, self.DESCRIPTORS_NAME)
          groupSize = 'nullptr'
          if rank > 0:
            groupSize = 'GroupSize{}'.format(position)
            cpp('{} {} {}[] = {{{}}};'.format(DATA_MODIFIERS, self._numberType, groupSize,
                                              ', '.join(str(size) for size in self._groupSize[baseName])))
          entries.append('{{"{}", {}, {}, {}}}'.format(baseName, rank, groupSize, qualified))
        if entries:
          with Block(cpp, '{} {} Entries[] ='.format(DATA_MODIFIERS, self.ENTRY_TYPE), foot=';'):
            for entry in entries:
              cpp(entry + ',')
          cpp('{} {} Table{{Entries, {}}};'.format(DATA_MODIFIERS, self.TABLE_TYPE, len(entries)))
        else:
          cpp('{} {} Table{{nullptr, 0}};'.format(DATA_MODIFIERS, self.TABLE_TYPE))
        cpp('return Table;')

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
    self._reservations = dataCache.reservations()
    self._members = self.assignMembers(pool, self._reservations)

  @classmethod
  def memberName(cls, baseNameWithNamespace, arrangement=None):
    """Name an arrangement of a tensor family goes by inside `Pool`.

    Flattened rather than nested by namespace: a kernel may read constants
    from several namespaces at once, so one flat table is the only shape that
    lets it bind all of them against a single object.

    The arrangement is part of the name, because one family may be held in
    more than one of them: a kernel generated against a different padding or
    a different sparsity reads a different array, and binding both to one
    member would hand one of them a stride nobody gave it. Both sides work it
    out from the tensors alone, each where it stands, so neither has to wait
    for the other.
    """
    flat = baseNameWithNamespace.replace('::', '_')
    if arrangement is None:
      return flat
    return '{}_{}'.format(flat, arrangement.tag())

  #: A short name for how one tensor is held; see `arrangement.layoutTag`.
  arrangementTag = staticmethod(layoutTag)

  def imageAlignment(self):
    """Alignment the image is declared with.

    The floor, or the strictest entry where that asks for more. alignas on a
    class states a minimum and a member asking for more raises the class past
    it, so declaring the floor alone would leave pool.h saying 128 for a type
    that is actually aligned to 256 -- and would have the declaration weaken
    an alignment that alignas is not meant to weaken. Naming the maximum
    keeps the declared number, alignof() and poolAlignment() one number
    instead of two.
    """
    return max([POOL_ALIGNMENT] + [entry.alignment() for entry in self._dataCache.entries()])

  @classmethod
  def assignMembers(cls, pool, reservations=()):
    """Member name per tensor, with the collisions flattening can cause refused.

    Two tensors that differ only in where the namespace separator sat --
    `a::b` and `a_b` -- flatten to the same identifier. Declaring the member
    twice would not compile, and were the name to come from a hint instead
    one of them would quietly write into the other's slot. Say which two, and
    let the caller rename one. A reservation names its own member and is held
    to the same rule, against the tensors and against the other reservations.
    """
    members = collections.OrderedDict()
    taken = dict()
    for key in pool:
      # A bare listing of names carries no arrangement, and a member named
      # from the name alone is the right answer for it.
      entry = pool.get(key) if hasattr(pool, 'get') else None
      baseName = key if entry is None else entry.baseName
      member = cls.memberName(baseName, None if entry is None else entry.arrangement)
      if member in taken:
        raise ValueError('The tensors {} and {} share the pool member {}. '
                         'Rename one of them.'.format(taken[member], baseName, member))
      taken[member] = baseName
      members[key] = member
    for reservation in reservations:
      member = reservation.name()
      if member in taken:
        raise ValueError('The reserved pool entry {} shares the pool member {} with {}. '
                         'Rename one of them.'.format(reservation.name(), member, taken[member]))
      taken[member] = reservation.name()
    return members

  @staticmethod
  def _elementPtrType(datatype):
    return '{} const*'.format(datatype.ctype())

  def _memberType(self, entry):
    elementPtr = self._elementPtrType(entry.datatype)
    if len(entry.groupSize) == 0:
      return elementPtr
    prefix, name = Tensor.splitBasename(entry.baseName)
    return '{}{}::{}::{}<{}>'.format(prefix,
                                     InitializerGenerator.TENSOR_NAMESPACE,
                                     name,
                                     InitializerGenerator.CONTAINER_CLASS_NAME,
                                     elementPtr)

  def _storageType(self):
    return '{}::{}'.format(self.STORAGE_NAMESPACE, self.STORAGE_STRUCT_NAME)

  def generateH(self, header):
    with header.Namespace(self.STORAGE_NAMESPACE):
      with header.Struct('alignas({}) {}'.format(self.imageAlignment(), self.STORAGE_STRUCT_NAME)):
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
      for key, entry in self._pool.items():
        header('{} {}{{}};'.format(self._memberType(entry), self._members[key]))
      for reservation in self._reservations:
        header('{} const* {}{{}};'.format(reservation.entry().typename(), reservation.name()))
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
      for key, entry in self._pool.items():
        member = self._members[key]
        stride = groupSizeToStride(entry.groupSize)
        for group, symbol in entry.symbols.items():
          target = member if len(group) == 0 else '{}.{}[{}]'.format(
            member, InitializerGenerator.CONTAINER_DATA_NAME, address(group, stride))
          cpp('result.{} = reinterpret_cast<{}>(origin + offsetof({}, {}));'.format(
            target, self._elementPtrType(entry.datatype), self._storageType(), symbol))
      for reservation in self._reservations:
        cpp('result.{} = reinterpret_cast<{} const*>(origin + offsetof({}, {}));'.format(
          reservation.name(), reservation.entry().typename(), self._storageType(),
          reservation.entry().name()))
      cpp('return result;')
    cpp.emptyline()

    with cpp.Function('{}::{}'.format(self.POOL_STRUCT_NAME, self.HOST_FUN_NAME), '', returnType):
      cpp('return {}({}());'.format(self.CREATE_FUN_NAME, self.DATA_FUN_NAME))
    cpp.emptyline()

    with cpp.Function(self.BYTES_FUN_NAME, '', self.SIZE_TYPE):
      cpp('return sizeof({});'.format(self._storageType()))
    cpp.emptyline()

    with cpp.Function(self.ALIGN_FUN_NAME, '', self.SIZE_TYPE):
      # Read off the image rather than repeated from the number the struct was
      # declared with, so that the two cannot drift apart -- a consumer that
      # allocated for less than the image wants would be one entry short.
      cpp('return alignof({});'.format(self._storageType()))
    cpp.emptyline()

    with cpp.Function(self.DATA_FUN_NAME, '', 'void const*'):
      cpp('return &{}::{};'.format(self.STORAGE_NAMESPACE, self.STORAGE_VAR_NAME))
