from ..common import TensorDescription, IndexedTensorDescription, BatchedOperationsAux
from ...ast.indices import BoundingBox
from ..cache import RoutineGenerator, GpuRoutineGenerator
from kernelforge.interface import YatetoInterface as yi
from kernelforge.generators.descriptions import GemmDescr
from kernelforge.common.basic_types import Addressing, FloatingPointType, DataFlowDirection
from kernelforge.common.context import Context
from kernelforge.common.aux import generate_tmp_matrix
from kernelforge.generators.generator import Generator as KernelForgeGenerator


class FusedGemms:
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
    self._batch_aux = BatchedOperationsAux(self._arch.typename)
    self._cache = {}
    self._tmp_matrices = {}

  def add_operation(self, operation):
    pass

  def generate(self, cpp, routineCache, cfg):
    self._tmp_matrices = {}
    self._cache = {}
    gemm_list = []
    flops = 0
    for item in self._descr:
      node, args, add, scalar = item
      res, op1, op2 = args

      self._cache_matrices(node, res, op1, op2)
      can_be_aligned = self._can_be_aligned(node, res, op1)
      gemm_list.append(GemmDescr(trans_a=node.transA(),
                                 trans_b=node.transB(),
                                 a=self._cache[op1.name],
                                 b=self._cache[op2.name],
                                 c=self._cache[res.name],
                                 alpha=scalar,
                                 beta=1.0 if add else 0.0,
                                 strict_match=False,
                                 prefer_align=can_be_aligned))
      # flops += gemm_list[-1].compute_flops()

    context = Context(arch=self._arch.name,
                      backend=self._arch.backend,
                      fp_type=FloatingPointType.str2enum(self._arch.typename))

    chainforge_generator = KernelForgeGenerator(gemm_list, context)
    chainforge_generator.register()

    cpp(f'{self._gen_call_site(chainforge_generator)}')
    routine_name = chainforge_generator.get_base_name()
    routineCache.addRoutine(routine_name, KernelForgeWriter(chainforge_generator, context.get_vm().get_headers()))
    return flops

  def _can_be_aligned(self, node, res, op1):
    res_tensor = IndexedTensorDescription.fromNode(res, node)
    op1_tensor = IndexedTensorDescription.fromNode(op1, node.leftTerm())

    aligned_res = res_tensor.memoryLayout.alignedStride()
    aligned_op1 = not node.transA() and op1_tensor.memoryLayout.alignedStride()
    return aligned_res and aligned_op1

  def _cache_matrices(self, node, res, op1, op2):
    res_tensor = IndexedTensorDescription.fromNode(res, node)
    op1_tensor = IndexedTensorDescription.fromNode(op1, node.leftTerm())
    op2_tensor = IndexedTensorDescription.fromNode(op2, node.rightTerm())
    m, n, k = FusedGemms._get_gemm_mnk(op1=op1_tensor,
                                       trans_op1=node.transA(),
                                       op2=op2_tensor,
                                       trans_op2=node.transB())

    can_be_aligned = self._can_be_aligned(node, res, op1)
    if can_be_aligned:
      aligned_m = m.aligned(self._arch)
      m.stop = aligned_m.stop

    matrix = self._get_chainforge_matrix(tensor=op1_tensor,
                                         tensor_variable=op1,
                                         range=(m, k))

    if not (op1.name in self._cache and matrix.is_same(self._cache[op1.name])):
      self._cache[op1.name] = matrix

    matrix = self._get_chainforge_matrix(tensor=op2_tensor,
                                         tensor_variable=op2,
                                         range=(k, n))

    if not (op2.name in self._cache and matrix.is_same(self._cache[op2.name])):
      self._cache[op2.name] = matrix

    if res.is_temporary:
      self._cache[res.name] = self._gen_tmp_matix(op1, op2, node, res.name)
    else:
      matrix = self._get_chainforge_matrix(tensor=res_tensor,
                                           tensor_variable=res,
                                           range=(m, n))

      if not (res.name in self._cache and matrix.is_same(self._cache[res.name])):
        self._cache[res.name] = matrix

  def _get_chainforge_matrix(self, tensor, tensor_variable, range):
    addr_mode = self._batch_aux.deduce_addresing(tensor)
    if tensor_variable.is_temporary:
      if not tensor_variable.name in self._tmp_matrices:
        raise RuntimeError(f'expected tmp. tensor {tensor_variable.name} to be cached '
                           f'while code generation for fused-gemms')
      else:
        return self._tmp_matrices[tensor_variable.name]

    return yi.gen_matrix(range,
                               tensor.memoryLayout.bbox(),
                               addressing=addr_mode,
                               name=tensor_variable.name,
                               is_tmp=tensor_variable.is_temporary,
                               transpose=False,
                               pattern = None,
                               values = None)

  def _gen_tmp_matix(self, op1, op2, res_node, res_name):
    tmp_matrix = generate_tmp_matrix(op1=self._cache[op1.name],
                                     op2=self._cache[op2.name],
                                     trans_a=res_node.transA(),
                                     trans_b=res_node.transB())
    self._tmp_matrices[res_name] = tmp_matrix
    return tmp_matrix

  def _gen_call_site(self, generator):
    mat_name_map = {}
    offset_name_map = {}
    for name, matrix in self._cache.items():
      if matrix.direction == DataFlowDirection.SOURCE:
        ptr_type = f'const {self._arch.typename}{Addressing.addr2ptr_type(matrix.addressing)}'
        mat_name_map[name] = f'const_cast<{ptr_type}>({name})'
      else:
        mat_name_map[name] = name

      if matrix.is_tmp or matrix.addressing == Addressing.NONE:
        offset_name_map[name] = '0'
      else:
        offset_name_map[name] = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{name}'

    beta = 1.0 if self._descr.add[-1] else 0.0
    alpha = self._descr.scalar[-1]
    return generator.generate_call_site(mat_name_map,
                                        offset_name_map,
                                        BatchedOperationsAux.NUM_ELEMENTS_NAME,
                                        BatchedOperationsAux.FLAGS_NAME,
                                        BatchedOperationsAux.STREAM_PTR_NAME)

  @classmethod
  def _get_gemm_mnk(cls, op1, trans_op1, op2, trans_op2):
    bbox_op1 = BoundingBox.fromSpp(op1.eqspp)
    bbox_op2 = BoundingBox.fromSpp(op2.eqspp)
    k_op1 = 1 if not trans_op1 else 0
    k_op2 = 0 if not trans_op2 else 1

    k = bbox_op1[k_op1] & bbox_op2[k_op2]
    m = bbox_op1[1 - k_op1]
    n = bbox_op2[1 - k_op2]
    return m, n, k


class KernelForgeWriter(GpuRoutineGenerator):
  def __init__(self, chainforge_generator, headers):
    self._headers = list(headers) + list(chainforge_generator.get_helper_headers())
    self._generator = chainforge_generator
    self._basename = self._generator.get_base_name()

  def __eq__(self, other):
    if isinstance(other, KernelForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.includes(self._headers)

  def __call__(self, routineName, fileName):
    self._generator.generate()
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, 'a') as file:
      file.write(kernel)
      file.write(launcher)

    return self._generator.get_header()
