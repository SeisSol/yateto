from ..common import BatchedOperationsAux, KernelAttributes
from ...ast.indices import BoundingBox
from ..cache import RoutineGenerator, GpuRoutineGenerator
from chainforge.interfaces import YatetoInterface as yi
from chainforge.common import GemmDescr, Addressing, FloatingPointType, DataFlowDirection
from chainforge.common import Context, generate_tmp_matrix
from chainforge.backend.generator import Generator as ChainForgeGenerator


class FusedGemms:
  def __init__(self, arch, descr, attrs=None):
    self._arch = arch
    self._descr = descr
    self._attrs = attrs if attrs is not None else KernelAttributes()
    self._datatype = self._descr.datatype
    self._batch_aux = BatchedOperationsAux()
    self._cache = {}
    self._tmp_matrices = {}

  def generate(self, cpp, routineCache):
    self._tmp_matrices = {}
    self._cache = {}
    gemm_list = []
    flops = 0
    for statement in self._descr:
      result, left, right = statement.result, *statement.terms

      self._cache_matrices(statement)
      gemm_list.append(GemmDescr(trans_a=statement.transA,
                                 trans_b=statement.transB,
                                 a=self._cache[left.name],
                                 b=self._cache[right.name],
                                 c=self._cache[result.name],
                                 alpha=statement.alpha,
                                 beta=1.0 if statement.add else 0.0,
                                 strict_match=False,
                                 prefer_align=self._can_be_aligned(statement)))
      flops += gemm_list[-1].compute_flops()

    context = Context(arch=self._arch.name,
                      backend=self._arch.backend,
                      fp_type=FloatingPointType.str2enum(self._datatype.ctype()))

    chainforge_generator = ChainForgeGenerator(gemm_list, context)
    chainforge_generator.register()

    cpp(f'{self._gen_call_site(chainforge_generator)}')
    routine_name = chainforge_generator.get_base_name()
    routineCache.addRoutine(routine_name, ChainForgeWriter(chainforge_generator))
    return flops

  def _can_be_aligned(self, statement):
    aligned_res = statement.result.memoryLayout.alignedStride()
    aligned_op1 = not statement.transA \
                  and statement.terms[0].memoryLayout.alignedStride()
    return aligned_res and aligned_op1

  def _cache_matrices(self, statement):
    result, left, right = statement.result, *statement.terms
    m, n, k = FusedGemms._get_gemm_mnk(op1=left,
                                       trans_op1=statement.transA,
                                       op2=right,
                                       trans_op2=statement.transB)

    if self._can_be_aligned(statement):
      aligned_m = m.aligned(self._arch)
      m.stop = aligned_m.stop

    matrix = self._get_chainforge_matrix(tensor=left, range=(m, k))

    if not (left.name in self._cache and matrix.is_same(self._cache[left.name])):
      self._cache[left.name] = matrix

    matrix = self._get_chainforge_matrix(tensor=right, range=(k, n))

    if not (right.name in self._cache and matrix.is_same(self._cache[right.name])):
      self._cache[right.name] = matrix

    if result.is_temporary:
      self._cache[result.name] = self._gen_tmp_matix(statement)
    else:
      matrix = self._get_chainforge_matrix(tensor=result, range=(m, n))

      if not (result.name in self._cache and matrix.is_same(self._cache[result.name])):
        self._cache[result.name] = matrix

  def _get_chainforge_matrix(self, tensor, range):
    addr_mode = self._batch_aux.deduce_addresing(tensor)
    if tensor.is_temporary:
      if not tensor.name in self._tmp_matrices:
        raise RuntimeError(f'expected tmp. tensor {tensor.name} to be cached '
                           f'while code generation for fused-gemms')
      else:
        return self._tmp_matrices[tensor.name]

    return yi.gen_dense_matrix(range,
                               tensor.memoryLayout.bbox(),
                               addressing=Addressing.str2addr(addr_mode),
                               name=tensor.name,
                               is_tmp=tensor.is_temporary)

  def _gen_tmp_matix(self, statement):
    left, right = statement.terms
    tmp_matrix = generate_tmp_matrix(op1=self._cache[left.name],
                                     op2=self._cache[right.name],
                                     trans_op1=statement.transA,
                                     trans_op2=statement.transB)
    self._tmp_matrices[statement.result.name] = tmp_matrix
    return tmp_matrix

  def _gen_call_site(self, generator):
    mat_name_map = {}
    offset_name_map = {}
    for name, matrix in self._cache.items():
      if matrix.direction == DataFlowDirection.SOURCE:
        ptr_type = f'{self._datatype.ctype()} {Addressing.addr2ptr_type(matrix.addressing)}'
        mat_name_map[name] = f'const_cast<{ptr_type}>({name})'
      else:
        mat_name_map[name] = name

      if matrix.is_tmp or matrix.addressing == Addressing.NONE:
        offset_name_map[name] = '0'
      else:
        offset_name_map[name] = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{name}'

    beta = 1.0 if self._descr.last.add else 0.0
    alpha = self._descr.last.alpha
    return generator.generate_call_site(mat_name_map,
                                        offset_name_map,
                                        alpha,
                                        beta,
                                        BatchedOperationsAux.NUM_ELEMENTS_NAME,
                                        BatchedOperationsAux.flags_arg(self._attrs),
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


class ChainForgeWriter(GpuRoutineGenerator):
  def __init__(self, chainforge_generator):
    self._generator = chainforge_generator
    self._basename = self._generator.get_base_name()

  def __eq__(self, other):
    if isinstance(other, ChainForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    #cpp.include('chainforge_aux.h')
    pass

  def __call__(self, routineName, fileName):
    self._generator.generate()
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, 'a') as file:
      file.write(kernel)
      file.write(launcher)

    return self._generator.get_header()
