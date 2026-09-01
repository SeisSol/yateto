from typing import List
from abc import ABC, abstractmethod
from enum import IntEnum
import operator

class Sparsity:
  """How much structure a GEMM operand's sparsity pattern has.

  Not an enum. The granularities a consumer may ask about form a lattice of
  tile shapes ordered componentwise, not a linear scale: a pattern that is
  complete in 8x1 blocks (good enough for an AVX-512 double kernel) and one
  that is complete in 4x4 tiles (good enough for a small matrix-unit fragment)
  are incomparable. Collapsing that into an ordered enum would force an
  arbitrary total order and lose the shape, which is exactly the information
  the next consumer needs.

  `blockShape` is per-dimension and measured from the pattern the layout
  actually stores, never from an `alignStride` request. A dense operand has no
  restriction and reports `None`.
  """

  __slots__ = ('blockShape',)

  def __init__(self, blockShape=None):
    self.blockShape = blockShape

  @classmethod
  def of(cls, memoryLayout):
    if not memoryLayout.isSparse():
      return cls(None)
    return cls(tuple(memoryLayout.sparsityBlockShape()))

  @property
  def dense(self):
    return self.blockShape is None

  def respects(self, tile):
    """Can a kernel treat this operand as tiles of shape `tile`?"""
    if self.dense:
      return True
    return all(t <= b for t, b in zip(tile, self.blockShape))

  def __bool__(self):
    # keeps historical `if sparseA:` checks in external GemmTool subclasses working
    return not self.dense

  def __repr__(self):
    return 'Sparsity(dense)' if self.dense else 'Sparsity{}'.format(self.blockShape)

  def __eq__(self, other):
    return isinstance(other, Sparsity) and self.blockShape == other.blockShape

  def __hash__(self):
    return hash(self.blockShape)

DENSE = Sparsity(None)

class Preference(object):
  HIGHEST = 4
  HIGH = 3
  MODERATE = 2
  LOW = 1
  LOWEST = 0

class GemmTool(ABC):
  def __init__(self, operation_name: str, includes: List[str] = []):
    self.operation_name = operation_name
    self.includes = includes

  def archSupported(self):
    return True

  @abstractmethod
  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    pass

  @abstractmethod
  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    pass

class BLASlike(GemmTool):
  def __init__(self, operation_name: str, includes: List[str], c_code_init: str = ''):
    super().__init__(operation_name, includes)
    self.c_code_init = c_code_init

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    return Preference.MODERATE

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    return (sparseA.dense and sparseB.dense and target == 'cpu')

  def bool2Trans(self, trans):
    return 'Cblas{}Trans'.format('' if trans else 'No')

  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,
           alignedA, alignedC, prefetchName):
    parameters = [
      'CblasColMajor',
      self.bool2Trans(transA),
      self.bool2Trans(transB),
      M, N, K,
      alpha, A, ldA,
      B, ldB,
      beta, C, ldC]
    return '{}({});'.format(self.operation_name, ', '.join(str(p) for p in parameters))

class MKL(BLASlike):
  def __init__(self, arch):
    self._arch = arch
    super().__init__('cblas_{}gemm'.format(arch.precision.lower()), ['mkl_cblas.h'])

  def archSupported(self):
    return self._arch.host_name.lower() in {'snb', 'hsw', 'skx', 'knl'} or self._arch.host_name.lower().startswith('avx')

class OpenBLAS(BLASlike):
  def __init__(self, arch):
    super().__init__('cblas_{}gemm'.format(arch.precision.lower()), ['cblas.h'])

class BLIS(BLASlike):
  def __init__(self, arch):
    super().__init__('bli_{}gemm'.format(arch.precision.lower()), ['blis.h'], '{0} _blis_alpha; {0} _blis_beta;'.format(arch.typename))
    self._typename = arch.typename

  def bool2Trans(self, trans):
    return 'BLIS{}TRANSPOSE'.format('_' if trans else '_NO_')

  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,
           alignedA, alignedC, prefetchName):
    init = '_blis_alpha = {}; _blis_beta = {};'.format(alpha, beta)
    parameters = [
      self.bool2Trans(transA),
      self.bool2Trans(transB),
      M, N, K,
      '&_blis_alpha', 'const_cast<{}*>({})'.format(self._typename, A), 1, ldA,
      'const_cast<{}*>({})'.format(self._typename, B), 1, ldB,
      '&_blis_beta', C, 1, ldC]
    return '{} {}({});'.format(init, self.operation_name, ', '.join(str(p) for p in parameters))

class Eigen(BLASlike):
  def __init__(self, arch):
    super().__init__(None, ['Eigen/Eigen'])
    self._arch = arch

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    return (sparseA.dense and sparseB.dense and target == 'cpu')

  def bool2Trans(self, trans):
    return '.transpose()' if trans else ''

  def sizeTrans(self, rows, cols, trans):
    return (cols,rows) if trans else (rows,cols)

  def align(self, ld, allow):
    aligned = 'Unaligned'
    if self._arch.checkAlignment(ld) and self._arch.alignment in [16,32,64,128] and allow:
      aligned = 'Aligned{}'.format(self._arch.alignment)
    return aligned

  def matrixType(self, prec, dims, ld, aligned):
    # write an Eigen matrix map

    m, n = dims

    # importent to note: the Eigen outer stride is correct, unless we're dealing with a vector.
    # meaning: at least one matrix dim is 1. Then, we need the inner stride instead.
    # cf. https://libeigen.gitlab.io/eigen/docs-5.0.1/classEigen_1_1Stride.html
    # meaning: if m == 1, we need to take care of potential padding.

    if m == 1:
      stride = f"Stride<{ld}, {ld}>"
    else:
      stride = f"Stride<{ld}, 1>"

    align = self.align(ld, aligned)

    return f"Map<Matrix<{prec}, {m}, {n}>, Eigen::{align}, {stride}>"

  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,
           alignedA, alignedC, prefetchName):
    AxB = '{alpha}_mapA{transA}*_mapB{transB}'.format(
            alpha=str(alpha) + '*' if alpha != 1.0 else '',
            transA=self.bool2Trans(transA), transB=self.bool2Trans(transB),
          )
    code = ''
    if beta == 1.0:
      code = '_mapC.noalias() += {AxB};'.format(AxB=AxB)
    elif beta == 0.0:
      code = '_mapC = {AxB};'.format(AxB=AxB)
    else:
      code = '_mapC *= {beta}; _mapC.noalias() += {AxB};'.format(AxB=AxB, beta=beta)

    code = """{{
  using Eigen::Matrix;
  using Eigen::Map;
  using Eigen::Stride;
  {matA} _mapA(const_cast<{prec}*>({A}));
  {matB} _mapB(const_cast<{prec}*>({B}));
  {matC} _mapC({C});
  {code}
}}
    """.format(prec=self._arch.typename, M=M, N=N,
               matA=self.matrixType(self._arch.typename, self.sizeTrans(M,K,transA), ldA, alignedA),
               matB=self.matrixType(self._arch.typename, self.sizeTrans(K,N,transB), ldB, False),
               matC=self.matrixType(self._arch.typename, (M, N), ldC, alignedC),
               A=A, B=B, C=C, code=code)
    return code


class CodeGenerator(GemmTool):
  def __init__(self, operation_name: str,
               includes: List[str],
               cmd: str,
               arch,
               is_internal=False):
    super().__init__(operation_name, includes)
    self.cmd = cmd
    self._arch = arch
    self._is_internal = is_internal

  def is_internal(self):
    return self._is_internal


class LIBXSMM_JIT(CodeGenerator):
  def __init__(self, arch, cmd: str = 'libxsmm_gemm_generator', threshold: int = 128):
    super().__init__('libxsmm_jit',
                     ['libxsmm.h'],
                     cmd,
                     arch,
                     is_internal=True)
    self._threshold = threshold
    self._arch = arch

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if (m*n*k)**(1./3.) <= self._threshold:
      return Preference.HIGH
    return Preference.LOW

  def archSupported(self):
    supported_set = {'noarch', 'power9', 'power10', 'power11', 'rvv128', 'rvv256', 'rvv512', 'rvv1024', 'rvv2048', 'rvv4096', 'wsm', 'snb', 'hsw', 'skx', 'knc', 'knl', 'naples', 'rome', 'milan', 'bergamo', 'turin', "a64fx", "thunderx2t99", 'neon', 'sve128', 'sve256', 'sve512', 'apple-m1', "apple-m2", "apple-m3", "apple-m4", 'avx2-128', 'avx2-256', 'avx10-128', 'avx10-256', 'avx10-512'}
    return self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    # Note:
    # Libxsmm falls back to blas for transA and more general alpha/beta
    # See e.g. here:
    # https://libxsmm.readthedocs.io/en/latest/libxsmm_qna/#what-is-a-small-matrix-multiplication
    # https://github.com/hfp/libxsmm/issues/396#issuecomment-674741063
    return self.archSupported() and sparseA.dense and sparseB.dense and (not transA) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'cpu'

class LIBXSMM(CodeGenerator):
  def __init__(self, arch, cmd: str = 'libxsmm_gemm_generator', threshold: int = 128):
    super().__init__('libxsmm', [], cmd, arch)
    self._threshold = threshold

  def archSupported(self):
    supported_set = {'noarch', 'wsm', 'snb', 'hsw', 'skx', 'knc', 'knl', 'naples', 'rome', 'milan', 'bergamo', 'turin', 'avx2-256', 'avx10-512'}
    return self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    # LIBXSMM bakes the pattern into the generated kernel, so any granularity is
    # fine; it just cannot take both operands sparse at once.
    return self.archSupported() and not (sparseA and sparseB) and (not transA and not transB) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'cpu'

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if sparseA:
      return Preference.LOW
    if sparseB:
      return Preference.MODERATE
    if (m*n*k)**(1./3.) <= self._threshold:
      return Preference.HIGH
    return Preference.LOW

class PSpaMM(CodeGenerator):
  def __init__(self, arch, cmd: str = 'pspamm-generator', threshold: int = 128):
    super().__init__('pspamm', [], cmd, arch)
    self._threshold = threshold

  def archSupported(self):
    supported_set = {'rvv128', 'rvv256', 'rvv512', 'rvv1024', 'rvv2048', 'rvv4096', 'thunderx2t99', 'knl', 'skx', 'a64fx', 'hsw', 'naples', 'rome', 'milan', 'bergamo', 'turin', 'neon', 'sve128', 'sve256', 'sve512', 'sve1024', 'sve2048', 'apple-m1', 'apple-m2', "apple-m3", "apple-m4", 'avx2-128', 'avx2-256', 'avx10-128', 'avx10-256', 'avx10-512', 'lsx', 'lasx'}
    return self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    # NOTE: PSpaMM 0.3.0+ supports SIMD-aligned block sparsity in A (which is currently covered by sparseA + alignedA)
    # also, it supports for AVX512/10 and SVE unaligned matmuls in 0.3.1
    noAlign = self._arch.host_name.lower() in {'thunderx2t99', 'knl', 'skx', 'a64fx', 'bergamo', 'turin', 'sve128', 'sve256', 'sve512', 'sve1024', 'sve2048', 'avx10-128', 'avx10-256', 'avx10-512'}
    # PSpaMM vectorizes over rows of A: it needs whole `alignedReals`-row
    # columns, and move_register_block() raises NotImplementedError otherwise.
    vectorTile = (self._arch.alignedReals, 1)
    if not sparseA.respects(vectorTile):
      return False
    alignment = (not sparseA.dense) and alignedA \
             or sparseA.dense and (noAlign or alignedA)
    return self.archSupported() and (alignedC or noAlign) and alignment and (not transA and not transB) and target == 'cpu'

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if sparseB:
      return Preference.HIGH
    if sparseA and alignedA:
      return Preference.HIGH
    if (m*n*k)**(1./3.) <= self._threshold:
      return Preference.HIGH
    return Preference.LOW

  """ You may choose application-specific block-size parameters by overriding this function.
      Return empty dict for automatic block-size.
      Add entries bm,bn,bk to set specific block-sizes.
  """
  def blockSize(self, m, n, k):
    return dict()


class GemmForge(CodeGenerator):
  def __init__(self, arch, threshold: int = 256):
    super().__init__('', ['gemmforge_aux.h'], '', arch)
    self._threshold = threshold

  def archSupported(self):
    return self._arch.backend.lower() in {'cuda', 'hip', 'oneapi', 'acpp', 'hipsycl'}

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    return self.archSupported() and not (sparseA or sparseB) and target == 'gpu'

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if sparseA and sparseB:
      return Preference.LOWEST
    if not transA:
      return Preference.HIGHEST
    if m < 16:
      return Preference.LOWEST
    return Preference.HIGH

class tinytc(CodeGenerator):
  def __init__(self, arch):
    super().__init__('', [], '', arch)
    self._arch = arch

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    return Preference.HIGHEST

  def archSupported(self):
      return self._arch.backend.lower() in {'oneapi'}

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    return self.archSupported() and not (sparseA or sparseB) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'gpu'


class GeneratorCollection(object):
  def __init__(self, gemmTools: List[GemmTool]):
    self.gemmTools = gemmTools
    self.selected = set()

  def getGemmTool(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                  beta, alignedA, alignedC, target):
    tools = dict()
    for gemmTool in reversed(self.gemmTools):
      if gemmTool.supported(m, n, k, sparseA, sparseB, transA, transB, alpha,
                            beta, alignedA, alignedC, target):
        tools[gemmTool.preference(m, n, k, sparseA, sparseB, transA, transB, alpha, beta,
                                  alignedA, alignedC)] = gemmTool

    select = None
    if tools:
      select = max(tools.items(), key=operator.itemgetter(0))[1]

    if select:
      self.selected.add(select)

    return select

class DefaultGeneratorCollection(GeneratorCollection):
  def __init__(self, arch):
    super().__init__([])

    # CPU/GemmGen
    libxsmm = LIBXSMM(arch)
    libxsmm_jit = LIBXSMM_JIT(arch)
    pspamm = PSpaMM(arch)

    # CPU/BlasLike
    mkl = MKL(arch)
    blis = BLIS(arch)
    openblas = OpenBLAS(arch)
    eigen = Eigen(arch)

    # GPU
    forge = GemmForge(arch)

    # GPU/Intel
    ttc = tinytc(arch)

    order = [libxsmm_jit, libxsmm, pspamm, mkl, openblas, blis, eigen, forge, ttc]

    generators = [gen for gen in order if gen.archSupported()]

    self.gemmTools = generators
