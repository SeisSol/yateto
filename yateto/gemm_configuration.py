from typing import List
from abc import ABC, abstractmethod
from .type import Datatype
import operator

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
  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    pass

  @abstractmethod
  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    pass
  
  # shortcut for legacy reasons
  @classmethod
  def _equalType(cls, datatypeA, datatypeB, datatypeC, types=(Datatype.F32, Datatype.F64)):
    return datatypeA == datatypeC and datatypeB == datatypeC and datatypeC in types

class BLASlike(GemmTool):
  def __init__(self, prefix, includes: List[str], c_code_init: str = ''):
    super().__init__(prefix, includes)
    self.c_code_init = c_code_init

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    return Preference.MODERATE

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    return (not sparseA and not sparseB and target == 'cpu' and self._equalType(datatypeA, datatypeB, datatypeC))

  def bool2Trans(self, trans):
    return 'Cblas{}Trans'.format('' if trans else 'No')

  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,
           alignedA, alignedC, datatypeA, datatypeB, datatypeC, prefetchName):
    precision = {
      Datatype.F32: 's',
      Datatype.F64: 'd'
    }[datatypeC]
    parameters = [
      'CblasColMajor',
      self.bool2Trans(transA),
      self.bool2Trans(transB),
      M, N, K,
      alpha, A, ldA,
      B, ldB,
      beta, C, ldC]
    return '{}_{}gemm({});'.format(self.prefix, precision, ', '.join(str(p) for p in parameters))

class MKL(BLASlike):
  def __init__(self, arch):
    self._arch = arch
    super().__init__('cblas', ['mkl_cblas.h'])
  
  def archSupported(self):
    return self._arch.host_name.lower() in {'snb', 'hsw', 'skx', 'knl'} or self._arch.host_name.lower().startswith('avx')

class OpenBLAS(BLASlike):
  def __init__(self, arch):
    super().__init__('cblas', ['cblas.h'])

class BLIS(BLASlike):
  def __init__(self, arch):
    super().__init__('bli', ['blis.h'])

  def bool2Trans(self, trans):
    return 'BLIS{}TRANSPOSE'.format('_' if trans else '_NO_')

  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,
           alignedA, alignedC, datatypeA, datatypeB, datatypeC, prefetchName):
    precision = {
      Datatype.F32: 's',
      Datatype.F64: 'd'
    }[datatypeC]
    initA = f'{datatypeC.ctype()} _blis_alpha = {alpha};'
    initB = f'{datatypeC.ctype()} _blis_beta = {beta};'
    parameters = [
      self.bool2Trans(transA),
      self.bool2Trans(transB),
      M, N, K,
      '&_blis_alpha', f'const_cast<{datatypeA.ctype()}*>({A})', 1, ldA,
      f'const_cast<{datatypeB.ctype()}*>({B})', 1, ldB,
      '&_blis_beta', C, 1, ldC]
    return '{{ {}{} {}_{}gemm({}); }}'.format(initA, initB, self.prefix, precision, ', '.join(str(p) for p in parameters))

class Eigen(BLASlike):
  def __init__(self, arch):
    super().__init__(None, ['Eigen/Eigen'])
    self._arch = arch

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    return (not sparseA and not sparseB and target == 'cpu')

  def bool2Trans(self, trans):
    return '.transpose()' if trans else ''

  def sizeTrans(self, rows, cols, trans):
    return f'{cols},{rows}' if trans else f'{rows},{cols}'

  def align(self, ld):
    aligned = 'Unaligned'
    if self._arch.checkAlignment(ld) and self._arch.alignment in [16,32,64,128]:
      aligned = f'Aligned{self._arch.alignment}'
    return aligned


  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC,
           alignedA, alignedC, datatypeA, datatypeB, datatypeC, prefetchName):
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
  Map<Matrix<{precA},{sizeA}>,Eigen::{alignA},Stride<{ldA},1>> _mapA(const_cast<{precA}*>({A}));
  Map<Matrix<{precB},{sizeB}>,Eigen::Unaligned,Stride<{ldB},1>> _mapB(const_cast<{precB}*>({B}));
  Map<Matrix<{precC},{M},{N}>,Eigen::{alignC},Stride<{ldC},1>> _mapC({C});
  {code}
}}
    """.format(precA=datatypeA.ctype(TypeFlavor.EIGEN),
               precB=datatypeB.ctype(TypeFlavor.EIGEN),
               precC=datatypeC.ctype(TypeFlavor.EIGEN),
               M=M, N=N,
               sizeA=self.sizeTrans(M,K,transA),
               sizeB=self.sizeTrans(K,N,transB),
               ldA=ldA, ldB=ldB, ldC=ldC, A=A, B=B, C=C,
               alignA=self.align(ldA), alignC=self.align(ldC),
               code=code)
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

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    if (m*n*k)**(1./3.) <= self._threshold:
      return Preference.HIGH
    return Preference.LOW

  def archSupported(self):
    supported_set = {'noarch', 'rvv128', 'rvv256', 'rvv512', 'rvv1024', 'rvv2048', 'rvv4096', 'wsm', 'snb', 'hsw', 'skx', 'knc', 'knl', 'naples', 'rome', 'milan', 'bergamo', 'turin', "a64fx", "thunderx2t99", 'neon', 'sve128', 'sve256', 'sve512', 'apple-m1', "apple-m2", "apple-m3", "apple-m4", 'avx2-128', 'avx2-256', 'avx10-128', 'avx10-256', 'avx10-512'}
    return self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    # Note:
    # Libxsmm falls back to blas for transA and more general alpha/beta
    # See e.g. here:
    # https://libxsmm.readthedocs.io/en/latest/libxsmm_qna/#what-is-a-small-matrix-multiplication
    # https://github.com/hfp/libxsmm/issues/396#issuecomment-674741063
    return self.archSupported() and not (sparseA or sparseB) and (not transA) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'cpu' and self._equalType(datatypeA, datatypeB, datatypeC) # TODO: no, there's more

class LIBXSMM(CodeGenerator):
  def __init__(self, arch, cmd: str = 'libxsmm_gemm_generator', threshold: int = 128):
    super().__init__('libxsmm', [], cmd, arch)
    self._threshold = threshold

  def archSupported(self):
    supported_set = {'noarch', 'wsm', 'snb', 'hsw', 'skx', 'knc', 'knl', 'naples', 'rome', 'milan', 'bergamo', 'turin', 'avx2-256', 'avx10-512'}
    return self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    return self.archSupported() and not (sparseA and sparseB) and (not transA and not transB) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'cpu' and (self._equalType(datatypeA, datatypeB, datatypeC) or (self._equalType(datatypeA, datatypeB, datatypeC, (Datatype.I16,)) and not sparseA and not sparseB))

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
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
    supported_set = {'rvv128', 'rvv256', 'rvv512', 'rvv1024', 'rvv2048', 'rvv4096', 'thunderx2t99', 'knl', 'skx', 'a64fx', 'hsw', 'naples', 'rome', 'milan', 'bergamo', 'turin', 'neon', 'sve128', 'sve256', 'sve512', 'sve1024', 'sve2048', 'apple-m1', 'apple-m2', "apple-m3", "apple-m4", 'avx2-128', 'avx2-256', 'avx10-128', 'avx10-256', 'avx10-512'}
    return self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    # NOTE: PSpaMM 0.3.0+ supports SIMD-aligned block sparsity in A (which is currently covered by sparseA + alignedA)
    return self.archSupported() and alignedC and alignedA and (not transA and not transB) and target == 'cpu' and self._equalType(datatypeA, datatypeB, datatypeC, [Datatype.BF16, Datatype.F16, Datatype.F32, Datatype.F64])

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
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
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    return self.archSupported() and not (sparseA or sparseB) and target == 'gpu' and self._equalType(datatypeA, datatypeB, datatypeC)

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
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

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    return Preference.HIGHEST

  def archSupported(self):
      return self._arch.backend.lower() in {'oneapi'}

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    return self.archSupported() and not (sparseA or sparseB) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'gpu' and self._equalType(datatypeA, datatypeB, datatypeC) # TODO: really?


class GeneratorCollection(object):
  def __init__(self, gemmTools: List[GemmTool]):
    self.gemmTools = gemmTools
    self.selected = set()

  def getGemmTool(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                  beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
    tools = dict()
    for gemmTool in reversed(self.gemmTools):
      if gemmTool.supported(m, n, k, sparseA, sparseB, transA, transB, alpha,
                            beta, alignedA, alignedC, datatypeA, datatypeB, datatypeC, target):
        tools[gemmTool.preference(m, n, k, sparseA, sparseB, transA, transB, alpha, beta,
                                  alignedA, alignedC, datatypeA, datatypeB, datatypeC, target)] = gemmTool

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
