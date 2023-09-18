from typing import List
from abc import ABC, abstractmethod
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
    return (not sparseA and not sparseB and target == 'cpu')

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
    super().__init__('cblas_{}gemm'.format(arch.precision.lower()), ['mkl_cblas.h'])

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
    return (not sparseA and not sparseB and target == 'cpu')

  def bool2Trans(self, trans):
    return '.transpose()' if trans else ''

  def sizeTrans(self, rows, cols, trans):
    return '{},{}'.format(cols,rows) if trans else '{},{}'.format(rows,cols)

  def align(self, ld):
    aligned = 'Unaligned'
    if self._arch.checkAlignment(ld) and self._arch.alignment in [16,32,64,128]:
      aligned = 'Aligned{}'.format(self._arch.alignment)
    return aligned


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
  Map<Matrix<{prec},{sizeA}>,Eigen::{alignA},Stride<{ldA},1>> _mapA(const_cast<{prec}*>({A}));
  Map<Matrix<{prec},{sizeB}>,Eigen::Unaligned,Stride<{ldB},1>> _mapB(const_cast<{prec}*>({B}));
  Map<Matrix<{prec},{M},{N}>,Eigen::{alignC},Stride<{ldC},1>> _mapC({C});
  {code}
}}
    """.format(prec=self._arch.typename, M=M, N=N,
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

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if (m*n*k)**(1./3.) <= self._threshold:
      return Preference.HIGH
    return Preference.LOW

  def _archSupported(self):
    supported_set = {'noarch', 'wsm', 'snb', 'hsw', 'skx', 'knc', 'knl', 'rome', "a64fx", "thunderx2t99"}

    if self._arch.name.lower() in supported_set:
      return True
    else:
      return self._arch.host_name and self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    # Note:
    # Libxsmm falls back to blas for transA and more general alpha/beta
    # See e.g. here:
    # https://libxsmm.readthedocs.io/en/latest/libxsmm_qna/#what-is-a-small-matrix-multiplication
    # https://github.com/hfp/libxsmm/issues/396#issuecomment-674741063
    return self._archSupported() and not (sparseA or sparseB) and (not transA) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'cpu'

class LIBXSMM(CodeGenerator):
  def __init__(self, arch, cmd: str = 'libxsmm_gemm_generator', threshold: int = 128):
    super().__init__('libxsmm', [], cmd, arch)
    self._threshold = threshold

  def _archSupported(self):
    supported_set = {'noarch', 'wsm', 'snb', 'hsw', 'skx', 'knc', 'knl', 'rome'}

    if self._arch.name.lower() in supported_set:
      return True
    else:
      return self._arch.host_name and self._arch.host_name.lower() in supported_set

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    return self._archSupported() and not (sparseA and sparseB) and (not transA and not transB) and alpha == 1.0 and beta in [0.0, 1.0] and target == 'cpu'

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if sparseA:
      return Preference.LOW
    if sparseB:
      return Preference.MODERATE
    if (m*n*k)**(1./3.) <= self._threshold:
      return Preference.HIGH
    return Preference.LOW

class PSpaMM(CodeGenerator):
  def __init__(self, arch, cmd: str = 'pspamm.py', threshold: int = 128):
    super().__init__('pspamm', [], cmd, arch)
    self._threshold = threshold

  def _archSupported(self):
    supported_set = {'thunderx2t99', 'knl', 'skx', 'a64fx'}
    if self._arch.name.lower() in supported_set:
      return True
    else:
      return self._arch.host_name and self._arch.host_name.lower() in supported_set


  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    return self._archSupported() and alignedA and alignedC and \
           not sparseA and (not transA and not transB) and target == 'cpu'

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if sparseB:
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

  def _is_arch_supported(self):
    return self._arch.backend.lower() in {'cuda', 'hip', 'oneapi', 'hipsycl'}

  def supported(self, m, n, k, sparseA, sparseB, transA, transB, alpha,
                beta, alignedA, alignedC, target):
    return self._is_arch_supported() and not (sparseA or sparseB) and target == 'gpu'

  def preference(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta, alignedA, alignedC):
    if sparseA and sparseB:
      return Preference.LOWEST
    if not transA:
      return Preference.HIGHEST
    if m < 16:
      return Preference.LOWEST
    return Preference.HIGH


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
    libxsmm = LIBXSMM(arch)
    libxsmm_jit = LIBXSMM_JIT(arch)
    pspamm = PSpaMM(arch)
    mkl = MKL(arch)
    blis = BLIS(arch)
    openblas = OpenBLAS(arch)
    eigen = Eigen(arch)
    forge = GemmForge(arch)
    defaults = {
      'snb' : [libxsmm_jit, libxsmm, mkl, blis, eigen],
      'hsw' : [libxsmm_jit, libxsmm, mkl, blis, eigen],
      'rome' : [libxsmm_jit, libxsmm, blis, eigen],
      'knl' : [libxsmm_jit, libxsmm, pspamm, mkl, blis, eigen],
      'skx' : [libxsmm_jit, libxsmm, pspamm, mkl, blis, eigen],
      'thunderx2t99' : [libxsmm_jit, pspamm, openblas, blis, eigen],
      'a64fx' : [libxsmm_jit, pspamm, openblas, blis, eigen],
      'power9' : [openblas, blis, eigen]
    }

    if arch.name in defaults:
      self.gemmTools = defaults[arch.name]
    elif arch.host_name in defaults:
      self.gemmTools = defaults[arch.host_name]
      if arch.is_accelerator:
        self.gemmTools.extend([forge])
    else:
      raise Exception("Default generator collection for architecture {} is missing.".format(arch))
