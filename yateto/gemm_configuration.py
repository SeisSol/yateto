from typing import List
from abc import ABC, abstractmethod

class GemmTool(ABC):
  def __init__(self, operation_name: str, includes: List[str] = []):
    self.operation_name = operation_name
    self.includes = includes

  @abstractmethod
  def isGoodIdea(self, m, n, k):
    pass

  @abstractmethod
  def supported(self, sparseA, sparseB, transA, transB, alpha, beta):
    pass

class BLASlike(GemmTool):
  def __init__(self, operation_name: str, includes: List[str], c_code_init: str = ''):
    super().__init__(operation_name, includes)
    self.c_code_init = c_code_init

  def isGoodIdea(self, m, n, k):
    return True

  def supported(self, sparseA, sparseB, transA, transB, alpha, beta):
    return (not sparseA and not sparseB)

  def bool2Trans(self, trans):
    return 'Cblas{}Trans'.format('' if trans else 'No')

  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC):
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
  def __init__(self):
    super().__init__('cblas_dgemm', ['mkl_cblas.h'])

class OpenBLAS(BLASlike):
  def __init__(self):
    super().__init__('cblas_dgemm', ['cblas.h'])

class BLIS(BLASlike):
  def __init__(self):
    super().__init__('bli_dgemm', ['blis.h'], 'double _blis_alpha; double _blis_beta;')

  def bool2Trans(self, trans):
    return 'BLIS{}TRANSPOSE'.format('_' if trans else '_NO_'),

  def call(self, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC):
    init = '_blis_alpha = {}; _blis_beta = {};'.format(alpha, beta)
    parameters = [
      self.bool2Trans(transA),
      self.bool2Trans(transB),
      M, N, K,
      '&_blas_alpha', 'const_cast<double*>({})'.format(A), 1, ldA,
      'const_cast<double*>({})'.format(B), 1, ldB,
      '&_blas_beta', C, 1, ldC]
    return '{} {}({});'.format(init, self.operation_name, ', '.join(str(p) for p in parameters))

class CodeGenerator(GemmTool):
  def __init__(self, operation_name: str, includes: List[str], cmd: str, threshold: int):
    super().__init__(operation_name, includes)
    self.cmd = cmd
    self._threshold = threshold

  @abstractmethod
  def _sparse(self, sparseA, sparseB):
    pass

  def isGoodIdea(self, m, n, k):
    return (m*n*k)**(1./3.) <= self._threshold

  def supported(self, sparseA, sparseB, transA, transB, alpha, beta):
    return self._sparse(sparseA, sparseB) and (not transA and not transB) and alpha == 1.0 and beta in [0.0, 1.0]

class LIBXSMM(CodeGenerator):
  def __init__(self, threshold: int = 128):
    super().__init__('libxsmm', [], 'libxsmm_gemm_generator', threshold)

  def _sparse(self, sparseA, sparseB):
    return not (sparseA and sparseB)

class PSpaMM(CodeGenerator):
  def __init__(self, threshold: int = 128):
    super().__init__('pspamm', [], 'pspamm.py', threshold)

  def _sparse(self, sparseA, sparseB):
    return not sparseA

class GeneratorCollection(object):
  def __init__(self, gemmTools: List[GemmTool]):
    self.gemmTools = gemmTools
    self.selected = set()

  def getGemmTool(self, m, n, k, sparseA, sparseB, transA, transB, alpha, beta):
    tools = dict()
    for gemmTool in reversed(self.gemmTools):
      if gemmTool.supported(sparseA, sparseB, transA, transB, alpha, beta):
        tools[gemmTool.isGoodIdea(m, n, k)] = gemmTool

    select = None
    if True in tools:
      select = tools[True]
    elif False in tools:
      select = tools[False]

    if select:
      self.selected.add(select)

    return select

class DefaultGeneratorCollection(GeneratorCollection):
  def __init__(self, arch):
    super().__init__([])
    libxsmm = LIBXSMM()
    pspamm = PSpaMM()
    mkl = MKL()
    blis = BLIS()
    openblas = OpenBLAS()
    defaults = {
      'snb' : [libxsmm, mkl, blis],
      'hsw' : [libxsmm, mkl, blis],
      'knl' : [libxsmm, pspamm, mkl, blis],
      'armv8' : [pspamm, openblas, blis]
    }

    if arch.name in defaults:
      self.gemmTools = defaults[arch.name]
    else:
      raise Exception("Default generator collection for architecture {} is missing.".format(arch))
