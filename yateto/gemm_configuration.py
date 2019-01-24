from typing import List
from abc import ABC, abstractmethod

class GemmTool(ABC):
  def __init__(self, operation_name: str, includes: List[str] = []):
    self.operation_name = operation_name
    self.includes = includes
  
  @abstractmethod
  def supported(self, sparseA, sparseB, transA, transB, alpha, beta):
    pass

class BLASlike(GemmTool):
  def __init__(self, operation_name: str, includes: List[str], c_code_init: str = ''):
    super().__init__(operation_name, includes)
    self.c_code_init = c_code_init

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
  def __init__(self, operation_name: str, includes: List[str], cmd: str):
    super().__init__(operation_name, includes)
    self.cmd = cmd

  @abstractmethod
  def _sparse(self, sparseA, sparseB):
    pass

  def supported(self, sparseA, sparseB, transA, transB, alpha, beta):
    return self._sparse(sparseA, sparseB) and (not transA and not transB) and alpha == 1.0 and beta in [0.0, 1.0]

class LIBXSMM(CodeGenerator):
  def __init__(self):
    super().__init__('libxsmm', [], 'libxsmm_gemm_generator')

  def _sparse(self, sparseA, sparseB):
    return not (sparseA and sparseB)

class PSpaMM(CodeGenerator):
  def __init__(self):
    super().__init__('pspamm', [], 'pspamm.py')

  def _sparse(self, sparseA, sparseB):
    return not sparseA

class GeneratorCollection(object):
  def __init__(self, gemmTools: List[GemmTool]):
    self.gemmTools = gemmTools
    self.selected = set()

  def getGemmTool(self, sparseA, sparseB, transA, transB, alpha, beta):
    for i in range(len(self.gemmTools)):
      if self.gemmTools[i].supported(sparseA, sparseB, transA, transB, alpha, beta):
        self.selected.add(self.gemmTools[i])
        return self.gemmTools[i]
    return None

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
