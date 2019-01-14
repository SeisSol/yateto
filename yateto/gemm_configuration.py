from typing import List
from abc import ABC

class GemmTool(ABC):
  def __init__(self, operation_name: str, includes: List[str] = [], supportsDense: bool = True, supportsSparse: bool = False):
    self.operation_name = operation_name
    self.includes = includes
    self.supportsDense = supportsDense
    self.supportsSparse = supportsSparse

class BLASlike(GemmTool):
  def __init__(self, operation_name: str, includes: List[str], parameters: List[str], supportsDense: bool = True, supportsSparse: bool = False, c_code_prep: str = "", c_code_init: str=""):
    super().__init__(operation_name, includes, supportsDense, supportsSparse)
    self.parameters = parameters
    self.c_code_prep = c_code_prep
    self.c_code_init = c_code_init

class CodeGenerator(GemmTool):
  def __init__(self, operation_name: str, cmd: str, includes: List[str], supportsDense: bool, supportsSparse: bool):
    super().__init__(operation_name, includes, supportsDense, supportsSparse)
    self.cmd = cmd

class GeneratorCollection(object):
  def __init__(self, gemmTools: List[GemmTool]):
    self.gemmTools = gemmTools
    self.selected = set([])

  def getSparseGemmTool(self):
    for i in range(len(self.gemmTools)):
      if self.gemmTools[i].supportsSparse:
        self.selected.add(self.gemmTools[i])
        return self.gemmTools[i]
    raise Exception("Yateto attempts to utilize sparse matrix multiplication but no such BLASlike or CodeGenerator is provided")
  
  def getDenseGemmTool(self):
    for i in range(len(self.gemmTools)):
      if self.gemmTools[i].supportsDense:
        self.selected.add(self.gemmTools[i])
        return self.gemmTools[i]
    raise Exception("Yateto attempts to utilize dense matrix multiplication but no such BLASlike or CodeGenerator is provided")

libxsmm = CodeGenerator('libxsmm', 
                        'libxsmm_gemm_generator',
                         [],
                         True,
                         False)

pspamm = CodeGenerator('pspamm',
                      'pspamm.py',
                       [],
                       True,
                       True)

mkl = BLASlike('cblas_dgemm',
              ['mkl_cblas.h'],
              ['CblasColMajor', 'CblasNoTrans', 'CblasNoTrans', '$M', '$N', '$K', '$ALPHA', '$A', '$LDA', '$B', '$LDB', '$BETA', '$C', '$LDC'])

openblas = BLASlike('cblas_dgemm',
                   ['cblas.h'],
                   ['CblasColMajor', 'CblasNoTrans', 'CblasNoTrans', '$M', '$N', '$K', '$ALPHA', '$A', '$LDA', '$B', '$LDB', '$BETA', '$C', '$LDC'])

blis = BLASlike('bli_dgemm',
               ['blis.h'],
               ['BLIS_NO_TRANSPOSE', 'BLIS_NO_TRANSPOSE', '$M', '$N', '$K', '&alpha', 'const_cast <double*>($A)', '1', '$LDA', 'const_cast <double*>($B)', '1', '$LDB', '&beta', '$C', '1', '$LDC'],
                True,
                False,
                'alpha = $ALPHA; beta = $BETA;',
                'double alpha; double beta;')

class DefaultGeneratorCollection(GeneratorCollection):
  def __init__(self, arch):
    super().__init__([])
    defaults = {
      'snb' : [libxsmm, mkl, blis],
      'hsw' : [libxsmm, mkl, blis],
      'knl' : [libxsmm, pspamm, mkl, blis],
      'armv8' : [pspamm, openblas, blis]
    }

    if arch.name in defaults:
      self.gemmTools = defaults[arch.name]
    else:
      raise Exception("Architecture not supported.")
