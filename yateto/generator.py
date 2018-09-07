import os
import itertools
from .ast.node import Node
from .ast.visitor import FindTensors
from .ast.transformer import *
from .codegen.cache import *
from .codegen.code import Cpp
from .codegen.visitor import *

class Kernel(object):
  def __init__(self, name, ast):
    self.name = name
    self.ast = ast
  
  def prepareUntilUnitTest(self):
    self.ast = DeduceIndices().visit(self.ast)
  
  def prepareUntilCodeGen(self):
    self.ast = EquivalentSparsityPattern().visit(self.ast)
    self.ast = StrengthReduction().visit(self.ast)
    self.ast = FindContractions().visit(self.ast)
    self.ast = ComputeMemoryLayout().visit(self.ast)
    self.ast = FindIndexPermutations().visit(self.ast)
    self.ast = SelectIndexPermutations().visit(self.ast)
    self.ast = ImplementContractions().visit(self.ast)
    
class KernelFamily(object):
  def __init__(self, name, parameterSpace, astGenerator):
    self._name = name
    self._parameterSpace = parameterSpace
    self._astGenerator = astGenerator

def simpleParameterSpace(*args):
  return itertools.product(*[list(range(i)) for i in args])

class Generator(object):
  HEADER = 'h'
  CPP = 'cpp'
  INIT_FILE_NAME = 'init'
  KERNELS_FILE_NAME = 'kernels'
  ROUTINES_FILE_NAME = 'routines'
  UNIT_TESTS_FILE_NAME = 'KernelTests.t'
  HEADER_GUARD_SUFFIX = 'H_'
  SUPPORT_LIBRARY_HEADER = 'yateto.h'
  
  def __init__(self, arch):
    self._kernels = list()
    self._arch = arch
  
  def add(self, name: str, ast: Node):
    kernel = Kernel(name, ast)
    self._kernels.append(kernel)
  
  def addFamily(self, name: str, parameterSpace, astGenerator):
    pass
    #~ family = KernelFamily(name, parameterSpace, astGenerator)
    #~ self._kernels.append(family)
  
  def _headerGuardName(self, namespace, fileBaseName):
    partlist = namespace.upper().split('::') + [fileBaseName.upper(), self.HEADER_GUARD_SUFFIX]
    return '_'.join(partlist)

  def generate(self, outputDir: str, namespace: str):
    for kernel in self._kernels:
      kernel.prepareUntilUnitTest()

    unitTestsHPath = os.path.join(outputDir, '{}.{}'.format(self.UNIT_TESTS_FILE_NAME, self.HEADER))
    
    kernelsHFileName = '{}.{}'.format(self.KERNELS_FILE_NAME, self.HEADER)
    kernelsHPath = os.path.join(outputDir, kernelsHFileName)
    kernelsCppPath = os.path.join(outputDir, '{}.{}'.format(self.KERNELS_FILE_NAME, self.CPP))
    
    routinesHPath = os.path.join(outputDir, '{}.{}'.format(self.ROUTINES_FILE_NAME, self.HEADER))
    routinesCppPath = os.path.join(outputDir, '{}.{}'.format(self.ROUTINES_FILE_NAME, self.CPP))
    
    initHFileName = '{}.{}'.format(self.INIT_FILE_NAME, self.HEADER)
    initHPath = os.path.join(outputDir, initHFileName)
    initCppPath = os.path.join(outputDir, '{}.{}'.format(self.INIT_FILE_NAME, self.CPP))

    with Cpp(unitTestsHPath) as cpp:
      cpp.include(kernelsHFileName)
      cpp.include(initHFileName)
      with cpp.Namespace(namespace):
        for kernel in self._kernels:
          UnitTestGenerator(cpp, self._arch).generate(kernel.name, kernel.ast)

    for kernel in self._kernels:
      kernel.prepareUntilCodeGen()

    cache = RoutineCache()
    with Cpp(kernelsCppPath) as cpp:
      with Cpp(kernelsHPath) as header:
        cpp.include(kernelsHFileName)
        with cpp.Namespace(namespace):
          with header.Namespace(namespace):
            for kernel in self._kernels:
              KernelGenerator(self._arch, cache).generate(cpp, header, kernel.name, kernel.ast)
    
    with Cpp(routinesHPath) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.ROUTINES_FILE_NAME)):
        cache.generate(header, routinesCppPath)
    
    tensors = dict()
    for kernel in self._kernels:
      tensors.update( FindTensors().visit(kernel.ast) )
    
    with Cpp(initHPath) as header:
      header.include(self.SUPPORT_LIBRARY_HEADER)
      with header.Namespace(namespace):
        InitializerGenerator(header, self._arch).generate(tensors.values())
