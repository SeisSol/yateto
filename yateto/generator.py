import os
import itertools
from .ast.node import Node
from .ast.visitor import ComputeOptimalFlopCount, FindTensors
from .ast.transformer import *
from .codegen.cache import *
from .codegen.code import Cpp
from .codegen.visitor import *
from .controlflow.visitor import AST2ControlFlow
from .controlflow.transformer import *

class Kernel(object):
  def __init__(self, name, ast):
    self.name = name
    self.ast = ast
    self.cfg = None
  
  def prepareUntilUnitTest(self):
    self.ast = DeduceIndices().visit(self.ast)
    ast2cf = AST2ControlFlow()
    ast2cf.visit(self.ast)
    self.cfg = ast2cf.cfg()
  
  def prepareUntilCodeGen(self):
    self.ast = EquivalentSparsityPattern().visit(self.ast)
    self.ast = StrengthReduction().visit(self.ast)
    self.ast = FindContractions().visit(self.ast)
    self.ast = ComputeMemoryLayout().visit(self.ast)
    self.ast = FindIndexPermutations().visit(self.ast)
    self.ast = SelectIndexPermutations().visit(self.ast)
    self.ast = ImplementContractions().visit(self.ast)

    ast2cf = AST2ControlFlow()
    ast2cf.visit(self.ast)
    self.cfg = ast2cf.cfg()
    self.cfg = FindLiving().visit(self.cfg)
    self.cfg = SubstituteForward().visit(self.cfg)
    self.cfg = SubstituteBackward().visit(self.cfg)
    self.cfg = RemoveEmptyStatements().visit(self.cfg)
    self.cfg = MergeActions().visit(self.cfg)
    self.cfg = ReuseTemporaries().visit(self.cfg)
    
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
  UNIT_TESTS_FILE_NAME = 'KernelTests'
  HEADER_GUARD_SUFFIX = 'H_'
  SUPPORT_LIBRARY_HEADER = 'yateto.h'
  TEST_CLASS = 'KernelTestSuite'
  TEST_NAMESPACE = 'unit_test'
  
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

  def generate(self, outputDir: str, namespace = 'yateto'):
    print('Deducing indices...')
    for kernel in self._kernels:
      kernel.prepareUntilUnitTest()

    unitTestsHPath = os.path.join(outputDir, '{}.t.{}'.format(self.UNIT_TESTS_FILE_NAME, self.HEADER))
    
    kernelsHFileName = '{}.{}'.format(self.KERNELS_FILE_NAME, self.HEADER)
    kernelsHPath = os.path.join(outputDir, kernelsHFileName)
    kernelsCppPath = os.path.join(outputDir, '{}.{}'.format(self.KERNELS_FILE_NAME, self.CPP))
    
    routinesHFileName = '{}.{}'.format(self.ROUTINES_FILE_NAME, self.HEADER)
    routinesHPath = os.path.join(outputDir, routinesHFileName)
    routinesCppPath = os.path.join(outputDir, '{}.{}'.format(self.ROUTINES_FILE_NAME, self.CPP))
    
    initHFileName = '{}.{}'.format(self.INIT_FILE_NAME, self.HEADER)
    initHPath = os.path.join(outputDir, initHFileName)
    initCppPath = os.path.join(outputDir, '{}.{}'.format(self.INIT_FILE_NAME, self.CPP))

    print('Generating unit tests...')
    with Cpp(unitTestsHPath) as cpp:
      with cpp.HeaderGuard(self._headerGuardName(namespace, self.UNIT_TESTS_FILE_NAME)):
        cpp.includeSys('cxxtest/TestSuite.h')
        cpp.include(kernelsHFileName)
        cpp.include(initHFileName)
        with cpp.PPIfndef('NDEBUG'):
          cpp('long long libxsmm_num_total_flops = 0;')
        with cpp.Namespace(namespace):
          with cpp.Namespace(self.TEST_NAMESPACE):
            cpp.classDeclaration(self.TEST_CLASS)
        with cpp.Class('{}::{}::{} : public CxxTest::TestSuite'.format(namespace, self.TEST_NAMESPACE, self.TEST_CLASS)):
          cpp.label('public')
          for kernel in self._kernels:
            UnitTestGenerator(self._arch).generate(cpp, kernel.name, kernel.cfg)

    print('Optimizing ASTs...')
    for kernel in self._kernels:
      kernel.prepareUntilCodeGen()

    print('Generating kernels...')
    cache = RoutineCache()
    with Cpp(kernelsCppPath) as cpp:
      cpp.includeSys('cassert')
      cpp.includeSys('cstring')
      cpp.include(routinesHFileName)
      with Cpp(kernelsHPath) as header:
        with header.HeaderGuard(self._headerGuardName(namespace, self.KERNELS_FILE_NAME)):
          cpp.include(kernelsHFileName)
          with cpp.Namespace(namespace):
            with header.Namespace(namespace):
              for kernel in self._kernels:
                nonZeroFlops = ComputeOptimalFlopCount().visit(kernel.ast)
                OptimisedKernelGenerator(self._arch, cache).generate(cpp, header, kernel.name, nonZeroFlops, kernel.cfg)

    print('Calling external code generators...')
    with Cpp(routinesHPath) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.ROUTINES_FILE_NAME)):
        cache.generate(header, routinesCppPath)
    
    tensors = dict()
    for kernel in self._kernels:
      tensors.update( FindTensors().visit(kernel.ast) )

    print('Generating initialization code...')
    with Cpp(initHPath) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.INIT_FILE_NAME)):
        header.include(self.SUPPORT_LIBRARY_HEADER)
        with header.Namespace(namespace):
          InitializerGenerator(header, self._arch).generate(tensors.values())
