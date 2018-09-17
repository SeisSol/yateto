import os
import itertools
import re
from .ast.cost import BoundingBoxCostEstimator
from .ast.node import Node
from .ast.visitor import ComputeOptimalFlopCount, FindIndexPermutations, FindTensors
from .ast.transformer import *
from .codegen.cache import *
from .codegen.code import Cpp
from .codegen.visitor import *
from .controlflow.visitor import AST2ControlFlow
from .controlflow.transformer import *

class Kernel(object):
  BASE_NAME = r'[a-zA-Z]\w*'
  VALID_NAME = r'^{}$'.format(BASE_NAME)

  def __init__(self, name, ast):
    self.name = name
    self.ast = ast
    self.cfg = None

  @classmethod
  def isValidName(cls, name):
    return re.match(cls.VALID_NAME, name) is not None
  
  def prepareUntilUnitTest(self):
    self.ast = DeduceIndices().visit(self.ast)
    ast2cf = AST2ControlFlow()
    ast2cf.visit(self.ast)
    self.cfg = ast2cf.cfg()
  
  def prepareUntilCodeGen(self, costEstimator):
    self.ast = EquivalentSparsityPattern().visit(self.ast)
    self.ast = StrengthReduction(costEstimator).visit(self.ast)
    self.ast = FindContractions().visit(self.ast)
    self.ast = ComputeMemoryLayout().visit(self.ast)
    permutationVariants = FindIndexPermutations().visit(self.ast)
    self.ast = SelectIndexPermutations(permutationVariants).visit(self.ast)
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
  GROUP_INDEX = r'\[(0|[1-9]\d*)\]'
  VALID_NAME = r'^{}({})$'.format(Kernel.BASE_NAME, GROUP_INDEX)

  def __init__(self):
    self._kernels = dict()
    self.name = None
    self._stride = None
  
  def items(self):
    return self._kernels.items()
  
  def __len__(self):
    return max(self._kernels.keys()) + 1
  
  @classmethod  
  def baseName(self, name):
    return re.match(Kernel.BASE_NAME, name).group(0)
  
  @classmethod
  def isValidName(cls, name):
    return re.match(cls.VALID_NAME, name) is not None
  
  @classmethod
  def group(cls, name):
    m = re.search(cls.GROUP_INDEX, name)
    return int(m.group(1))
  
  def setStride(self, stride):
    self._stride = stride
  
  def stride(self):
    if self._stride is not None:
      return self._stride
    return (len(self),)
    
  @classmethod
  def linear(cls, stride, group):
    assert len(stride) == len(group)
    index = 0
    for i,p in enumerate(group):
      index += p*stride[i]
    return index

  def add(self, name, ast):
    baseName = self.baseName(name)
    if not self.name:
      self.name = baseName
    assert baseName == self.name
    
    group = self.group(name)
    internalName = '_{}_{}'.format(baseName, group)
    self._kernels[group] = Kernel(internalName, ast)
  
  def prepareUntilUnitTest(self):
    for kernel in self._kernels.values():
      kernel.prepareUntilUnitTest()
  
  def prepareUntilCodeGen(self, costEstimator):
    for kernel in self._kernels.values():
      kernel.prepareUntilCodeGen(costEstimator)

def simpleParameterSpace(*args):
  return list(itertools.product(*[list(range(i)) for i in args]))

class Generator(object):
  HEADER = 'h'
  CPP = 'cpp'
  INIT_FILE_NAME = 'init'
  SIZES_FILE_NAME = 'sizes'
  KERNELS_FILE_NAME = 'kernels'
  ROUTINES_FILE_NAME = 'routines'
  UNIT_TESTS_FILE_NAME = 'KernelTests'
  HEADER_GUARD_SUFFIX = 'H_'
  SUPPORT_LIBRARY_HEADER = 'yateto.h'
  TEST_CLASS = 'KernelTestSuite'
  TEST_NAMESPACE = 'unit_test'
  
  def __init__(self, arch):
    self._kernels = list()
    self._kernelFamilies = dict()
    self._arch = arch
  
  def add(self, name: str, ast: Node):
    if KernelFamily.isValidName(name):
      baseName = KernelFamily.baseName(name)
      if baseName not in self._kernelFamilies:
        self._kernelFamilies[baseName] = KernelFamily()
      self._kernelFamilies[baseName].add(name, ast)
    else:      
      if not Kernel.isValidName(name):
        raise ValueError('Kernel name invalid (must match regexp {}): {}'.format(Kernel.VALID_NAME, name))
      kernel = Kernel(name, ast)
      self._kernels.append(kernel)
  
  def addFamily(self, name: str, parameterSpace, astGenerator):
    if name not in self._kernelFamilies:
      self._kernelFamilies[name] = KernelFamily()
    family = self._kernelFamilies[name]
    pmax = max(parameterSpace)
    stride = [1]
    for i in range(len(pmax)-1):
      stride.append(stride[i] * (pmax[i]+1))
    stride = tuple(stride)
    family.setStride(stride)
    for p in parameterSpace:
      indexedName = '{}[{}]'.format(name, KernelFamily.linear(stride, p))
      ast = astGenerator(*p)
      family.add(indexedName, ast)
  
  def _headerGuardName(self, namespace, fileBaseName):
    partlist = namespace.upper().split('::') + [fileBaseName.upper(), self.HEADER_GUARD_SUFFIX]
    return '_'.join(partlist)

  def generate(self, outputDir: str, namespace = 'yateto', costEstimator = BoundingBoxCostEstimator):
    print('Deducing indices...')
    for kernel in self._kernels:
      kernel.prepareUntilUnitTest()
    for family in self._kernelFamilies.values():
      family.prepareUntilUnitTest()

    unitTestsHPath = os.path.join(outputDir, '{}.t.{}'.format(self.UNIT_TESTS_FILE_NAME, self.HEADER))
    
    kernelsHFileName = '{}.{}'.format(self.KERNELS_FILE_NAME, self.HEADER)
    kernelsHPath = os.path.join(outputDir, kernelsHFileName)
    kernelsCppPath = os.path.join(outputDir, '{}.{}'.format(self.KERNELS_FILE_NAME, self.CPP))
    
    routinesHFileName = '{}.{}'.format(self.ROUTINES_FILE_NAME, self.HEADER)
    routinesHPath = os.path.join(outputDir, routinesHFileName)
    routinesCppPath = os.path.join(outputDir, '{}.{}'.format(self.ROUTINES_FILE_NAME, self.CPP))

    sizesHFileName = '{}.{}'.format(self.SIZES_FILE_NAME, self.HEADER)
    sizesHPath = os.path.join(outputDir, sizesHFileName)

    initHFileName = '{}.{}'.format(self.INIT_FILE_NAME, self.HEADER)
    initHPath = os.path.join(outputDir, initHFileName)

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
            UnitTestGenerator(self._arch).generate(cpp, kernel.name, kernel.name, kernel.cfg)
          for family in self._kernelFamilies.values():
            for group, kernel in family.items():
              UnitTestGenerator(self._arch).generate(cpp, kernel.name, family.name, kernel.cfg, group)

    print('Optimizing ASTs...')
    for kernel in self._kernels:
      print(kernel.name)
      kernel.prepareUntilCodeGen(costEstimator)
    for family in self._kernelFamilies.values():
      print(family.name)
      family.prepareUntilCodeGen(costEstimator)

    print('Generating kernels...')
    cache = RoutineCache()
    optKernelGenerator = OptimisedKernelGenerator(self._arch, cache)
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
                kernelOutline = optKernelGenerator.generateKernelOutline(nonZeroFlops, kernel.cfg)
                optKernelGenerator.generate(cpp, header, kernel.name, [kernelOutline])
              for family in self._kernelFamilies.values():
                kernelOutlines = [None] * len(family)
                for group, kernel in family.items():
                  nonZeroFlops = ComputeOptimalFlopCount().visit(kernel.ast)
                  kernelOutlines[group] = optKernelGenerator.generateKernelOutline(nonZeroFlops, kernel.cfg)
                optKernelGenerator.generate(cpp, header, family.name, kernelOutlines, family.stride())

    print('Calling external code generators...')
    with Cpp(routinesHPath) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.ROUTINES_FILE_NAME)):
        cache.generate(header, routinesCppPath)
    
    tensors = dict()
    for kernel in self._kernels:
      tensors.update( FindTensors().visit(kernel.ast) )
    for family in self._kernelFamilies.values():
      for group, kernel in family.items():
        tensors.update( FindTensors().visit(kernel.ast) )

    print('Generating initialization code...')
    initGen = InitializerGenerator(self._arch, tensors.values())
    with Cpp(sizesHPath) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.SIZES_FILE_NAME)):
        with header.Namespace(namespace):
          initGen.generateSizes(header)
    with Cpp(initHPath) as header:
      with header.HeaderGuard(self._headerGuardName(namespace, self.INIT_FILE_NAME)):
        header.include(sizesHFileName)
        header.include(self.SUPPORT_LIBRARY_HEADER)
        with header.Namespace(namespace):
          initGen.generateInit(header)
