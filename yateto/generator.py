import itertools
from .ast.node import Node
#~ from .ast.visitor import PrettyPrinter
from .ast.transformer import DeduceIndices, EquivalentSparsityPattern

class Kernel(object):
  def __init__(self, name, ast):
    self._name = name
    self._ast = ast
  
  def prepare(self):
    self._ast = DeduceIndices().visit(self._ast)
    self._ast = EquivalentSparsityPattern().visit(self._ast)
    #~ PrettyPrinter().visit(self._ast)
    
class KernelFamily(object):
  def __init__(self, name, parameterSpace, astGenerator):
    self._name = name
    self._parameterSpace = parameterSpace
    self._astGenerator = astGenerator

def simpleParameterSpace(*args):
  return itertools.product(*[list(range(i)) for i in args])

class Generator(object):
  def __init__(self):
    self._kernels = list()
  
  def add(self, name: str, ast: Node):
    kernel = Kernel(name, ast)
    self._kernels.append(kernel)
  
  def addFamily(self, name: str, parameterSpace, astGenerator):
    pass
    #~ family = KernelFamily(name, parameterSpace, astGenerator)
    #~ self._kernels.append(family)
    

  def generate(self, outputDir: str):
    for kernel in self._kernels:
      kernel.prepare()
