from io import StringIO
from ..ast.visitor import Visitor
from .code import Cpp
from .factory import KernelFactory

class KernelGenerator(Visitor):
  ARGUMENT_NAME = 'p'
  
  def __init__(self, cpp):
    self._cpp = cpp
    self._tmp = 0
    self._tensors = list()
    self._factory = None
  
  def generate(self, node):
    structName = 'test__params'
    cpp = self._cpp
    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as self._cpp:
      self._factory = KernelFactory(self._cpp)
      with self._cpp.Function('test', '{}& {}'.format(structName, self.ARGUMENT_NAME)):
        self.visit(node)
      function = functionIO.getvalue()
    with cpp.Struct(structName):
      for tensor in self._tensors:
        cpp('double* {};'.format(tensor))
    cpp(function)

  def generic_visit(self, node):
    names = [self.visit(child) for child in node]
    size = 1
    for s in node.indices.shape():
      size *= s
    tmpName = 'tmp{}'.format(self._tmp)
    self._cpp('double {}[{}];'.format(tmpName, size))
    self._tmp += 1
    self._factory.create(node, tmpName, names)
    return tmpName
  
  def visit_IndexedTensor(self, node):
    self._tensors.append(node.name())
    return '{}.{}'.format(self.ARGUMENT_NAME, node.name())
