from io import StringIO
from ..ast.visitor import Visitor
from .code import Cpp
from .factory import KernelFactory

class KernelGenerator(Visitor):
  ARGUMENT_NAME = 'p'
  
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch
    self._tmp = 0
    self._tensors = set()
    self._factory = None
  
  def generate(self, node):
    structName = 'test__params'
    cpp = self._cpp
    functionIO = StringIO()
    function = ''
    with Cpp(functionIO) as self._cpp:
      self._factory = KernelFactory(self._cpp, self._arch)
      with self._cpp.Function('test', '{}& {}'.format(structName, self.ARGUMENT_NAME)):
        self.visit(node)
      function = functionIO.getvalue()
    with cpp.Struct(structName):
      for tensor in self._tensors:
        cpp('{}* {};'.format(self._arch.typename, tensor))
    cpp(function)

  def generic_visit(self, node, **kwargs):
    names = [self.visit(child) for child in node]
    resultName = self._maybeCreateTemporary(node, **kwargs)
    add = kwargs['add'] if 'add' in kwargs else False
    self._factory.create(node, resultName, names, add)
    return resultName
  
  def visit_Assign(self, node, **kwargs):
    resultName = node[0].name()
    for child in node:
      self.visit(child, resultName=resultName)
    return resultName
  
  def visit_Add(self, node, **kwargs):
    resultName = self._maybeCreateTemporary(node, **kwargs)
    add = False
    names = list()
    for child in node:
      names.append( self.visit(child, resultName=resultName, add=add) )
      add = True
    #self._factory.create(node, resultName, names, add)
    # TODO add or copy sole IndexedTensor
    return resultName
  
  def visit_IndexedTensor(self, node, **kwargs):
    self._tensors.add(node.name())
    return '{}.{}'.format(self.ARGUMENT_NAME, node.name())
  
  def _maybeCreateTemporary(self, node, **kwargs):
    resultName = kwargs['resultName'] if 'resultName' in kwargs else None
    if not resultName:      
      size = 1
      for s in node.indices.shape():
        size *= s
      resultName = 'tmp{}'.format(self._tmp)
      self._cpp('{} {}[{}];'.format(self._arch.typename, resultName, size))
      self._tmp += 1
    return resultName
