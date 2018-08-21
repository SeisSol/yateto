from numpy import einsum, ndindex
from .node import Op
import re
import os.path

# Optional modules
import importlib.util
pltSpec = importlib.util.find_spec('matplotlib.pylab')
scipyspSpec = importlib.util.find_spec('scipy.sparse')
if pltSpec:
  plt = pltSpec.loader.load_module()
if scipyspSpec:
  scipysp = scipyspSpec.loader.load_module()

# Similar as ast.NodeVisitor
class Visitor(object):
  def visit(self, node):
    method = 'visit_' + node.__class__.__name__
    visitor = getattr(self, method, self.generic_visit)
    return visitor(node)
  
  def generic_visit(self, node):
    for child in node:
      self.visit(child)

def addIndent(string, indent):
  return '\n'.join([indent + line for line in string.splitlines()])

class PrettyPrinter(Visitor):
  def __init__(self):
    self._indent = 0
    
  def generic_visit(self, node):
    print('  ' * self._indent + str(node))
    self._indent = self._indent + 1
    super().generic_visit(node)
    self._indent = self._indent - 1

class ComputeSparsityPattern(Visitor):
  def generic_visit(self, node):
    super().generic_visit(node)
    return node.eqspp()

  def visit_IndexSum(self, node):
    eqspp = self.visit(node.term())
    einsumDescription = '{}->{}'.format(node.term().indices.tostring(), node.indices.tostring())
    return einsum(einsumDescription, eqspp)
  
  def visit_Product(self, node):
    leftEqspp = self.visit(node.leftTerm())
    rightEqspp = self.visit(node.rightTerm())
    einsumDescription = '{},{}->{}'.format(node.leftTerm().indices.tostring(), node.rightTerm().indices.tostring(), node.indices.tostring())
    return einsum(einsumDescription, leftEqspp, rightEqspp)
  
  def visit_Contraction(self, node):
    leftEqspp = self.visit(node.leftTerm())
    rightEqspp = self.visit(node.rightTerm())
    einsumDescription = '{},{}->{}'.format(node.leftTerm().indices.tostring(), node.rightTerm().indices.tostring(), node.indices.tostring())
    return einsum(einsumDescription, leftEqspp, rightEqspp)

class PrintEquivalentSparsityPatterns(Visitor):
  def __init__(self, directory):
    if not (pltSpec and scipyspSpec):
      raise NotImplementedError('Missing modules matplotlib and scipy')
    self._directory = directory
    self._prefix = ''
  
  def generic_visit(self, node):
    oldPrefix = self._prefix
    nameFun = getattr(node, 'name', None)
    name = re.sub('\]', '', re.sub('\[', '_', str(node)))
    basePrefix = '{}--{}'.format(self._prefix, name) if len(self._prefix) > 0 else name
    counter = 0
    for child in node:
      self._prefix = basePrefix + '.' + str(counter)
      self.visit(child)
      counter = counter + 1
    fileName = os.path.join(self._directory, basePrefix + '.pdf')
    eqspp = node.eqspp()
    nSubplots = 1
    for dim in range(2, eqspp.ndim):
      nSubplots *= eqspp.shape[dim]
    fig, axs = plt.subplots(nSubplots)
    if nSubplots == 1:
      plt.spy(eqspp)
    else:
      nSubplot = 0
      for index in ndindex(*list(eqspp.shape)[2:]):
        sl = eqspp[(slice(None, None), slice(None, None)) + index]
        axs[nSubplot].spy(sl)
        axs[nSubplot].set_title('(:,:,{})'.format(','.join([str(i) for i in index])))
        nSubplot = nSubplot + 1
    plt.savefig(fileName, bbox_inches='tight')
    plt.close()
    self._prefix = oldPrefix
