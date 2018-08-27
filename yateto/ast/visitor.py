from numpy import arange, einsum, ndindex
import re
import os.path
from .node import Op

# Optional modules
import importlib.util
pltSpec = importlib.util.find_spec('matplotlib.pylab')
colorsSpec = importlib.util.find_spec('matplotlib.colors')
scipyspSpec = importlib.util.find_spec('scipy.sparse')
if pltSpec:
  plt = pltSpec.loader.load_module()
if colorsSpec:
  colors = colorsSpec.loader.load_module()
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
    spps = [self.visit(child) for child in node]
    return node.computeSparsityPattern(*spps)
  
  def visit_IndexedTensor(self, node):
    return node.eqspp()

class PrintEquivalentSparsityPatterns(Visitor):
  def __init__(self, directory):
    if not (pltSpec and colorsSpec and scipyspSpec):
      raise NotImplementedError('Missing modules matplotlib and scipy')
    self._directory = directory
    self._prefix = ''
    self._cmap = colors.ListedColormap(['white', 'black'])
    self._norm = colors.BoundaryNorm([0.0, 0.5, 1.0], 2, clip=True)
  
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
      axs.imshow(eqspp.astype(bool), cmap=self._cmap, norm=self._norm)
    else:
      nSubplot = 0
      for index in ndindex(*list(eqspp.shape)[2:]):
        sl = eqspp[(slice(None, None), slice(None, None)) + index]
        axs[nSubplot].imshow(sl.astype(bool), cmap=self._cmap, norm=self._norm)
        axs[nSubplot].set_title('(:,:,{})'.format(','.join([str(i) for i in index])), y=1.2)
        nSubplot = nSubplot + 1
    #plt.setp(axs, xticks=arange(eqspp.shape[1]), yticks=arange(eqspp.shape[0]))
    fig.set_size_inches(nSubplots*eqspp.shape[0] / 3.0, eqspp.shape[1] / 3.0)
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches='tight')
    plt.close()
    self._prefix = oldPrefix
