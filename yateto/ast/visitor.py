from numpy import arange, einsum, ndindex
import itertools
import re
import os.path
from .node import Op
from .indices import LoGCost
from .log import LoG

# Optional modules
import importlib.util
mplSpec = importlib.util.find_spec('matplotlib')
pltSpec = importlib.util.find_spec('matplotlib.pylab') if mplSpec else None
colorsSpec = importlib.util.find_spec('matplotlib.colors') if mplSpec else None
if pltSpec:
  plt = pltSpec.loader.load_module()
if colorsSpec:
  colors = colorsSpec.loader.load_module()

# Similar as ast.NodeVisitor
class Visitor(object):
  def visit(self, node, **kwargs):
    method = 'visit_' + node.__class__.__name__
    visitor = getattr(self, method, self.generic_visit)
    return visitor(node, **kwargs)
  
  def generic_visit(self, node, **kwargs):
    for child in node:
      self.visit(child, **kwargs)

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

class ComputeOptimalFlopCount(Visitor):
  def generic_visit(self, node):
    childFlops = 0
    for child in node:
      childFlops += self.visit(child)
    return childFlops + node.nonZeroFlops()

class FindTensors(Visitor):
  def generic_visit(self, node):
    tensors = dict()
    for child in node:
      tensors.update( self.visit(child) )
    return tensors

  def visit_IndexedTensor(self, node):
    return {node.name(): node.tensor}

class FindIndexPermutations(Visitor):
  class Variant(object):
    def __init__(self, cost, choices):
      self._cost = cost
      self._choices = choices
  
  def findVariants(self, node):
    permutationVariants = dict()
    for child in node:
      permutationVariants.update( self.visit(child) )
    return permutationVariants
  
  def generic_visit(self, node):
    permutationVariants = self.findVariants(node)
    choices = list()
    for child in node:
      choices.append( str(child.indices) )
    variants = {str(node.indices): self.Variant(LoGCost.addIdentity(), choices)}
    permutationVariants[node] = variants
    return permutationVariants
  
  def allPermutations(self, node, inheritIndices):
    permutationVariants = self.findVariants(node)
    iterator = itertools.permutations(node.indices)
    if inheritIndices:
      variants = {''.join(Cs): self.Variant(LoGCost.addIdentity(), [''.join(Cs)] * len(node)) for Cs in iterator}
    else:
      choices = [str(child.indices) for child in node]
      variants = {''.join(Cs): self.Variant(LoGCost.addIdentity(), choices) for Cs in iterator}
    permutationVariants[node] = variants
    return permutationVariants
  
  def visit_Add(self, node):
    if any([child.fixedIndexPermutation() for child in node]):
      return self.generic_visit(node)
    return self.allPermutations(node, True)
  
  def visit_Product(self, node):
    return self.allPermutations(node, False)
    
  def visit_IndexSum(self, node):
    return self.allPermutations(node, False)

  def visit_Contraction(self, node):
    permutationVariants = self.findVariants(node)
    
    variants = dict()
    iterator = itertools.permutations(node.indices)
    for Cs in iterator:
      C = ''.join(Cs)
      minCost = LoGCost()
      minAind = None
      minBind = None
      lV = permutationVariants[node.leftTerm()]
      rV = permutationVariants[node.rightTerm()]
      for Aind in sorted(lV):
        for Bind in sorted(rV):
          log = LoG(node, Aind, Bind, C)
          cost = log.cost() + lV[Aind]._cost + rV[Bind]._cost
          if cost < minCost:
            minCost = cost
            minAind = Aind
            minBind = Bind
      if minAind is not None and minBind is not None:
        variants[C] = self.Variant(minCost, [minAind, minBind])
    assert variants, 'Could not find implementation for Contraction. (Note: Matrix-Vector multiplication currently not supported.)'
    permutationVariants[node] = variants
    return permutationVariants

class PrintEquivalentSparsityPatterns(Visitor):
  def __init__(self, directory):
    if not (pltSpec and colorsSpec):
      raise NotImplementedError('Missing modules matplotlib')
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
