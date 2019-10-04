from numpy import ndindex, arange, float64, add, einsum
import math
import collections
import itertools
import re
import os.path
from .node import Op
from .indices import LoGCost
from .log import LoG
from functools import reduce

# Optional modules
import importlib.util
mplSpec = importlib.util.find_spec('matplotlib')
pltSpec = importlib.util.find_spec('matplotlib.pylab') if mplSpec else None
colorsSpec = importlib.util.find_spec('matplotlib.colors') if mplSpec else None
try:
  if pltSpec:
    plt = pltSpec.loader.load_module()
except:
  print('An exception occured trying to load matplotlib. This can be ignored in most cases')
  plt = None
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

class CachedVisitor(Visitor):
  def __init__(self):
    self._cache = dict()

  def visit(self, node, **kwargs):
    if node in self._cache:
      return self._cache[node]
    result = super().visit(node, **kwargs)
    self._cache[node] = result
    return result

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
  def __init__(self, useAvailable):
    self._useAvailable = useAvailable

  def generic_visit(self, node):
    if self._useAvailable:
      spps = [child.eqspp() if child.eqspp() is not None else self.visit(child) for child in node]
    else:
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
  
  def variantsFixedRootPermutation(self, node, fixedPerm, permutationVariants):
    variants = dict()
    minCost = LoGCost.addIdentity()
    minInd = list()
    for child in node:
      childMinCost = LoGCost()
      childMinInd = None
      for ind in sorted(permutationVariants[child]):
        transpose = int(fixedPerm != ind)
        cost = permutationVariants[child][ind]._cost + LoGCost(0,transpose,0,0)
        if cost < childMinCost:
          childMinCost = cost
          childMinInd = ind
      assert childMinInd is not None
      minCost = minCost + childMinCost
      minInd.append(childMinInd)
    assert minInd is not None
    variants[fixedPerm] = self.Variant(minCost, minInd)
    return variants

  def allPermutationsNoCostBinaryOp(self, node):
    permutationVariants = self.findVariants(node)
    lV = permutationVariants[node.leftTerm()]
    rV = permutationVariants[node.rightTerm()]
    minCost = LoGCost()
    minAind = None
    minBind = None
    for Aind in sorted(lV):
      for Bind in sorted(rV):
        cost = lV[Aind]._cost + rV[Bind]._cost
        if cost < minCost:
          minCost = cost
          minAind = Aind
          minBind = Bind
    assert minAind is not None and minBind is not None
    iterator = itertools.permutations(node.indices)
    permutationVariants[node] = {''.join(Cs): self.Variant(minCost, [minAind, minBind]) for Cs in iterator}
    return permutationVariants

  def generic_visit(self, node):
    permutationVariants = self.findVariants(node)
    variants = self.variantsFixedRootPermutation(node, str(node.indices), permutationVariants)
    assert variants, 'Could not find implementation for {}.'.format(type(node))
    permutationVariants[node] = variants
    return permutationVariants

  def visit_Add(self, node):
    permutationVariants = self.findVariants(node)
    iterator = itertools.permutations(node.indices)
    variants = dict()
    for Cs in iterator:
      variants.update( self.variantsFixedRootPermutation(node, ''.join(Cs), permutationVariants) )
    assert variants, 'Could not find implementation for Add.'
    permutationVariants[node] = variants
    return permutationVariants

  def visit_ScalarMultiplication(self, node):
    permutationVariants = self.visit(node.term())
    permutationVariants[node] = {key: self.Variant(variant._cost, [key]) for key,variant in permutationVariants[node.term()].items()}
    return permutationVariants
  
  def visit_Product(self, node):
    return self.allPermutationsNoCostBinaryOp(node)
    
  def visit_IndexSum(self, node):
    permutationVariants = self.findVariants(node)
    tV = permutationVariants[node.term()]
    minCost = LoGCost()
    minTind = None
    for Tind in sorted(tV):
        cost = tV[Tind]._cost
        if cost < minCost:
            minCost = cost
            minTind = Tind
    assert minTind is not None
    iterator = itertools.permutations(node.indices)
    permutationVariants[node] = {''.join(Cs): self.Variant(minCost, [minTind]) for Cs in iterator}
    return permutationVariants

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
          if log is not None:
            cost = log.cost() + lV[Aind]._cost + rV[Bind]._cost
            if cost < minCost:
              minCost = cost
              minAind = Aind
              minBind = Bind
      if minAind is not None and minBind is not None:
        variants[C] = self.Variant(minCost, [minAind, minBind])
    assert variants, 'Could not find implementation for Contraction.'
    permutationVariants[node] = variants
    return permutationVariants

class PrintEquivalentSparsityPatterns(Visitor):
  def __init__(self, directory):
    if not (pltSpec and colorsSpec):
      raise NotImplementedError('Missing modules matplotlib')
    self._directory = directory
    self._cmap = colors.ListedColormap(['white', 'black'])
    self._norm = colors.BoundaryNorm([0.0, 0.5, 1.0], 2, clip=True)
  
  def generic_visit(self, node):
    nameFun = getattr(node, 'name', None)
    name = nameFun() if nameFun else '_result'
    baseDirectory = self._directory
    counter = 0
    for child in node:
      if len(child) > 0:
        self._directory = os.path.join(baseDirectory, '{}_{}'.format(counter, type(child).__name__))
      else:
        self._directory = baseDirectory
      os.makedirs(self._directory, exist_ok=True)
      self.visit(child)
      counter = counter + 1
    fileName = os.path.join(baseDirectory, name + '.pdf')
    file_name_mm = os.path.join(baseDirectory, name + '.mtx')
    eqspp = node.eqspp()
    pattern = eqspp.as_ndarray()
    with open(file_name_mm, 'w') as f:
      f.write('%%TensorMarket tensor coordinate real general\n')
      nzs = pattern.nonzero()
      if nzs:
        f.write('{} {}\n'.format(' '.join([str(s) for s in pattern.shape]), len(nzs[0])))
        for idx in zip(*nzs):
          f.write('{} {}\n'.format(' '.join([str(i) for i in idx]), float(pattern[idx])))
    nSubplots = 1
    for dim in range(2, eqspp.ndim):
      nSubplots *= eqspp.shape[dim]
    nrows = math.ceil(math.sqrt(nSubplots))
    ncols = math.ceil(nSubplots / nrows)
    fig, axs = plt.subplots(nrows, ncols)
    if ncols > 1:
      axs = [y for x in axs for y in x]
    if nSubplots == 1:
      axs.imshow(pattern.astype(bool), cmap=self._cmap, norm=self._norm)
    else:
      nSubplot = 0
      for index in ndindex(*list(eqspp.shape)[2:]):
        sl = pattern[(slice(None, None), slice(None, None)) + index]
        axs[nSubplot].imshow(sl.astype(bool), cmap=self._cmap, norm=self._norm)
        axs[nSubplot].set_title('(:,:,{})'.format(','.join([str(i) for i in index])), y=1.2)
        nSubplot = nSubplot + 1
    #plt.setp(axs, xticks=arange(eqspp.shape[1]), yticks=arange(eqspp.shape[0]))
    fig.tight_layout()
    fig.savefig(fileName, bbox_inches='tight')
    plt.close()
    self._directory = baseDirectory


class FindPrefetchCapabilities(Visitor):
  def generic_visit(self, node):
    sizes = collections.OrderedDict()
    for child in node:
      sizes.update( self.visit(child) )
    return sizes

  def visit_LoopOverGEMM(self, node):
    sizes = self.generic_visit(node)
    sizes[node] = node.memoryLayout().requiredReals()
    return sizes

class ComputeConstantExpression(Visitor):
  def __init__(self, dtype = float64):
    self._dtype = dtype

  def generic_visit(self, node):
    return [self.visit(child) for child in node]

  def visit_Einsum(self, node):
    terms = self.generic_visit(node)
    childIndices = [child.indices for child in node]
    assert None not in childIndices and node.indices is not None, 'Use DeduceIndices before {}.'.format(self.__class__.__name__)
    einsumDescription = ','.join([indices.tostring() for indices in childIndices])
    einsumDescription = '{}->{}'.format(einsumDescription, node.indices.tostring())
    return einsum(einsumDescription, *terms)

  def visit_Add(self, node):
    terms = self.generic_visit(node)
    assert len(terms) > 1
    permute = lambda indices, tensor: tensor.transpose(tuple([indices.find(idx) for idx in node.indices]))
    return reduce(add, [permute(child.indices, terms[i]) for i,child in enumerate(node)])

  def visit_ScalarMultiplication(self, node):
    assert node.is_constant() is not None, '{} may only be used when all involved scalars are constant.'.format(self.__class__.__name__)
    terms = self.generic_visit(node)
    assert len(terms) == 1
    return node.scalar() * terms[0]

  def visit_IndexedTensor(self, node):
    term = node.tensor.values_as_ndarray(self._dtype)
    assert term is not None, '{} may only be used when all involved tensors are constant.'.format(self.__class__.__name__)
    return term

class ComputeIndexSet(CachedVisitor):
  def generic_visit(self, node):
    union = set()
    for child in node:
      union = union | super().visit(child)
    return union

  def visit_IndexedTensor(self, node):
    return set(node.indices)
