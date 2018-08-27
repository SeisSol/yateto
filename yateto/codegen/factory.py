class Factory(object):
  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)
  
  def generic_create(self, node, *args):
    #~ raise NotImplementedError
    pass

class KernelFactory(Factory):
  def __init__(self, cpp):
    self._cpp = cpp

  def create_LoopOverGEMM(self, node, resultName, argNames):
    assert len(argNames) == 2
    loopIndices = node.loopIndices()
    if len(loopIndices) > 0:
      for index in loopIndices:
        with self._cpp.For('int {0} = 0; {0} < {1}; ++{0}'.format(index, loopIndices.indexSize(index))):
          self._cpp('gemm({}, {}, {})'.format(argNames[0], argNames[1], resultName))
    else:
      self._cpp('gemm({}, {}, {})'.format(argNames[0], argNames[1], resultName))
  
  def create_IndexSum(self, node, resultName, argNames):
    assert len(argNames) == 1
    sumIndex = node.sumIndex()
    with self._cpp.For('int {0} = 0; {0} < {1}; ++{0}'.format(sumIndex, sumIndex.shape()[0])):
      self._cpp('{1}[{0}] = {2}[{0}];'.format(sumIndex, resultName, argNames[0]))
    

'''
class LoopOverGEMM(BinOp):
  def __init__(self, indices, aTerm, bTerm, m, n, k):
    super().__init__(aTerm, bTerm)
    self.indices = indices
    self._m = m
    self._n = n
    self._k = k
    self._Atrans = aTerm.indices.find(m[0]) > aTerm.indices.find(k[0])
    self._Btrans = bTerm.indices.find(k[0]) > bTerm.indices.find(n[0])
  
  def computeSparsityPattern(self, *spps):
    return _productContractionLoGSparsityPattern(self, *spps)
  
  def cost(self):
    A = self.leftTerm().indices
    B = self.rightTerm().indices
    AstrideOne = (A.find(self._m[0]) == 0) if not self._Atrans else (A.find(self._k[0]) == 0)
    BstrideOne = (B.find(self._k[0]) == 0) if not self._Btrans else (B.find(self._n[0]) == 0)
    cost = LoGCost(int(not AstrideOne) + int(not BstrideOne), int(self._Atrans), int(self._Btrans), len(self._m) + len(self._n) + len(self._k))
    return cost

  @staticmethod
  def indexString(name, subset, indices, transpose=False):
    indexStr = ''.join([i if i in subset else ':' for i in indices])
    matrixStr = '{}_{}'.format(name, indexStr)
    return '({})\''.format(matrixStr) if transpose else matrixStr
  
  def __str__(self):
    Astr = self.indexString('A', self._m + self._k, self.leftTerm().indices, self._Atrans)
    Bstr = self.indexString('B', self._k + self._n, self.rightTerm().indices, self._Btrans)
    Cstr = self.indexString('C', self._m + self._n, self.indices)
    return '{} [{}]: {} = {} {}'.format(type(self).__name__, self.indices, Cstr, Astr, Bstr)
    '''
