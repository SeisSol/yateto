from .indices import Indices

class Node(object):
  def __init__(self):
    self.indices = None
    self._children = []
    self._eqspp = None
  
  def size(self):
    return self.indices.size()

  def __iter__(self):
    return iter(self._children)
  
  def __getitem__(self, key):
    return self._children[key]
  
  def setChildren(self, children):
    self._children = children

  def eqspp(self):
    return self._eqspp
  
  def setEqspp(self, spp):
    self._eqspp = spp
    
  def setIndexPermutation(self, indices):
    self.indices.permute(indices)

  def _binOp(self, other, opType):
    if isinstance(self, opType):
      if isinstance(other, opType):
        self._children.extend(other._children)
      else:
        self._children.append(other)
      return self
    elif isinstance(other, opType):
      other._children.insert(0, self)
      return other
    return opType(self, other)

  def __mul__(self, other):
    return self._binOp(other, Einsum)
  
  def __add__(self, other):
    return self._binOp(other, Add)
    
  def __le__(self, other):
    return Assign(self, other)

class IndexedTensor(Node):
  def __init__(self, tensor, indexNames):
    super().__init__()
    self.tensor = tensor
    self.indices = Indices(indexNames, self.tensor.shape())
  
  def spp(self):
    return self.tensor.spp()

  def __str__(self):
    return '{}[{}]'.format(self.tensor.name(), str(self.indices))

class Op(Node):
  def __init__(self, *args):
    super().__init__()
    self._children = list(args)
  
  def __str__(self):
    return '{}[{}]'.format(type(self).__name__, self.indices if self.indices != None else '<not deduced>')

class Assign(Op):
  def setChildren(self, children):
    if len(children) != 2:
      raise ValueError('Assign node must have exactly 2 children')
    if not isinstance(children[0], IndexedTensor):
      raise ValueError('First child of Assign node must be an IndexedTensor: ' + str(children[0]))
    
    super().setChildren(children)

class Einsum(Op):
  pass
    
class Add(Op):
  pass

class Product(Op):
  def __init__(self, lTerm, rTerm, target_indices):
    super().__init__(lTerm, rTerm)
    if target_indices.firstIndex() <= rTerm.indices:
      lTerm, rTerm = rTerm, lTerm

    K = lTerm.indices & rTerm.indices
    assert lTerm.indices.subShape(K) == rTerm.indices.subShape(K)

    self.indices = lTerm.indices.merged(rTerm.indices - K)

    self._cost = 1
    for size in self.indices.shape():
      self._cost *= size
    for child in self._children:
      self._cost += getattr(child, '_cost', 0)
  
  def leftTerm(self):
    return self._children[0]
  
  def rightTerm(self):
    return self._children[1]
  
  def __str__(self):
    return '{} [{}] ({})'.format(type(self).__name__, self.indices, self._cost)

  
class IndexSum(Op):
  def __init__(self, term, sumIndex):
    super().__init__(term)
    self.indices = term.indices - set([sumIndex])
    self._sumIndex = sumIndex
    self._cost = term.indices.size()[sumIndex]
    for size in self.indices.shape():
      self._cost *= size
    self._cost += getattr(term, '_cost', 0)
  
  def sumIndex(self):
    return self._sumIndex
  
  def term(self):
    return self._children[0]

  def __str__(self):
    return '{}_{} [{}] ({})'.format(type(self).__name__, self._sumIndex, self.indices, self._cost)

class Contraction(Op):
  def __init__(self, indices, lTerm, rTerm, sumIndices):
    super().__init__(lTerm, rTerm)
    li = lTerm.indices - sumIndices
    lr = rTerm.indices - sumIndices
    self.indices = li.merged(lr)
    self.setIndexPermutation(indices)
  
  def leftTerm(self):
    return self._children[0]
  
  def rightTerm(self):
    return self._children[1]
  
  def __str__(self):
    return '{} [{}]'.format(type(self).__name__, self.indices)