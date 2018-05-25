class Node(object):
  def __mul__(self, other):
    return Contract(self, other)
  
  def __add__(self, other):
    return Add(self, other)
    
  def __le__(self, other):
    return Assign(self, other)

class AbstractTensor(Node):
  def __init__(self):
    self.indices = None
    self._eqspp = None
  
  def size(self):
    return self.indices.size()
    
  def eqspp(self):
    return self._eqspp
  
  def setEqspp(self, spp):
    self._eqspp = spp

class Op(AbstractTensor):
  def __init__(self, *args):
    super().__init__()
    self._children = list(args)

  def __iter__(self):
    return iter(self._children)
  
  def __getitem__(self, key):
    return self._children[key]
  
  def setChildren(self, children):
    self._children = children
  
  def __str__(self):
    return '{}[{}]'.format(type(self).__name__, self.indices if self.indices != None else '<not deduced>')

class Contract(Op):
  pass
    
class Add(Op):
  pass

class Assign(Op):
  def setChildren(self, children):
    if len(children) != 2:
      raise ValueError('Assign node must have exactly 2 children')
    if not isinstance(children[0], IndexedTensor):
      raise ValueError('First child of Assign node must be an IndexedTensor: ' + str(children[0]))
    
    super().setChildren(children)

class IndexedTensor(AbstractTensor):
  def __init__(self, tensor, indexNames):
    self.tensor = tensor
    self.indices = Indices(indexNames, self.tensor.shape())
    self._eqspp = None
  
  def spp(self):
    return self.tensor.spp()

  def __str__(self):
    return '{}[{}]'.format(self.tensor.name(), str(self.indices))

class Indices(object):
  def __init__(self, indexNames = '', shape = ()):
    self._indices = tuple(indexNames)
    self._size = dict()
    
    assert len(self._indices) == len(shape), 'Indices {} do not match tensor shape {}.'.format(str(self), shape)

    for i, size in enumerate(shape):
      index = self._indices[i]
      if index in self._size and self._size[index] != size:
        raise ValueError('Repeated indices may only be used on same sized dimensions. {} {}'.format(indexNames, shape)) 
      self._size[ self._indices[i] ] = size
  
  def tostring(self):
    return ''.join(self._indices)

  def shape(self):
    return self.subShape(self._indices)
  
  def subShape(self, indexNames):
    return tuple([self._size[index] for index in indexNames])
  
  def __eq__(self, other):
    return other != None and self._indices == other._indices and self._size == other._size
    
  def __ne__(self, other):
    return other == None or self._indices != other._indices or self._size != other._size
  
  def __iter__(self):
    return iter(self._indices)
  
  def __and__(self, other):
    return set(self) & set(other)
  
  def __rand__(self, other):
    return self & other
    
  def __le__(self, other):
    indexNamesContained = set(self._indices) <= set(other._indices)
    return indexNamesContained and all([self._size[index] == other._size[index] for index in self._indices])
  
  def __sub__(self, other):
    indexNames = [index for index in self._indices if index not in other]
    return Indices(indexNames, self.subShape(indexNames))

  def merged(self, other):
    indexNames = self._indices + other._indices
    shape = self.subShape(self._indices) + other.subShape(other._indices)
    return Indices(indexNames, shape)
    
  def sorted(self):
    indexNames = sorted(self._indices)
    return Indices(indexNames, self.subShape(indexNames))
  
  def __str__(self):
    return self.tostring()
    
  def __repr__(self):  
    return '({})'.format(','.join(['{}={}'.format(index, self._size[index]) for index in self._indices]))
  
  def size(self):
    return self._size
