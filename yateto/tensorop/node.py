class AbstractTensor(object):
  def __init__(self):
    self.indices = None

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

class Multiply(Op):
  pass
    
class Sum(Op):
  pass
