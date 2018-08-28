from ..common import IndexedTensorDescription
from .generic import Generic

class Description(object):
  def __init__(self, add: bool, result: IndexedTensorDescription, term: IndexedTensorDescription):
    self.add = add
    self.result = result
    self.term = term
    

def generator(arch, descr):
  return Generic(arch, descr)

