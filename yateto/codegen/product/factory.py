from ..common import IndexedTensorDescription
from .generic import Generic

class Description(object):
  def __init__(self, add: bool, result: IndexedTensorDescription, leftTerm: IndexedTensorDescription, rightTerm: IndexedTensorDescription):
    self.add = add
    self.result = result
    self.leftTerm = leftTerm
    self.rightTerm = rightTerm
    

def generator(arch, descr):
  return Generic(arch, descr)

