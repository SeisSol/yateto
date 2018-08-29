from ..common import TensorDescription
from .generic import Generic

class Description(object):
  def __init__(self, alpha, beta, result: TensorDescription, term: TensorDescription):
    self.alpha = alpha
    self.beta = beta
    self.result = result
    self.term = term
    

def generator(arch, descr):
  return Generic(arch, descr)

