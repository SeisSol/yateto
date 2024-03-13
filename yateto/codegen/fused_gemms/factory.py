import importlib.util
import pkg_resources
from .external_generator import FusedGemms

class Description(object):
  def __init__(self, node, result, arguments, add, scalar):
    self.node = node
    self.result = result
    self.args = arguments
    self.add = add
    self.scalar = scalar
    self._inter_counter: int = 0

  def __iter__(self):
    self._inter_counter = 0
    return self

  def __next__(self):
    index = self._inter_counter
    self._inter_counter += 1
    try:
      return (self.node.get_child(index),
              self.args[index],
              self.add[index],
              self.scalar[index])
    except IndexError:
      raise StopIteration


def generator(arch, descr, target):
  if target == 'gpu':
    return FusedGemms(arch, descr)
  else:
    raise NotImplementedError(f'no implementation found for {target} target')
