import importlib.util

from .tinytc import FusedGemmsTinytc
from ...gemm_configuration import tinytc

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
    args_index = 3 * index
    self._inter_counter += 1
    try:
      return (self.node.get_child(index),
              self.args[args_index:args_index + 3],
              self.add[index],
              self.scalar[index])
    except IndexError:
      raise StopIteration

class GBSpec:
  gb_spec = None
  @classmethod
  def load(cls):
    if cls.gb_spec is None:
      cls.gb_spec = importlib.util.find_spec('chainforge')
    return cls.gb_spec

def generator(arch, descr, gemm_cfg, target):
  if target == 'gpu':
      hasTinytc = any([isinstance(tool, tinytc) for tool in gemm_cfg.gemmTools])
      if hasTinytc:
          return FusedGemmsTinytc(arch, descr)
      elif GBSpec.load():
          from .external_generator import FusedGemms
          return FusedGemms(arch, descr)
  raise NotImplementedError(f'no implementation found for {target} target')
