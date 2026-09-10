import importlib.util

from .tinytc import FusedGemmsTinytc
from ...gemm_configuration import GemmForge, tinytc

class Description(object):
  """The contractions a chaining backend takes together.

  Each of them is a statement in its own right, and states everything a matrix
  product is stated with: a destination, two operands with their index maps,
  whether either is read the other way round, a factor and an accumulation.
  What the chain writes is what the last of them writes; the destinations
  before it are where the chain keeps what it has computed so far.
  """

  def __init__(self, statements):
    self.statements = list(statements)

  def __iter__(self):
    return iter(self.statements)

  def __len__(self):
    return len(self.statements)

  @property
  def last(self):
    return self.statements[-1]

  @property
  def datatype(self):
    return self.last.result.datatype

class GBSpec:
  gb_spec = None
  @classmethod
  def load(cls):
    if cls.gb_spec is None:
      cls.gb_spec = importlib.util.find_spec('chainforge')
    return cls.gb_spec

def available(gemm_cfg, target):
  """Whether anything here takes several contractions together.

  Asked of the configuration rather than settled by a flag from outside:
  which backend writes a statement is a code generation decision, and so is
  whether there is one that would rather have the whole chain.
  """
  if target != 'gpu':
    return False
  if any(isinstance(tool, tinytc) for tool in gemm_cfg.gemmTools):
    return True
  return any(isinstance(tool, GemmForge) for tool in gemm_cfg.gemmTools) \
      and bool(GBSpec.load())

def generator(arch, descr, gemm_cfg, target, attrs=None):
  if target == 'gpu':
      hasTinytc = any([isinstance(tool, tinytc) for tool in gemm_cfg.gemmTools])
      if hasTinytc:
          return FusedGemmsTinytc(arch, descr)
      elif GBSpec.load():
          from .external_generator import FusedGemms
          return FusedGemms(arch, descr, attrs)
  raise NotImplementedError(f'no implementation found for {target} target')
