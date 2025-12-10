from ..common import *
from .generic import Generic
from ...gemm_configuration import tinytc
from .tinytc import CopyScaleAddTinytc

import importlib
gf_spec = importlib.util.find_spec('gemmforge')
try:
  if gf_spec:
    gf = gf_spec.loader.load_module()
  from .csa_gen import CopyScaleAddGenerator
except RuntimeError as err:
  raise err
except:
  raise ('gemmforge module is not found. You can install it with pip3. e.g., pip3 install gemmforge')


class Description(object):
  def __init__(self, alpha, beta, result: IndexedTensorDescription, term: IndexedTensorDescription):
    self.alpha = alpha
    self.beta = beta
    self.result = result
    self.term = term
    
    assert self.alpha != 0.0, 'copyscaleadd does not support alpha=0.0 at the moment.'
    assert self.beta == 1.0 or self.beta == 0.0, 'copyscaleadd supports only beta=0.0 or beta=1.0 at the moment.'

    rA = loopRanges(self.term, self.term.indices)
    rB = loopRanges(self.result, self.result.indices)
    assert testLoopRangesAContainedInB(rA, rB)
    assert self.term.indices <= self.result.indices

    # restrict ranges to rA where possible;
    # broadcast to all other ranges

    rAB = rA
    for idx in rB:
      if idx not in rA:
        rAB[idx] = rB[idx]
    
    self.loopRanges = rAB


def generator(arch, descr, gemm_cfg, target):
  if target == 'gpu':
      hasTinytc = any([isinstance(tool, tinytc) for tool in gemm_cfg.gemmTools])
      if hasTinytc:
          return CopyScaleAddTinytc(arch, descr)
      elif gf_spec:
          return CopyScaleAddGenerator(arch, descr)
      else:
          raise NotImplementedError(f'no implementation found for {target} target')
  return Generic(arch, descr)
