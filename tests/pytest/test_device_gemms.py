"""Device kernels that yateto generates itself, from GEMMs alone.

Without an exporter for the device, a device kernel is written with the GEMM,
fused-GEMM and copy-scale-add generators behind gemmforge and chainforge, and
nothing element-wise. These tests check yateto's side of that: which products
become GEMMs and how, which order a contraction is given so that its products
can, where gemmforge finds an operand that is a window into a tensor, and
which GEMMs run as one chain.

gemmforge is not a dependency of the test suite, so the GEMM generator is
replaced by one that records what it is asked to generate.
"""

import pytest

from yateto import Scalar, Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator, FusedGemmsBoundingBoxCostEstimator, ProductsAsGemms
from yateto.ast.indices import Range
from yateto.ast.node import FusedGEMMs
from yateto.codegen import gemm
from yateto.codegen.cache import RoutineCache
from yateto.codegen.common import BatchedOperationsAux
from yateto.codegen.datacache import DataCache
from yateto.codegen.visitor import OptimizedKernelGenerator
from yateto.gemm_configuration import GeneratorCollection
from yateto.generator import Kernel
from yateto.memory import DenseMemoryLayout, MemoryLayout


@pytest.fixture
def device():
  yield useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
  # layouts built by other tests align against the default again
  MemoryLayout.DEFAULT_ALIGNMENT_ARCH = None


def prepare(arch, statements, estimator=BoundingBoxCostEstimator, fused=False, productsAsGemms=True):
  kernel = Kernel('k', statements, target='gpu')
  kernel.prepareUntilUnitTest(arch)
  kernel.prepareUntilCodeGen(estimator, fused, productsAsGemms)
  return kernel


def lower(arch, statements, monkeypatch, **kwargs):
  """The GEMMs a device kernel is generated as."""
  seen = []

  class Recorder:
    def __init__(self, descr):
      self.descr = descr

    def generate(self, cpp, routineCache):
      return 0

  def generator(arch, descr, gemm_cfg, target, attrs=None):
    seen.append(descr)
    return Recorder(descr)

  monkeypatch.setattr(gemm, 'generator', generator)
  kernel = prepare(arch, statements, **kwargs)
  OptimizedKernelGenerator(arch, RoutineCache(), DataCache(), {}).generateKernelOutline(
    kernel.nonZeroFlops, kernel.cfg, GeneratorCollection([]), 'gpu')
  return seen


def sizes(descr):
  return tuple(r.size() for r in descr.mnk())


class TestContractionOrder:
  """An order that leaves a product no GEMM can do is searched for again.

  The shape of SeisSol's free-surface-gravity flux: a rank-zero factor, and
  two vectors that contract with a matrix each. Charged per thread, the search
  scales one of the matrices by the factor -- a product of a scalar and a
  matrix, which no GEMM of depth one is.
  """

  def statement(self):
    rho = Tensor('rho', ())
    P = Tensor('P', (56, 21))
    avg = Tensor('avg', (21,))
    sel = Tensor('sel', (9,))
    A = Tensor('A', (9, 9))
    Q = Tensor('Q', (56, 9))
    return Q['kp'] <= rho[''] * P['kn'] * avg['n'] * sel['o'] * A['op']

  def test_the_cheapest_order_scales_a_matrix(self, device):
    kernel = prepare(device, self.statement(), FusedGemmsBoundingBoxCostEstimator,
                     productsAsGemms=False)
    assert sum(ProductsAsGemms.unrepresentable(ast) for ast in kernel.ast) > 0

  def test_with_gemms_alone_it_does_not(self, device):
    kernel = prepare(device, self.statement(), FusedGemmsBoundingBoxCostEstimator)
    assert sum(ProductsAsGemms.unrepresentable(ast) for ast in kernel.ast) == 0

  def test_what_can_be_had_without_is_left_as_it_is(self, device):
    a = Tensor('a', (9, 9))
    b = Tensor('b', (9, 9))
    c = Tensor('c', (9, 9))
    D = Tensor('D', (9, 9))
    plain = prepare(device, D['ij'] <= a['ik'] * b['kl'] * c['lj'], productsAsGemms=False)
    gemms = prepare(device, D['ij'] <= a['ik'] * b['kl'] * c['lj'])
    assert [str(ast) for ast in plain.ast] == [str(ast) for ast in gemms.ast]
