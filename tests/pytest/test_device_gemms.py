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


class TestProductsAsGemms:
  """A product the device has no element-wise generator for is a GEMM with K = 1."""

  N, M = 6, 5

  def test_an_outer_product_is_a_gemm(self, device, monkeypatch):
    a = Tensor('a', (self.N,))
    b = Tensor('b', (self.M,))
    C = Tensor('C', (self.N, self.M))
    [descr] = lower(device, C['kp'] <= a['k'] * b['p'], monkeypatch)
    assert sizes(descr) == (self.N, self.M, 1)
    assert not descr.transA and not descr.transB

  def test_it_is_written_the_way_the_result_is_indexed(self, device, monkeypatch):
    a = Tensor('a', (self.N,))
    b = Tensor('b', (self.M,))
    C = Tensor('C', (self.M, self.N))
    [descr] = lower(device, C['pk'] <= a['k'] * b['p'], monkeypatch)
    assert sizes(descr) == (self.M, self.N, 1)
    assert descr.leftTerm.name == 'b' and descr.rightTerm.name == 'a'

  def test_a_vector_scaled_by_a_rank_zero_tensor_is_one(self, device, monkeypatch):
    s = Tensor('s', ())
    v = Tensor('v', (self.N,))
    C = Tensor('C', (self.N,))
    [descr] = lower(device, C['k'] <= s[''] * v['k'], monkeypatch)
    assert sizes(descr) == (self.N, 1, 1)
    assert descr.leftTerm.name == 'v' and descr.rightTerm.name == 's'

  def test_an_index_of_extent_one_is_none(self, device, monkeypatch):
    s = Tensor('s', ())
    v = Tensor('v', (self.N, 1))
    C = Tensor('C', (self.N, 1))
    [descr] = lower(device, C['kq'] <= s[''] * v['kq'], monkeypatch)
    assert sizes(descr) == (self.N, 1, 1)

  def test_a_row_of_a_matrix_is_read_transposed(self, device, monkeypatch):
    # the row is a vector whose entries are a column apart
    s = Tensor('s', ())
    X = Tensor('X', (self.N, self.M))
    C = Tensor('C', (1, self.M))
    [descr] = lower(device, C['ij'] <= s[''] * X['ij'].subslice('i', 0, 1), monkeypatch)
    assert descr.transA and descr.leftTerm.name == 'X'
    assert descr.leftTerm.memoryLayout.stridei(1) == self.N
    # and gemmforge takes the extent of the first dimension for that distance
    bbox, _ = BatchedOperationsAux.forge_region(descr.leftTerm.memoryLayout, descr.mnk()[::2], True)
    assert bbox[0].stop == self.N
    assert sizes(descr) == (self.M, 1, 1)

  def test_a_row_is_written_as_a_row(self, device, monkeypatch):
    s = Tensor('s', ())
    v = Tensor('v', (1, self.M))
    C = Tensor('C', (self.N, self.M))
    [descr] = lower(device, C['ij'].subslice('i', 0, 1) <= s[''] * v['ij'], monkeypatch)
    assert descr.leftTerm.name == 's' and descr.rightTerm.name == 'v'
    assert sizes(descr) == (1, self.M, 1)
    bbox, region = BatchedOperationsAux.forge_region(descr.result.memoryLayout, descr.mnk()[:2])
    # the first row of C, its columns a column of C apart
    assert bbox[0].stop == self.N and region[0] == Range(0, 1)

  def test_a_window_away_from_the_start_along_a_fixed_index_is_refused(self, device, monkeypatch):
    # folded to a matrix, the fixed index is gone, and the offset with it
    s = Tensor('s', ())
    v = Tensor('v', (1, self.M))
    C = Tensor('C', (self.N, self.M))
    with pytest.raises(NotImplementedError, match='not at the start of its storage'):
      lower(device, C['ij'].subslice('i', 2, 3) <= s[''] * v['ij'], monkeypatch)

  def test_a_product_that_shares_an_index_is_refused(self, device, monkeypatch):
    a = Tensor('a', (self.N,))
    b = Tensor('b', (self.N,))
    C = Tensor('C', (self.N,))
    with pytest.raises(NotImplementedError, match='share an index'):
      lower(device, C['k'] <= a['k'] * b['k'], monkeypatch)


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


class TestForgeRegion:
  """gemmforge reads a window into a tensor through the tensor's pointer."""

  def test_a_window_is_shifted_to_where_it_lies(self):
    view = DenseMemoryLayout((9, 9)).subslice(0, 3, 6)
    bbox, region = BatchedOperationsAux.forge_region(view, (Range(0, 3), Range(0, 9)))
    assert [(r.start, r.stop) for r in bbox] == [(0, 9), (0, 9)]
    assert region == (Range(3, 6), Range(0, 9))

  def test_so_is_a_transposed_one(self):
    view = DenseMemoryLayout((9, 9)).subslice(0, 3, 6)
    _, region = BatchedOperationsAux.forge_region(view, (Range(0, 9), Range(0, 3)), transpose=True)
    assert region == (Range(0, 9), Range(3, 6))

  def test_a_window_into_a_window_is_shifted_twice(self):
    view = DenseMemoryLayout((9, 9)).subslice(1, 2, 8).subslice(1, 1, 3)
    _, region = BatchedOperationsAux.forge_region(view, (Range(0, 9), Range(0, 2)))
    assert region == (Range(0, 9), Range(3, 5))

  def test_a_tensor_of_its_own_is_left_alone(self):
    layout = DenseMemoryLayout((9, 9))
    bbox, region = BatchedOperationsAux.forge_region(layout, (Range(1, 4), Range(2, 5)))
    assert bbox is layout.bbox() and region == (Range(1, 4), Range(2, 5))
