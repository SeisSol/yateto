import unittest
from yateto.type import Tensor
from yateto.ast.node import IndexedTensor, Contraction
from yateto.ast.indices import Indices
from yateto.ast.log import LoG
from yateto.memory import DenseMemoryLayout


class LogNode(unittest.TestCase):
  def test_log_is_gemm(self):
    left = IndexedTensor(tensor=Tensor(name='A',
                                       shape=(30, 10)),
                         indexNames='ij')

    right = IndexedTensor(tensor=Tensor(name='B',
                                        shape=(10, 40)),
                          indexNames='jk')

    contraction_shape = (30, 40)
    contraction = Contraction(indices=Indices(indexNames='ik', shape=contraction_shape),
                              lTerm=left,
                              rTerm=right,
                              sumIndices={'j'})
    contraction.setMemoryLayout(DenseMemoryLayout(shape=contraction_shape))

    log = LoG(contraction)
    self.assertTrue(log.is_pure_gemm())

  def test_log_is_not_gemm(self):
    left = IndexedTensor(tensor=Tensor(name='A',
                                       shape=(30, 10, 20)),
                         indexNames='ijk')

    right = IndexedTensor(tensor=Tensor(name='B',
                                        shape=(40, 10, 30)),
                         indexNames='lji')

    contraction_shape = (20, 40)
    contraction = Contraction(indices=Indices(indexNames='kl', shape=contraction_shape),
                              lTerm=left,
                              rTerm=right,
                              sumIndices={'i', 'j'})
    contraction.setMemoryLayout(DenseMemoryLayout(shape=contraction_shape))

    log = LoG(contraction)
    self.assertFalse(log.is_pure_gemm())
