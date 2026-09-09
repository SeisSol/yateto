# Element-wise steps that share an index space are put into one loop nest, so
# what would have been a buffer becomes a variable of the loop body. The
# kernels here are shaped to exercise that, and to exercise the cases where it
# must not happen: a step that walks a different space, an operand addressed
# over fewer axes, a sparse operand written out entry by entry, and two
# statements that do not run together.
import numpy as np
from yateto import Tensor, Scalar, Datatype
from yateto.memory import CSCMemoryLayout
import yateto.functions as yf

N = 4


def add(g):
  shape = (N,) * 6
  vectors = [Tensor(chr(ord('a') + k), (N,)) for k in range(6)]
  T = Tensor('T', shape)
  U = Tensor('U', shape)
  O = Tensor('O', shape)
  mean = Scalar('mean')

  # nothing is contracted, so the search builds a tree of outer products whose
  # intermediates all walk a different space: no two of them share a nest
  g.add('outer', T['abcdef'] <= vectors[0]['a'] * vectors[1]['b'] * vectors[2]['c']
                              * vectors[3]['d'] * vectors[4]['e'] * vectors[5]['f'])

  # a long chain over one space, with a scaling in the middle and an
  # intermediate the last step reads: all of it lands in one nest
  scaled = yf.add(T['abcdef'], mean * U['abcdef'])
  g.add('chain', O['abcdef'] <= yf.where(
      yf.greater(yf.sqrt(yf.mul(scaled, scaled)), U['abcdef']),
      yf.mul(T['abcdef'], U['abcdef']),
      T['abcdef']))

  # an operand over fewer axes than the nest walks
  v = Tensor('v', (N,))
  g.add('broadcast', O['abcdef'] <= yf.add(yf.mul(T['abcdef'], v['a']), U['abcdef']))

  # sparse operands, written out entry by entry rather than looped over
  pattern = np.zeros((N, N), dtype=bool)
  pattern[0, 0] = pattern[1, 1] = pattern[3, 2] = True
  P = Tensor('P', (N, N), spp=pattern)
  P.setMemoryLayout(CSCMemoryLayout)
  Q = Tensor('Q', (N, N), spp=pattern)
  Q.setMemoryLayout(CSCMemoryLayout)
  R = Tensor('R', (N, N))
  g.add('sparse', R['ij'] <= yf.add(yf.maximum(P['ij'], Q['ij']), P['ij']))

  # two statements under different guards
  flag = Tensor('flag', (), datatype=Datatype.BOOL)
  S = Tensor('S', (N, N))
  g.add('guarded', [yf.assignIf(flag[''], R['ij'], yf.sqrt(S['ij'])),
                    R['ij'] <= yf.add(S['ij'], S['ij'])])
