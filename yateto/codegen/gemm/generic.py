from ..common import forLoops

class Generic(object):
  OUTER_INDEX = 'o'
  INNER_INDEX = 'i'

  def __init__(self, arch, descr):
    self._descr = descr

  def _flopInit(self, beta):
    return 0 if beta in [0.0, 1.0] else 1

  def _flop(self, alpha):
    if alpha == 0.0:
      return 0
    elif alpha != 1.0:
      return 3
    return 2

  def _denseAccess(self, name, offset, stride, i, j):
    return '{name}[{offset} + {stride[0]}*{i} + {stride[1]}*{j}]'.format(
			name = name,
			offset = offset,
			stride = stride,
      i = i,
      j = j
    )

  def _sparseAccess(self, name, offset, idx):
    return '{name}[{offset} + {idx}]'.format(
			name = name,
			offset = offset,
			idx = idx
    )

  def _accessFun(self, term, offset2, sparse, transpose):
    if transpose:
      offset2 = offset2[::-1]
    offset = term.memoryLayout.subtensorOffset(offset2)
    if sparse:
      def access(idx):
        return self._sparseAccess(term.name, offset, idx)
      return access

    stride = term.memoryLayout.stride()
    if transpose:
      stride = stride[::-1]
    def access(i, j):
      return self._denseAccess(term.name, offset, stride, i, j)
    return access

  def _generateSparseSparse(self, cpp):
    d = self._descr
    m, n, k = d.mnk()

    # (assume C to be dense; though we actually _could_ assume otherwise)
    # (but that's not implemented right now anyways)

    Aaccess = self._accessFun(d.leftTerm, (m.start, k.start), d.isACsc, d.transA)
    Baccess = self._accessFun(d.rightTerm, (k.start, n.start), d.isBCsc, d.transB)
    Caccess = self._accessFun(d.result, (m.start, n.start), False, False)

    rows, cols = (k, m) if d.transA else (m, k)
    sppA = d.leftTerm.memoryLayout.entries(rows, cols)

    rows, cols = (n, k) if d.transB else (k, n)
    sppB = d.rightTerm.memoryLayout.entries(rows, cols)

    if d.beta != 1.0:
      with cpp.For(f'int n = 0; n < {n.size()}; ++n'):
        with cpp.For(f'int m = 0; m < {m.size()}; ++m'):
          cpp('{} = {}{};'.format(
              Caccess('m', 'n'),
              d.beta,
              ' * ' + Caccess('m', 'n') if d.beta != 0.0 else ''
            )
          )

    # (the following loop is a bit redundant, but it works and is fast enough)
    # note that we explicitly count all nonzero operations

    nzcount = 0
    for idxA, entry in enumerate(sppA):
      eA = entry[::-1] if d.transA else entry
      for idxB, entry in enumerate(sppB):
        eB = entry[::-1] if d.transB else entry

        if eA[1] == eB[0]:
          # (i.e.: if k(A) == k(B))

          nzcount += 1

          cpp( '{result} += {alpha} * {a} * {b};'.format(
                result = Caccess(eA[0], eB[1]),
                alpha = d.alpha,
                a = Aaccess(idxA),
                b = Baccess(idxB)
              )
            )

    return nzcount * self._flop(d.alpha) + n.size() * m.size() * self._flopInit(d.beta)

  def _generateSparseDense(self, cpp):
    d = self._descr
    m, n, k = d.mnk()
    
    assert d.isACsc != d.isBCsc

    Aaccess = self._accessFun(d.leftTerm, (m.start, k.start), d.isACsc, d.transA)
    Baccess = self._accessFun(d.rightTerm, (k.start, n.start), d.isBCsc, d.transB)
    Caccess = self._accessFun(d.result, (m.start, n.start), False, False)

    if d.isACsc:
      rows, cols = (k, m) if d.transA else (m, k)
      spp = d.leftTerm.memoryLayout.entries(rows, cols)
      sparse = Aaccess
      result = lambda e: Caccess(e[0], self.OUTER_INDEX)
      dense = lambda e: Baccess(e[1], self.OUTER_INDEX)
      sizes = {0: m.size(), 1: k.size(), self.OUTER_INDEX: n.size(), self.INNER_INDEX: m.size()}
      trans = d.transA
    elif d.isBCsc:
      rows, cols = (n, k) if d.transB else (k, n)
      spp = d.rightTerm.memoryLayout.entries(rows, cols)
      sparse = Baccess
      result = lambda e: Caccess(self.OUTER_INDEX, e[1])
      dense = lambda e: Aaccess(self.OUTER_INDEX, e[0])
      sizes = {0: k.size(), 1: n.size(), self.OUTER_INDEX: m.size(), self.INNER_INDEX: n.size()}
      trans = d.transB

    with cpp.For('int {0} = 0; {0} < {1}; ++{0}'.format(self.OUTER_INDEX, sizes[self.OUTER_INDEX])):
      if d.beta != 1.0:
        with cpp.For('int {0} = 0; {0} < {1}; ++{0}'.format(self.INNER_INDEX, sizes[self.INNER_INDEX])):
          CAddr = result([self.INNER_INDEX, self.INNER_INDEX])
          cpp('{} = {}{};'.format(
              CAddr,
              d.beta,
              ' * ' + CAddr if d.beta != 0.0 else ''
            )
          )
      for idx, entry in enumerate(spp):
        e = entry[::-1] if trans else entry
        if e[0] < sizes[0] and e[1] < sizes[1]:
          cpp( '{result} += {alpha} * {dense} * {sparse};'.format(
              result = result(e),
              alpha = d.alpha,
              dense = dense(e),
              sparse = sparse(idx)
            )
          )

    return sizes[self.OUTER_INDEX] * (sizes[self.INNER_INDEX] * self._flopInit(d.beta) + self._flop(d.alpha) * len(spp))

  def _generateDenseDense(self, cpp):
    d = self._descr
    m, n, k = d.mnk()
    
    Aaccess = self._accessFun(d.leftTerm, (m.start, k.start), False, d.transA)
    Baccess = self._accessFun(d.rightTerm, (k.start, n.start), False, d.transB)
    Caccess = self._accessFun(d.result, (m.start, n.start), False, False)
    
    with cpp.For('int n = 0; n < {0}; ++n'.format(n.size())):
      if d.beta != 1.0:
        with cpp.For('int m = 0; m < {0}; ++m'.format(m.size())):
          cpp('{} = {}{};'.format(
              Caccess('m', 'n'),
              d.beta,
              ' * ' + Caccess('m', 'n') if d.beta != 0.0 else ''
            )
          )
      with cpp.For('int k = 0; k < {0}; ++k'.format(k.size())):
        with cpp.For('int m = 0; m < {0}; ++m'.format(m.size())):
          cpp( '{C} += {alpha} * {A} * {B};'.format(
              C = Caccess('m', 'n'),
              alpha = d.alpha,
              A = Aaccess('m', 'k'),
              B = Baccess('k', 'n')
            )
          )

    return  m.size() * n.size() * (self._flopInit(d.beta) + self._flop(d.alpha) * k.size())

  def generate(self, cpp, routineCache):
    d = self._descr

    if d.isACsc and d.isBCsc:
      return self._generateSparseSparse(cpp)

    if d.isACsc or d.isBCsc:
      return self._generateSparseDense(cpp)

    return self._generateDenseDense(cpp)
