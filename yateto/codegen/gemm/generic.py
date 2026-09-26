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

  @staticmethod
  def _scale(alpha):
    """The scale factor as a prefix, or nothing when it is one."""
    return '' if alpha == 1.0 else f'{alpha} * '

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
    sppA = d.leftTerm.memoryLayout.entriesRel(rows, cols)

    rows, cols = (n, k) if d.transB else (k, n)
    sppB = d.rightTerm.memoryLayout.entriesRel(rows, cols)

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
    for idxA, entry in sppA:
      eA = entry[::-1] if d.transA else entry
      for idxB, entry in sppB:
        eB = entry[::-1] if d.transB else entry

        if eA[1] == eB[0]:
          # (i.e.: if k(A) == k(B))

          nzcount += 1

          cpp( '{result} += {alpha}{a} * {b};'.format(
                result = Caccess(eA[0], eB[1]),
                alpha = self._scale(d.alpha),
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
      spp = d.leftTerm.memoryLayout.entriesRel(rows, cols)
      sparse = Aaccess
      result = lambda e: Caccess(e[0], self.OUTER_INDEX)
      dense = lambda e: Baccess(e[1], self.OUTER_INDEX)
      sizes = {0: m.size(), 1: k.size(), self.OUTER_INDEX: n.size(), self.INNER_INDEX: m.size()}
      trans = d.transA
    elif d.isBCsc:
      rows, cols = (n, k) if d.transB else (k, n)
      spp = d.rightTerm.memoryLayout.entriesRel(rows, cols)
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
      for idx, entry in spp:
        e = entry[::-1] if trans else entry
        if e[0] < sizes[0] and e[1] < sizes[1]:
          cpp( '{result} += {alpha}{dense} * {sparse};'.format(
              result = result(e),
              alpha = self._scale(d.alpha),
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
        # neither the element of B nor the scale factor depends on m, so both
        # are read and multiplied once per column of A rather than once per
        # entry of it -- and the inner loop is then an axpy over m
        # `auto`, so that the operand keeps the type the whole expression used
        # to promote it to -- an integer tensor with a floating scale factor
        # would otherwise be truncated one multiplication too early
        cpp(f'auto const _b = {self._scale(d.alpha)}{Baccess("k", "n")};')
        with cpp.For('int m = 0; m < {0}; ++m'.format(m.size())):
          cpp('{C} += {A} * _b;'.format(C = Caccess('m', 'n'), A = Aaccess('m', 'k')))

    return (m.size() * n.size() * (self._flopInit(d.beta) + 2 * k.size())
            + n.size() * k.size() * (0 if d.alpha == 1.0 else 1))

  def _immediateEntries(self, term, rows, cols, trans):
    """(row, column, number) of an immediate operand within rows x cols.

    Relative to the corner, in the orientation the GEMM reads the operand in,
    and without its zeros: an entry that holds none is one no statement has
    to mention. The address is formed exactly as `_accessFun` would form it
    for a load, and the number is the one found there.
    """
    ml = term.memoryLayout
    numbers = term.immediate
    if ml.isSparse():
      stored = (cols, rows) if trans else (rows, cols)
      for idx, entry in ml.entriesRel(*stored):
        i, j = entry[::-1] if trans else entry
        if i < rows.size() and j < cols.size():
          yield i, j, numbers[idx]
      return
    corner = (cols.start, rows.start) if trans else (rows.start, cols.start)
    offset = ml.subtensorOffset(corner)
    stride = ml.stride()[::-1] if trans else ml.stride()
    for i in range(rows.size()):
      for j in range(cols.size()):
        address = offset + stride[0] * i + stride[1] * j
        if 0 <= address < len(numbers):
          yield i, j, numbers[address]

  def _nonzero(self, term, rows, cols, trans):
    datatype = term.datatype
    return [(i, j, datatype.asnumber(value))
            for i, j, value in self._immediateEntries(term, rows, cols, trans)
            if datatype.asnumber(value) != 0]

  def _sum(self, terms):
    """`c1 * x1 + c2 * x2 ...` with the numbers folded into the scale factor.

    A one is not a multiplication and a minus one is a sign, so a selector --
    one entry, holding one -- reads its operand and does nothing else.
    Returns the expression and its flops.
    """
    d = self._descr
    datatype = d.result.datatype
    text = ''
    flops = 0
    for n, (value, operand) in enumerate(terms):
      if isinstance(d.alpha, (int, float)):
        factor = d.alpha * value
        negative = factor < 0
        magnitude = abs(factor)
        scale = '' if magnitude == 1 else f'{datatype.literal(magnitude)} * '
      else:
        negative = value < 0
        magnitude = abs(value)
        scale = f'{d.alpha} * ' if magnitude == 1 else f'{d.alpha} * {datatype.literal(magnitude)} * '
      flops += scale.count('*')
      if n == 0:
        text = f'{"-" if negative else ""}{scale}{operand}'
      else:
        text += f' {"-" if negative else "+"} {scale}{operand}'
        flops += 1
    return text, flops

  def _assign(self, target, expression, flops):
    """`target` is `expression`, under whatever beta says about its old value."""
    d = self._descr
    if d.beta == 0.0:
      return f'{target} = {expression};', flops
    if d.beta == 1.0:
      return f'{target} += {expression};', flops + 1
    return f'{target} = {d.beta} * {target} + {expression};', flops + 2

  def _generateImmediate(self, cpp):
    """A GEMM with an operand whose numbers are written into the code.

    One loop per column of the result (per row, where the immediate is the
    left operand) over the other operand, and in it the sum of that column's
    entries -- the loop over k is gone, and so are the operand and its
    zeros. For a selector the sum is one term: the column of the other
    operand the one points at.
    """
    d = self._descr
    m, n, k = d.mnk()
    Caccess = self._accessFun(d.result, (m.start, n.start), False, False)
    zero = d.result.datatype.literal(0)
    flops = 0

    if d.leftTerm.immediate is not None and d.rightTerm.immediate is not None:
      # a product of two tables of numbers is a table of numbers
      A = self._nonzero(d.leftTerm, m, k, d.transA)
      B = self._nonzero(d.rightTerm, k, n, d.transB)
      entries = dict()
      for i, ka, a in A:
        for kb, j, b in B:
          if ka == kb:
            entries[(i, j)] = entries.get((i, j), 0) + a * b
      for j in range(n.size()):
        for i in range(m.size()):
          value = entries.get((i, j), 0)
          if value == 0 and d.beta == 1.0:
            continue
          if isinstance(d.alpha, (int, float)):
            expression, cost = d.result.datatype.literal(d.alpha * value), 0
          else:
            expression, cost = f'{d.alpha} * {d.result.datatype.literal(value)}', 1
          statement, cost = self._assign(Caccess(i, j), expression, cost)
          cpp(statement)
          flops += cost
      return flops

    if d.rightTerm.immediate is not None:
      assert not d.isACsc, 'an immediate operand meets a dense one'
      Aaccess = self._accessFun(d.leftTerm, (m.start, k.start), False, d.transA)
      B = self._nonzero(d.rightTerm, k, n, d.transB)
      for j in range(n.size()):
        terms = [(value, Aaccess('m', kk)) for kk, jj, value in B if jj == j]
        if not terms and d.beta == 1.0:
          continue
        expression, cost = self._sum(terms) if terms else (zero, 0)
        with cpp.For(f'int m = 0; m < {m.size()}; ++m'):
          statement, cost = self._assign(Caccess('m', j), expression, cost)
          cpp(statement)
        flops += m.size() * cost
      return flops

    assert not d.isBCsc, 'an immediate operand meets a dense one'
    Baccess = self._accessFun(d.rightTerm, (k.start, n.start), False, d.transB)
    A = self._nonzero(d.leftTerm, m, k, d.transA)
    for i in range(m.size()):
      terms = [(value, Baccess(kk, 'n')) for ii, kk, value in A if ii == i]
      if not terms and d.beta == 1.0:
        continue
      expression, cost = self._sum(terms) if terms else (zero, 0)
      with cpp.For(f'int n = 0; n < {n.size()}; ++n'):
        statement, cost = self._assign(Caccess(i, 'n'), expression, cost)
        cpp(statement)
      flops += n.size() * cost
    return flops

  def generate(self, cpp, routineCache):
    d = self._descr

    if d.leftTerm.immediate is not None or d.rightTerm.immediate is not None:
      return self._generateImmediate(cpp)

    if d.isACsc and d.isBCsc:
      return self._generateSparseSparse(cpp)

    if d.isACsc or d.isBCsc:
      return self._generateSparseDense(cpp)

    return self._generateDenseDense(cpp)
