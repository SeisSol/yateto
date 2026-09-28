import collections
import hashlib
import operator
from functools import reduce

_FIELDS = [
  'operation', 'arch',
  'm', 'n', 'k',
  'ldA', 'ldB', 'ldC',
  'alpha', 'beta',
  'alignedA', 'alignedC', 'prefetch',
  'transA', 'transB',
  'datatypeA', 'datatypeB', 'datatypeC',
  'sppA', 'sppB',
]


class GemmDescriptor(collections.namedtuple('GemmDescriptor', _FIELDS)):
  """Everything the identity of a GEMM routine rests on, as plain data.

  The name a routine is called by follows from this and from nothing else:
  no values, no layout objects, no architecture object. A routine can
  therefore be named where its kernel is written and emitted somewhere else
  entirely -- in another run, in another process, from a file -- and the two
  still agree on what to call it.

  Everything is spelled canonically on the way in. Index positions arrive
  from the layouts as whatever integer type the array library hands out, and
  letting that reach the name would tie it to which versions are installed
  rather than to the kernel being generated.
  """

  __slots__ = ()

  #: Sparsity encoded in the name as the pair (A dense, B dense).
  _SPARSITY = {
    (True, True): 'dense',
    (True, False): 'bsparse',
    (False, True): 'asparse',
    (False, False): 'absparse',
  }

  _NAME = ('{operation}_{sparsity}{patterns}'
           '_{datatypeA}_{datatypeB}_{datatypeC}_{arch}'
           '_m{m}_n{n}_k{k}_ldA{ldA}_ldB{ldB}_ldC{ldC}'
           '_alpha{alpha}_beta{beta}_alignedA{alignedA}_alignedC{alignedC}'
           '_transA{transA}_transB{transB}_{prefetch}')

  @classmethod
  def create(cls, operation, arch, gemm, sppA=None, sppB=None):
    """Reads one off a GEMM's parameters, the sparsity patterns given apart.

    ``gemm`` carries alpha and beta already reduced to the cases the
    generators treat specially, since which cases those are is the
    generator's business and not the name's.
    """
    return cls(
      operation=str(operation),
      arch=str(arch).replace('-', '_'),
      m=int(gemm['M']), n=int(gemm['N']), k=int(gemm['K']),
      ldA=int(gemm['LDA']), ldB=int(gemm['LDB']), ldC=int(gemm['LDC']),
      alpha=str(gemm['alpha']), beta=str(gemm['beta']),
      alignedA=int(gemm['alignedA']), alignedC=int(gemm['alignedC']),
      prefetch=str(gemm['prefetch']),
      transA=bool(gemm['transA']), transB=bool(gemm['transB']),
      datatypeA=str(gemm['datatypeA']),
      datatypeB=str(gemm['datatypeB']),
      datatypeC=str(gemm['datatypeC']),
      sppA=cls._pattern(sppA), sppB=cls._pattern(sppB),
    )

  def routineName(self):
    patterns = ''.join('_' + self._digest(spp) for spp in (self.sppA, self.sppB)
                       if spp is not None)
    return self._NAME.format(
      sparsity=self._SPARSITY[(self.sppA is None, self.sppB is None)],
      patterns=patterns,
      **self._asdict())

  def asJson(self):
    """A dict of nothing but numbers, strings, booleans, lists and None."""
    data = dict(self._asdict())
    for side in ('sppA', 'sppB'):
      if data[side] is not None:
        data[side] = [list(entry) for entry in data[side]]
    return data

  @classmethod
  def fromJson(cls, data):
    fields = dict(data)
    for side in ('sppA', 'sppB'):
      if fields.get(side) is not None:
        fields[side] = tuple(tuple(int(i) for i in entry) for entry in fields[side])
    return cls(**fields)

  @staticmethod
  def _pattern(spp):
    if spp is None:
      return None
    return tuple(tuple(int(i) for i in entry) for entry in spp)

  @staticmethod
  def _digest(spp):
    # cf. https://stackoverflow.com/a/65766676
    sha = hashlib.new('md5', usedforsecurity=False)
    sha.update(str([tuple(entry) for entry in spp]).encode())
    return sha.hexdigest()


_QUERY_FIELDS = [
  'arch', 'alignedReals',
  'm', 'n', 'k',
  'alpha', 'beta',
  'transA', 'transB',
  'prefetch',
  'datatypeA', 'datatypeB', 'datatypeC',
  'patternA', 'patternB',
]


class GemmQuery(collections.namedtuple('GemmQuery', _QUERY_FIELDS)):
  """What a GEMM is, asked before it is decided how its operands are arranged.

  The operation without the arrangement: the extents the operands really
  occupy, what is really nonzero in them, the types, the scalars, the
  transpositions, and the machine. No leading dimensions, no alignment flags,
  no sparsity encoding -- those are the answer, and putting them in the
  question would mean asking with the answer already in hand.

  Carried as data, for the same reason a descriptor is: a generator that
  answers need not be in this process.

  A pattern is ``None`` where everything within the extent is nonzero. That
  is about the operation, not about how it ends up stored: an operand whose
  values happen to be dense is dense here even if its layout compresses them,
  and one with holes in it says so even if its layout stores them.
  """

  __slots__ = ()

  @classmethod
  def create(cls, arch, descr):
    m, n, k = descr.logicalMnk()
    kA = 1 if not descr.transA else 0
    kB = 0 if not descr.transB else 1
    rangesA = (k, m) if descr.transA else (m, k)
    rangesB = (n, k) if descr.transB else (k, n)
    return cls(
      arch=str(arch.name).replace('-', '_'),
      alignedReals=int(arch.alignedReals),
      m=[int(m.start), int(m.stop)],
      n=[int(n.start), int(n.stop)],
      k=[int(k.start), int(k.stop)],
      alpha=str(descr.alpha), beta=str(descr.beta),
      transA=bool(descr.transA), transB=bool(descr.transB),
      prefetch=descr.prefetchName is not None,
      datatypeA=str(descr.leftTerm.datatype),
      datatypeB=str(descr.rightTerm.datatype),
      datatypeC=str(descr.result.datatype),
      patternA=cls._holes(descr.leftTerm.eqspp, rangesA),
      patternB=cls._holes(descr.rightTerm.eqspp, rangesB),
    )

  def asJson(self):
    """A dict of nothing but numbers, strings, booleans, lists and None."""
    data = dict(self._asdict())
    for side in ('patternA', 'patternB'):
      if data[side] is not None:
        data[side] = [list(entry) for entry in data[side]]
    data['m'] = list(data['m'])
    data['n'] = list(data['n'])
    data['k'] = list(data['k'])
    return data

  @classmethod
  def fromJson(cls, data):
    fields = dict(data)
    for side in ('patternA', 'patternB'):
      if fields.get(side) is not None:
        fields[side] = tuple(tuple(int(i) for i in entry) for entry in fields[side])
    for extent in ('m', 'n', 'k'):
      fields[extent] = [int(i) for i in fields[extent]]
    return cls(**fields)

  @staticmethod
  def _holes(eqspp, ranges):
    inside = [tuple(int(i - r.start) for i, r in zip(entry, ranges))
              for entry in zip(*eqspp.nonzero())
              if all(r.start <= i < r.stop for i, r in zip(entry, ranges))]
    if len(inside) == reduce(operator.mul, (r.size() for r in ranges), 1):
      return None
    return tuple(sorted(inside))
