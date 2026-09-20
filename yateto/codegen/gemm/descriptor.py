import collections
import hashlib

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
