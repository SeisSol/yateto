class Affine:
  """An affine expression over loop indices: ``c + sum(a_i * i)``.

  Addresses are formed from these rather than from text, so that pinning an
  index to a value, folding the result and asking whether an address is
  constant are all ordinary operations on a number-like object. A coefficient
  of zero is dropped, which is what makes ``isConstant()`` answer truthfully
  after a substitution.
  """

  __slots__ = ('_constant', '_terms')

  def __init__(self, constant=0, terms=None):
    self._constant = int(constant)
    # index -> coefficient, in insertion order: the emitted address lists its
    # terms in that order, and for an address built axis by axis that is the
    # axis order
    self._terms = dict()
    if terms:
      for index, coefficient in terms.items():
        if coefficient != 0:
          self._terms[index] = int(coefficient)

  @classmethod
  def of(cls, value):
    if isinstance(value, Affine):
      return value
    if isinstance(value, Index):
      return cls(0, {value: 1})
    return cls(value)

  def isConstant(self):
    return len(self._terms) == 0

  def constant(self):
    """The constant part, which is the whole expression once it is constant."""
    return self._constant

  def indices(self):
    return list(self._terms)

  def coefficient(self, index):
    return self._terms.get(index, 0)

  def __add__(self, other):
    other = Affine.of(other)
    terms = dict(self._terms)
    for index, coefficient in other._terms.items():
      terms[index] = terms.get(index, 0) + coefficient
    return Affine(self._constant + other._constant, terms)

  __radd__ = __add__

  def __neg__(self):
    return Affine(-self._constant, {i: -c for i, c in self._terms.items()})

  def __sub__(self, other):
    return self + (-Affine.of(other))

  def __rsub__(self, other):
    return Affine.of(other) + (-self)

  def __mul__(self, factor):
    factor = int(factor)
    return Affine(self._constant * factor,
                  {i: c * factor for i, c in self._terms.items()})

  __rmul__ = __mul__

  def substituted(self, values):
    """Replace indices; `values` maps an index to a number or another index."""
    result = Affine(self._constant)
    for index, coefficient in self._terms.items():
      if index in values:
        result = result + coefficient * Affine.of(values[index])
      else:
        result = result + Affine(0, {index: coefficient})
    return result

  def ccode(self, prefix='_'):
    if self.isConstant():
      return str(self._constant)
    terms = [f'{coefficient}*{prefix}{index.name}'
             for index, coefficient in self._terms.items()]
    code = ' + '.join(terms)
    if self._constant > 0:
      code += f' + {self._constant}'
    elif self._constant < 0:
      code += f' - {-self._constant}'
    return code

  def _key(self):
    return (self._constant, tuple(sorted(self._terms.items(), key=lambda kv: id(kv[0]))))

  def __eq__(self, other):
    return isinstance(other, Affine) and self._key() == other._key()

  def __hash__(self):
    return hash(self._key())

  def __repr__(self):
    return f'Affine({self.ccode()})'


class Index:
  """A loop index.

  The name reaches the generated code, so it is the index letter the kernel
  was written with: an address built here reads the same as the one the
  surrounding loop nest spells.
  """

  __slots__ = ('name',)

  def __init__(self, name):
    self.name = str(name)

  def __repr__(self):
    return f'Index({self.name})'
