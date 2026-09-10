"""The condition a statement runs under."""

from .ast.indices import Indices


class Guard:
  """A conjunction of literals over condition values.

  A literal is a ``(variable, version)`` pair mapped to the polarity the value
  must have. Versions matter because a condition tensor may be written inside
  the kernel: two reads separated by a write denote different values and must
  not be treated as the same literal.

  Guards form a meet-semilattice under conjunction. That is all the guard
  language needs -- ``assignIf`` produces a single positive literal, and nesting
  conjoins. Disjunction would require a richer representation (a truth-table
  bitmask over the condition variables is the cheapest one that stays exact),
  but nothing produces a disjunctive guard.
  """

  __slots__ = ('_literals', '_never')

  def __init__(self, literals=None, never=False):
    self._literals = dict(literals) if literals else dict()
    self._never = never

  @classmethod
  def always(cls):
    return cls()

  @classmethod
  def never(cls):
    return cls(never=True)

  @classmethod
  def literal(cls, variable, version=0, polarity=True):
    return cls({(variable, version): polarity})

  @classmethod
  def coerce(cls, value):
    if isinstance(value, Guard):
      return value
    if isinstance(value, bool):
      return cls.always() if value else cls.never()
    return cls.literal(value)

  def isAlways(self):
    return not self._never and len(self._literals) == 0

  def isNever(self):
    return self._never

  def __and__(self, other):
    other = Guard.coerce(other)
    if self._never or other._never:
      return Guard.never()
    literals = dict(self._literals)
    for key, polarity in other._literals.items():
      if literals.get(key, polarity) != polarity:
        return Guard.never()
      literals[key] = polarity
    return Guard(literals)

  __rand__ = __and__

  def implies(self, other):
    """Whether this guard holds only where `other` does.

    Exact for conjunctions: a superset of literals is the stronger formula.
    """
    other = Guard.coerce(other)
    if self._never or other.isAlways():
      return True
    if other._never:
      return False
    return all(self._literals.get(key, None) is polarity
               for key, polarity in other._literals.items())

  def literals(self):
    """The literals as ``(variable, version, polarity)`` triples, in a stable order."""
    return [(var, version, polarity)
            for (var, version), polarity in sorted(self._literals.items(),
                                                   key=lambda kv: (str(kv[0][0]), kv[0][1]))]

  def variables(self):
    """The values the guard is decided on.

    A literal is keyed by whatever the statement was guarded on, which is
    something the generated code reads: it has storage, a datatype and a
    liveness, and every one of those is asked of it somewhere.
    """
    return {var for var, _ in self._literals}

  def _key(self):
    return (self._never, frozenset(self._literals.items()))

  def __eq__(self, other):
    return isinstance(other, Guard) and self._key() == other._key()

  def __hash__(self):
    return hash(self._key())

  def __bool__(self):
    raise TypeError('A Guard is not a bool; use isAlways()/isNever()/implies().')

  def ccode(self):
    if self.isAlways():
      return 'true'
    if self.isNever():
      return 'false'
    # a by-value operand is named directly, a by-pointer one is dereferenced
    printvar = lambda var: f'{var}' if var.isPassedByValue() \
                           else f'{var}[{var.memoryLayout.addressString(Indices())}]'
    formatlit = lambda var, polarity: printvar(var) if polarity else f'!{printvar(var)}'
    return ' && '.join(f'({formatlit(var, polarity)})'
                       for var, _, polarity in self.literals())

  def __repr__(self):
    if self.isAlways():
      return 'Guard(always)'
    if self.isNever():
      return 'Guard(never)'
    body = ', '.join(f'{"" if polarity else "~"}{var}@{version}'
                     for var, version, polarity in self.literals())
    return f'Guard({body})'
