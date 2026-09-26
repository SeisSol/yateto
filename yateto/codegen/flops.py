import collections


class FlopCount(object):
  """How much arithmetic a kernel issues, and of what kind.

  One number stops being an answer as soon as a generator may work in a
  precision other than the one the operation is written in. An FP64 multiply
  carried out as three TF32 products is not three FP64 multiplies and it is
  not one either; the peak to divide by differs between the two by an order
  of magnitude, so a ratio taken against either is wrong. Keeping the count
  per kind makes that visible, and summing across kinds becomes something a
  caller asks for rather than something it gets by accident.

  What a kind is called is the generator's to say. yateto knows the shape --
  a name and a count -- and not the vocabulary: a matrix instruction a
  backend gains is then a fact about that backend, and not a constant that
  has to be added here before it can be counted.

  Adds like a number in both directions, so a generator with nothing to say
  about kinds goes on saying what it always said.
  """

  #: What a count with no kind attached is filed under. Plain arithmetic, in
  #: whatever the operation's own type is, which is what a count meant while
  #: there was only one kind of it.
  PLAIN = 'plain'

  def __init__(self, counts=None):
    self._counts = collections.OrderedDict()
    if isinstance(counts, FlopCount):
      self._counts.update(counts._counts)
    elif isinstance(counts, dict):
      for kind, count in counts.items():
        self._add(str(kind), int(count))
    elif counts:
      self._add(self.PLAIN, int(counts))

  def total(self):
    """Every kind summed, which only means something where there is one kind."""
    return sum(self._counts.values())

  def kinds(self):
    return collections.OrderedDict(self._counts)

  def isPlain(self):
    """Whether nothing but unqualified arithmetic was counted."""
    return set(self._counts) <= {self.PLAIN}

  def _add(self, kind, count):
    self._counts[kind] = self._counts.get(kind, 0) + count

  def __add__(self, other):
    out = FlopCount(self)
    for kind, count in FlopCount(other)._counts.items():
      out._add(kind, count)
    return out

  __radd__ = __add__

  def __int__(self):
    return self.total()

  __index__ = __int__

  def __eq__(self, other):
    if isinstance(other, FlopCount):
      return self._counts == other._counts
    return self.total() == other

  def __hash__(self):
    return hash(tuple(sorted(self._counts.items())))

  def __bool__(self):
    return self.total() != 0

  def __str__(self):
    return str(self.total())

  def __format__(self, spec):
    return format(self.total(), spec)

  def __repr__(self):
    return 'FlopCount({!r})'.format(dict(self._counts))
