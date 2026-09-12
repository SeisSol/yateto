import collections
import hashlib
import re


class DataEntry(object):
  """One constant array as it is laid out in memory.

  An entry is a rendered element list plus the little the surrounding C++
  needs to know about it: the element type, how many elements there are and
  what alignment the consumer asked for. What produced the elements -- the
  layout arithmetic in MemoryLayout.pack, or an external generator that
  packed them itself -- deliberately does not show up here. The pool stores
  images and hands out addresses; it does not interpret them.

  Two entries are the same entry when they render the same elements of the
  same type. Hashing the rendering rather than the numbers is conservative
  in the safe direction: it can fail to merge two arrays that hold equal
  values written differently, but it can never merge two arrays that hold
  different values.
  """

  def __init__(self, hint, values, typename, alignment=1):
    self._hint = hint
    self._values = list(values)
    self._typename = typename
    self._alignment = alignment
    self._name = None

    text = ', '.join(str(value) for value in self._values)
    digest = hashlib.sha256('{}|{}'.format(typename, text).encode('utf-8'))
    self._key = digest.hexdigest()

  def key(self):
    return self._key

  def hint(self):
    return self._hint

  def name(self):
    return self._name

  def setName(self, name):
    self._name = name

  def values(self):
    return self._values

  def typename(self):
    return self._typename

  def elements(self):
    return len(self._values)

  def alignment(self):
    return self._alignment

  def raiseAlignment(self, alignment):
    """Widens the alignment of an entry two consumers share.

    Honouring the stricter of the two requests satisfies both; the looser
    consumer does not care that it got more than it asked for.
    """
    self._alignment = max(self._alignment, alignment)


class DataCache(object):
  """Registry for the constant arrays a generator run needs in memory.

  The counterpart of RoutineCache, and used the same way: whoever needs an
  array announces it and gets a symbol name back, identical announcements
  collapse into one entry, and at the end the cache writes out everything it
  collected. Where RoutineCache holds code, this holds data.

  Registration order is preserved, so the layout of the emitted pool is a
  function of the input and not of dictionary iteration.
  """

  NAME_SUFFIX_LENGTH = 8

  def __init__(self):
    self._entries = collections.OrderedDict()
    self._names = dict()

  def add(self, hint, values, typename, alignment=1):
    """Registers one array and returns the symbol name it will be stored under.

    ``hint`` only shapes the name, so that the emitted pool stays readable;
    it has no part in deciding whether two arrays are the same.
    """
    entry = DataEntry(hint, values, typename, alignment)
    known = self._entries.get(entry.key())
    if known is not None:
      known.raiseAlignment(alignment)
      return known.name()

    entry.setName(self._makeName(hint, entry.key()))
    self._entries[entry.key()] = entry
    return entry.name()

  def entries(self):
    return list(self._entries.values())

  def __len__(self):
    return len(self._entries)

  def _makeName(self, hint, key):
    stem = re.sub(r'\W', '_', hint)
    length = self.NAME_SUFFIX_LENGTH
    while True:
      name = '{}_{}'.format(stem, key[:length])
      if self._names.get(name, key) == key:
        self._names[name] = key
        return name
      # A stem plus eight hex digits colliding across two different entries is
      # not something to plan for, but silently aliasing two arrays onto one
      # symbol would be a wrong-values bug, so widen instead of hoping.
      length += self.NAME_SUFFIX_LENGTH
      if length > len(key):
        raise RuntimeError('Could not find a unique pool symbol for {}.'.format(hint))
