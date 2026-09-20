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


class Reservation(object):
  """A pool entry announced before its content is known.

  The symbol is fixed when the reservation is made, so code emitted at that
  moment can already name what it will read. What ends up behind the symbol
  is decided later, when the values arrive, and two reservations filled with
  the same image come to rest on one array.

  That the symbol does not follow from the content is the whole point: a
  generator that decides how it wants its operand laid out while a kernel is
  being written out, and only produces the image once it is asked to emit,
  needs a name in between.
  """

  def __init__(self, name, hint):
    self._name = name
    self._hint = hint
    self._entry = None

  def name(self):
    """The symbol, valid from the moment the reservation is made."""
    return self._name

  def hint(self):
    return self._hint

  def isFilled(self):
    return self._entry is not None

  def entry(self):
    return self._entry

  def setEntry(self, entry):
    self._entry = entry


class DataCache(object):
  """Registry for the constant arrays a generator run needs in memory.

  The counterpart of RoutineCache, and used the same way: whoever needs an
  array announces it and gets a symbol name back, identical announcements
  collapse into one entry, and at the end the cache writes out everything it
  collected. Where RoutineCache holds code, this holds data.

  There are two ways in. ``add`` states hint and content together and names
  the result after the content, so the same array announced twice is one
  entry under one name. ``reserve`` states only the hint, hands back a symbol
  straight away and takes the content through ``fill`` whenever it turns up.
  Both end in the same place: one array per distinct image, and as many
  symbols pointing at it as there were announcements.

  Registration order is preserved, so the layout of the emitted pool is a
  function of the input and not of dictionary iteration.
  """

  NAME_SUFFIX_LENGTH = 8

  def __init__(self):
    self._entries = collections.OrderedDict()
    self._reservations = collections.OrderedDict()
    self._names = dict()

  def add(self, hint, values, typename, alignment=1):
    """Registers one array and returns the symbol name it will be stored under.

    ``hint`` only shapes the name, so that the emitted pool stays readable;
    it has no part in deciding whether two arrays are the same.
    """
    return self._intern(hint, values, typename, alignment).name()

  def reserve(self, hint):
    """Announces an array whose content is not known yet.

    The reservation carries a symbol of its own, distinct from the name of
    whatever array it ends up on, because it has to be handed out before
    there is anything to name the array after.
    """
    reservation = Reservation(self._takeName(hint, None), hint)
    self._reservations[reservation.name()] = reservation
    return reservation

  def fill(self, reservation, values, typename, alignment=1):
    """Supplies the content of a reservation, and reports the array it lands on."""
    if reservation.isFilled():
      raise RuntimeError('Pool entry {} was filled twice.'.format(reservation.name()))
    entry = self._intern(reservation.hint(), values, typename, alignment)
    reservation.setEntry(entry)
    return entry.name()

  def entries(self):
    return list(self._entries.values())

  def reservations(self):
    """Every reservation, in the order it was made.

    A reservation that was never filled is refused here rather than emitted:
    its symbol has already been written into generated code, and nothing
    behind it would be a dangling read, not a missing array.
    """
    unfilled = [r.name() for r in self._reservations.values() if not r.isFilled()]
    if unfilled:
      raise RuntimeError('Reserved in the constant pool but never filled: {}.'.format(
        ', '.join(unfilled)))
    return list(self._reservations.values())

  def __len__(self):
    return len(self._entries)

  def _intern(self, hint, values, typename, alignment):
    entry = DataEntry(hint, values, typename, alignment)
    known = self._entries.get(entry.key())
    if known is not None:
      known.raiseAlignment(alignment)
      return known

    entry.setName(self._takeName(hint, entry.key()))
    self._entries[entry.key()] = entry
    return entry

  def _takeName(self, hint, key):
    """A symbol nothing else in this cache answers to.

    With a ``key``, the name states it, so that the same content announced
    twice asks for the same symbol and gets it. Without one, the stem is
    numbered until it is free, because there is nothing yet to derive a name
    from and two reservations of the same hint are two different arrays until
    proven otherwise.
    """
    stem = re.sub(r'\W', '_', hint)
    if key is None:
      name = stem
      ordinal = 0
      while name in self._names:
        ordinal += 1
        name = '{}_{}'.format(stem, ordinal)
      self._names[name] = None
      return name

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
