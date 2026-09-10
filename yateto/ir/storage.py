"""Which temporaries share storage, and how much of it there is."""

from .ops import Call, Load, Memset, Pointer, Store


def assign(region):
  """Give every temporary the region touches a share of storage.

  A temporary lives from the statement that first touches it to the one that
  last does, and two whose lives do not overlap can be given the same share.
  Statements are the region's own operations in the order they stand: a nest
  is one statement however much it does inside, which is enough, since a
  temporary that outlives a statement outlives all of it.

  Hands back which share each temporary is given and how many bytes each
  share needs. A statement that does not say what it touches makes every
  lifetime unknown, and then nothing is shared with anything.
  """
  touched, sizes, stated = _lifetimes(region)

  shares = {}
  used = {}
  free = []
  count = 0
  bytes = {}

  for position, names in enumerate(touched):
    for name in sorted(names):
      if name in used:
        continue
      if free and stated:
        share = free.pop()
      else:
        share, count = count, count + 1
      used[name] = share
      shares[name] = share
      bytes[share] = max(bytes.get(share, 0), sizes[name])

    if not stated:
      continue
    # a temporary nothing touches again has nothing left to keep
    for name in [name for name in used
                 if not any(name in later for later in touched[position + 1:])]:
      free.insert(0, used.pop(name))

  return shares, bytes


def _lifetimes(region):
  """Per statement, the temporaries it touches; their sizes; and whether all
  of the statements said."""
  touched = []
  sizes = {}
  stated = True
  for op in region.ops:
    names = set()
    for inner in [op] + list(_nested(op)):
      for buffer in _buffers(inner):
        if buffer is None:
          stated = False
          continue
        if buffer.temporary:
          names.add(buffer.name)
          sizes[buffer.name] = max(sizes.get(buffer.name, 0), _size(buffer))
    touched.append(names)
  return touched, sizes, stated


def _nested(op):
  for region in op.regions():
    yield from region.walk()


def _buffers(op):
  if isinstance(op, (Load, Store, Memset, Pointer)):
    return [op.buffer]
  if isinstance(op, Call):
    return op.touches() if op.states() else [None]
  return []


def _size(buffer):
  return buffer.memoryLayout.storage().requiredReals() * buffer.datatype.size()
