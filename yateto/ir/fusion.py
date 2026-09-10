"""Putting adjacent nests over one iteration space into one nest."""

from .address import entry
from .core import Region
from .ops import Call, Const, Load, Loop, Memset, Pointer, Read, Store
from .passes import _clone


def fuseLoops(region):
  """Merge every run of adjacent nests that walk the same iteration space.

  Each of them walks that space on its own today, so what one leaves in a
  buffer the next reads back from memory a whole pass later. Merging them
  keeps the order every element is treated in -- each nest touches one element
  per iteration and no other -- and reads what was just written while it is
  still warm.

  A nest that hands work to someone else is left alone: what a call reads and
  writes is not stated here, so nothing can be said about moving it.
  """
  for op in region.ops:
    for nested in op.regions():
      fuseLoops(nested)

  ops = []
  position = 0
  while position < len(region.ops):
    group, hoisted, after, position = _group(region, position)
    if len(group) > 1:
      ops.extend(hoisted)
      ops.append(_merge(group))
      ops.extend(after)
    else:
      ops.extend(hoisted)
      ops.extend(group)
      ops.extend(after)
  region.ops = ops
  return region


def _group(region, position):
  """The run of nests that starts at `position`, and what moves with it.

  A zeroing between two nests of the group moves in front of the group, but
  only where nothing the group has done so far touches what it zeroes. One
  behind the last nest of the group stays behind it.

  Hands back where to carry on, which is behind everything accounted for
  here. What was looked at and not accounted for is looked at again.
  """
  first = region.ops[position]
  group = [first]
  hoisted = []
  pending = []
  index = position + 1
  end = position + 1

  if _nest(first) is None:
    return [first], [], [], position + 1

  while index < len(region.ops):
    candidate = region.ops[index]
    if isinstance(candidate, (Const, Read)):
      # nothing it reads is anything a nest writes, so where it stands is
      # only a matter of standing before whoever uses it
      pending.append(candidate)
      index += 1
      continue
    if isinstance(candidate, Memset):
      if any(candidate.buffer.name in _touched(member) for member in group):
        break
      pending.append(candidate)
      index += 1
      continue
    if _nest(candidate) is None or not _sameSpace(first, candidate) \
       or not _independent(group, candidate):
      break
    hoisted.extend(pending)
    pending = []
    group.append(candidate)
    index += 1
    end = index

  if len(group) == 1:
    return group, hoisted, [], end
  return group, hoisted, pending, index


def _nest(op):
  """The chain of loops and the body it ends on, or None where there is none.

  A nest that states its entries, or that holds a call or a pointer into a
  buffer, is not one this can reason about. Neither is one that reaches into
  a buffer at a place that is a lookup rather than an expression: where it
  lands cannot be compared with where the nest beside it lands.
  """
  if not isinstance(op, Loop) or op.isUnrollable():
    return None
  for inner in op.region.walk():
    if isinstance(inner, (Call, Pointer, Memset)):
      return None
    if isinstance(inner, (Load, Store)) and entry(inner.buffer, inner.coords) is None:
      return None
  indices, ranges = [], []
  loop = op
  while True:
    if len(loop.index) != 1:
      return None
    indices.append(loop.index[0].name)
    ranges.append((loop.domain.start, loop.domain.stop))
    body = loop.region.ops
    if len(body) == 1 and isinstance(body[0], Loop) and not body[0].isUnrollable():
      loop = body[0]
    else:
      return indices, ranges, loop


def _sameSpace(first, second):
  a, b = _nest(first), _nest(second)
  # the names may differ; the ranges, their order and the clauses may not
  return (a[1] == b[1] and first.simd == second.simd
          and first.collapse == second.collapse)


def _accesses(op):
  """Per buffer, which entry this nest touches, said in the nest's own terms."""
  canonical = _canonical(op)
  reads, writes = {}, {}
  for inner in op.region.walk():
    if isinstance(inner, Load):
      reads.setdefault(inner.buffer.name, set()).add(_signature(inner, canonical))
    elif isinstance(inner, Store):
      writes.setdefault(inner.buffer.name, set()).add(_signature(inner, canonical))
  return reads, writes


def _canonical(op):
  """Each loop of the nest by its depth rather than by its name.

  Two nests over the same space need not spell their indices alike: what a
  copy calls `a` an element-wise operation calls by the index the kernel was
  written with. Which loop of the nest an index belongs to is what they have
  in common, and it is what decides whether they touch the same entry.
  """
  canonical = {}
  loop, depth = op, 0
  while True:
    canonical[loop.index[0]] = str(depth)
    body = loop.region.ops
    if len(body) == 1 and isinstance(body[0], Loop):
      loop, depth = body[0], depth + 1
    else:
      return canonical


def _touched(op):
  reads, writes = _accesses(op)
  return set(reads) | set(writes)


def _signature(access, canonical):
  """Which entry is touched, said as an offset from the loops of the nest.

  The offset into the buffer and not the coordinate that names it: two
  operands may be views into one buffer that start at different places, and
  then one coordinate is two entries.
  """
  offset = entry(access.buffer, access.coords)
  return (offset.constant(),
          tuple(sorted((canonical.get(index, index.name),
                        offset.coefficient(index))
                       for index in offset.indices())))


def _independent(group, candidate):
  """Whether the candidate may join without changing what anything reads.

  Every buffer the candidate shares with the group has to be touched at one
  and the same entry throughout. Then what one nest writes, the next reads in
  the same iteration, and running them together per element is the order they
  already had. A buffer touched at two different entries could have a nest
  reading what a later iteration of an earlier nest wrote.
  """
  spellings = {}
  for member in group + [candidate]:
    reads, writes = _accesses(member)
    for accesses in (reads, writes):
      for name, entries in accesses.items():
        spellings.setdefault(name, set()).update(entries)

  shared = _touched(candidate) & set().union(*(_touched(m) for m in group))
  return all(len(spellings[name]) == 1 for name in shared)


def _merge(group):
  """One nest running every body of the group, in the order they had."""
  first = group[0]
  indices, _, innermost = _nest(first)
  body = Region(list(innermost.region.ops))
  for member in group[1:]:
    memberIndices, _, memberBody = _nest(member)
    rename = _rename(member, first)
    body.extend(_clone(memberBody.region, rename, {}))
  innermost.region = body
  return first


def _rename(member, target):
  """Maps the member's loop indices onto the ones the group already runs."""
  rename = {}
  source, destination = member, target
  while True:
    rename[source.index[0]] = destination.index[0]
    body = source.region.ops
    if len(body) == 1 and isinstance(body[0], Loop):
      source = body[0]
      destination = destination.region.ops[0]
    else:
      return rename
