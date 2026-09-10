"""Keeping in the loop body what a nest would otherwise put in memory."""

from .address import entry
from .core import Region, ValueOp
from .ops import (Arith, Call, Fold, If, Load, Loop, Memset, Pointer, Scope,
                  Store)


def scalarize(region):
  """Read back what was just written, and stop writing what nobody reads.

  A statement that leaves its result in a buffer for the next one to read is
  a pass over memory that a nest sharing both statements need not make: the
  value is still there, in the iteration that produced it. Once nothing reads
  a temporary any more, writing it is work with no reader, and the buffer
  goes with it.
  """
  _forwardAll(region)
  # Asked of the whole kernel, and only once: what a nest writes for the next
  # nest to read is dead within that nest and alive in the region.
  _dropDeadStores(region)
  _dropUnusedValues(region)
  _dropEmptyRegions(region)
  return region


def _forwardAll(region):
  for op in region.ops:
    for nested in op.regions():
      _forwardAll(nested)
    if isinstance(op, Loop) and not op.isUnrollable():
      _forward(op.region)


def _forward(region):
  """Within one iteration, a load of what was just stored is that value."""
  known = {}
  replaced = {}
  ops = []
  for op in region.ops:
    _substitute(op, replaced)

    if isinstance(op, Load):
      place = _place(op)
      if place is not None:
        if place in known:
          replaced[id(op)] = known[place]
          continue
        known[place] = op
      ops.append(op)
      continue

    if isinstance(op, Store):
      place = _place(op)
      if op.accumulate is not None and place in known:
        # what it accumulates into is a value of this iteration, so the
        # combination is one too and the store just puts the result there
        combined = Arith(op.accumulate, [known[place], op.value],
                         op.buffer.datatype)
        ops.append(combined)
        op.value = combined
        op.accumulate = None
      # what the buffer holds elsewhere is no longer known -- and where the
      # store lands is not known at all, that is everywhere in it
      for other in [other for other in known
                    if other[0] == op.buffer.name and other != place]:
        del known[other]
      if place is not None and op.accumulate is None:
        known[place] = op.value
      ops.append(op)
      continue

    if isinstance(op, (Call, Pointer, Memset)) or op.regions():
      # what any of these does to a buffer is not stated here
      known.clear()
    ops.append(op)

  region.ops = ops


def _substitute(op, replaced):
  if isinstance(op, Arith):
    op.args = [replaced.get(id(arg), arg) for arg in op.args]
  elif isinstance(op, Store):
    op.value = replaced.get(id(op.value), op.value)
  elif hasattr(op, 'value') and not isinstance(op, ValueOp):
    op.value = replaced.get(id(op.value), op.value)


def _place(op):
  """Which entry of which buffer an access touches, or None where it is unclear.

  Named by the offset the access is formed from rather than by the coordinate
  it spells, so that two views into one buffer -- one coordinate, two entries
  -- are told apart.
  """
  offset = entry(op.buffer, op.coords)
  return None if offset is None else (op.buffer.name, offset.ccode())


def _dropDeadStores(region):
  """Stop writing a temporary nothing reads back."""
  read = set()
  opaque = False
  for op in region.walk():
    if isinstance(op, Load):
      read.add(op.buffer.name)
    elif isinstance(op, Pointer):
      read.add(op.buffer.name)
    elif isinstance(op, Store) and op.accumulate is not None:
      read.add(op.buffer.name)
    elif isinstance(op, Call):
      if not op.states():
        opaque = True
      else:
        read |= op.names()
  if opaque:
    return

  def dead(op):
    if isinstance(op, (Store, Memset)):
      return op.buffer.temporary and op.buffer.name not in read
    return False

  _prune(region, dead)


def _dropUnusedValues(region):
  """A value nobody reads is arithmetic nobody asked for."""
  while True:
    used = set()
    for op in region.walk():
      for operand in op.operands():
        used.add(id(operand))
    def unused(op, used=used):
      return isinstance(op, ValueOp) and id(op) not in used
    if not _prune(region, unused):
      return


def _prune(region, condition):
  removed = False
  for op in region.ops:
    for nested in op.regions():
      removed |= _prune(nested, condition)
  kept = [op for op in region.ops if not condition(op)]
  if len(kept) != len(region.ops):
    region.ops = kept
    removed = True
  return removed


def _dropEmptyRegions(region):
  """A loop with nothing left in it is a loop with nothing to do."""
  while _prune(region, lambda op: isinstance(op, (Loop, Scope, If))
               and len(op.region) == 0):
    pass


def buffers(region):
  """Every buffer the region still mentions, by name."""
  found = {}
  for op in region.walk():
    if isinstance(op, (Load, Store, Memset, Pointer)):
      found.setdefault(op.buffer.name, op.buffer)
    elif isinstance(op, Call) and op.states():
      for buffer in op.touches():
        found.setdefault(buffer.name, buffer)
  return found
