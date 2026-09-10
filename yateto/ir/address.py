from ..memory import MemoryLayoutView
from .affine import Affine


def address(memoryLayout, coords):
  """The address of an entry, as an affine expression over the loop indices.

  Two coordinate systems meet here. The coordinates name an entry in the
  logical space the operand is written in; the address is an offset into the
  storage. A view sits between the two and shifts one axis, so it is peeled
  off by shifting the coordinate instead, which leaves a layout that answers
  in its own space.

  A sparse layout has no affine address: where an entry sits is a lookup, and
  it only has an answer for coordinates that are known. Asking with an index
  still in them is an error rather than an approximation.
  """
  coords = [Affine.of(coord) for coord in coords]

  if isinstance(memoryLayout, MemoryLayoutView):
    shifted = list(coords)
    shifted[memoryLayout.index] = coords[memoryLayout.index] + memoryLayout.start
    return address(memoryLayout.base, shifted)

  if memoryLayout.isSparse():
    entry = constantEntry(coords)
    if entry is None:
      raise ValueError(
        f'{type(memoryLayout).__name__} stores its entries in an order only it '
        f'knows, so an address in it exists for a coordinate that is a number, '
        f'not for one that is still an index. Unroll the loop first.')
    return Affine(memoryLayout.address(entry))

  if len(memoryLayout.shape()) == 0:
    return Affine(0)

  bbox = memoryLayout.bbox()
  stride = memoryLayout.stride()
  result = Affine(0)
  for position, coord in enumerate(coords):
    result = result + stride[position] * (coord - bbox[position].start)
  return result


def constantEntry(coords):
  """The coordinates as a tuple of numbers, or None if any is still an index."""
  coords = [Affine.of(coord) for coord in coords]
  if not all(coord.isConstant() for coord in coords):
    return None
  return tuple(coord.constant() for coord in coords)


def storesValue(memoryLayout, coords):
  """Whether the layout keeps the entry at `coords` at all.

  Answered for constant coordinates only; anything else is assumed stored,
  since a loop that runs over entries the layout skips is a question for the
  pass that built the loop.
  """
  entry = constantEntry(coords)
  if entry is None:
    return True
  return memoryLayout.hasValue(entry)
