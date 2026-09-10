from ..memory import MemoryLayoutView
from .affine import Affine


def address(memoryLayout, coords, axes=None):
  """The address of an entry, as an affine expression over the loop indices.

  `axes` names the axes to count, for an address that is meant to move along
  some of them and stay put on the rest -- which is what a pointer handed to a
  kernel working on a subtensor is.

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

  if axes is not None and (len(axes) == 0 or memoryLayout.isSparse()):
    # Nothing moves, so the address is where the buffer starts. The same holds
    # of a sparse layout however many axes are given: where it is read is
    # decided entry by entry once the indices are numbers, and until then it
    # has no expression to move a pointer along.
    return Affine(0)

  if isinstance(memoryLayout, MemoryLayoutView):
    shifted = list(coords)
    shifted[memoryLayout.index] = coords[memoryLayout.index] + memoryLayout.start
    return address(memoryLayout.base, shifted, axes)

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
    if axes is not None and position not in axes:
      continue
    result = result + stride[position] * (coord - bbox[position].start)
  return result


def entry(buffer, coords):
  """Which entry of a buffer an access touches, as an offset into its storage.

  Two accesses touch one entry when they offset into one buffer by the same
  expression. The coordinates do not say so on their own: a view starts
  somewhere inside what it views, so two operands that are views into one
  buffer name the same coordinate and mean two different entries.

  None where there is no expression to compare -- a sparse layout reached by
  something that is still an index. Where in the buffer such an access lands
  is not known, and no two of them may be taken for the same entry.
  """
  try:
    return address(buffer.memoryLayout, coords)
  except ValueError:
    return None


def constantEntry(coords):
  """The coordinates as a tuple of numbers, or None if any is still an index."""
  coords = [Affine.of(coord) for coord in coords]
  if not all(coord.isConstant() for coord in coords):
    return None
  return tuple(coord.constant() for coord in coords)


def storesValue(memoryLayout, coords, eqspp=None):
  """Whether there is a value to read at `coords`.

  Two questions in one: whether the layout keeps the entry at all, and whether
  the operand has a value there. Room the layout keeps for an entry that is
  structurally zero holds whatever was last put in it, which is why the
  pattern is asked as well as the layout.

  Answered for constant coordinates only; anything else is assumed to have a
  value, since a loop that runs over entries the operand does not have is a
  question for the pass that built the loop.
  """
  entry = constantEntry(coords)
  if entry is None:
    return True
  if eqspp is not None:
    # The pattern is asked first and alone. It spans the operand's whole
    # shape, whereas a layout only answers within the box it keeps room in --
    # and an entry outside that box is exactly the case this is asked about.
    return bool(eqspp.as_ndarray()[entry])
  return memoryLayout.hasValue(entry)
