import collections
import hashlib

from ..memory import PreparedImage


def layoutTag(memoryLayout):
  """A short name for how one tensor is held.

  Over what an address depends on -- the kind of layout, the extents, the box,
  how much it occupies and what it promises about alignment. Two layouts that
  agree on all of it address alike; two that do not are named apart. A tag
  shared by two that differ is caught where the pool members are assigned
  rather than silently merged.
  """
  if isinstance(memoryLayout, PreparedImage):
    facts = memoryLayout.identity()
    digest = hashlib.new('md5', usedforsecurity=False)
    digest.update(str(facts).encode())
    return digest.hexdigest()[:8]
  storage = memoryLayout.storage()
  arch = storage.alignmentArch()
  facts = (type(storage).__name__,
           tuple(storage.shape()),
           tuple((rng.start, rng.stop) for rng in storage.bbox()),
           tuple(storage.stridei(axis) for axis in range(len(storage.shape())))
           if hasattr(storage, 'stridei') else (),
           storage.requiredReals(),
           storage.alignedStride(),
           0 if arch is None else arch.alignment)
  digest = hashlib.new('md5', usedforsecurity=False)
  digest.update(str(facts).encode())
  return digest.hexdigest()[:8]


class Arrangement(object):
  """How a family of tensors is held in memory, member by member.

  A tensor name stands for a family -- one member per group index -- and the
  pool holds the family as one table of pointers, one per member. What an
  address into it depends on is therefore every member's layout and not one of
  them: two members may well be laid out differently, with different extents or
  a different sparsity, and that is one arrangement of the family rather than
  two arrangements in conflict.

  Compared and named by tag, so that both sides work out from the tensors alone
  which member of `Pool` a family is held in, each where it stands.
  """

  __slots__ = ('_members',)

  def __init__(self, members):
    # By group, so that a family met in one order and the same family met in
    # another are one arrangement rather than two.
    self._members = collections.OrderedDict(
      sorted(members.items(), key=lambda item: item[0]))

  @classmethod
  def of(cls, tensors):
    """The arrangement a family gives itself."""
    return cls({group: tensor.memoryLayout() for group, tensor in tensors.items()})

  def groups(self):
    return list(self._members.keys())

  def layoutOf(self, group, default=None):
    """How that member is held, or `default` where this says nothing about it."""
    return self._members.get(group, default)

  def withMember(self, group, layout):
    """The same arrangement, with one member held differently."""
    members = collections.OrderedDict(self._members)
    members[group] = layout
    return Arrangement(members)

  def mapped(self, rearrange):
    """The same family, each member put through `rearrange`."""
    return Arrangement({group: rearrange(layout)
                        for group, layout in self._members.items()})

  def tag(self):
    """A short name for this arrangement of the family.

    Over the members' own tags, in group order. A family of one is still named
    over its single member rather than by that member's tag directly: the tag
    says how a family is held, and a family that gains a member is a
    differently held family.
    """
    facts = tuple((group, layoutTag(layout))
                  for group, layout in self._members.items())
    digest = hashlib.new('md5', usedforsecurity=False)
    digest.update(str(facts).encode())
    return digest.hexdigest()[:8]

  def __eq__(self, other):
    if not isinstance(other, Arrangement):
      return NotImplemented
    return self.tag() == other.tag()

  def __ne__(self, other):
    result = self.__eq__(other)
    return result if result is NotImplemented else not result

  def __hash__(self):
    return hash(self.tag())

  def __repr__(self):
    return 'Arrangement({})'.format(dict(self._members))
