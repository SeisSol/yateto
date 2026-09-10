"""What the graph is taken to look like, checked rather than hoped.

Propagating a copy and dropping a dead one rest on three things being true of
every temporary: it has one definition, or one definition and a run of
accumulations after it; nothing views it; and it is read only where it was
written. All three hold because of the way the graph is built and nowhere
because anything says so, which is the awkward part -- a pass that broke one
would not fail, it would quietly make the bookkeeping wrong.

So they are asked. What comes back is what the graph does not look like, one
line per finding, and a graph nothing is said about is one the passes may
reason about the easy way.
"""

import collections

from .graph import Variable

#: A substitution that replaces nothing: no variable is named this, so asking
#: whether it may be made reduces to asking whether the statement stands up as
#: it is.
_NOTHING = Variable('', False, None)


def verify(cfg):
  """The findings, or an empty list where the graph is as it is taken to be."""
  return _generatable(cfg) + _definitions(cfg) + _views(cfg) + _guards(cfg)


def _generatable(cfg):
  """Every statement is one that can still be generated.

  Three things are asked before a rewrite is made: that every operand is read
  from storage keeping the entries it has values at, that the destination
  keeps the entries the statement writes, and that a contraction's operands
  are still matrices. They are asked here of what came out of the rewrite,
  and by the same rule -- a substitution that replaces nothing reduces
  `maySubstitute` to exactly those three. So a rewrite that should not have
  been made is caught by the rule that was meant to forbid it, rather than by
  the numbers being wrong somewhere else.
  """
  return [f'the statement at {position} does not stand up as it is'
          for position, action in enumerate(cfg)
          if not action.maySubstitute(_NOTHING, _NOTHING)]


def _definitions(cfg):
  """A temporary is defined once, and accumulated into after that at most.

  A run of accumulations is one definition in several steps, and the code
  generators rest on it being a run: a product accumulating into what is
  already there writes with beta = 1 rather than into a buffer of its own.
  Something reading it in between would be reading half a definition.
  """
  written = collections.defaultdict(list)
  for position, action in enumerate(cfg):
    if action.result.isLocal():
      written[action.result.name].append(position)

  findings = []
  for name in sorted(written):
    positions = written[name]
    if cfg[positions[0]].add or not all(cfg[position].add
                                        for position in positions[1:]):
      findings.append(
        f'"{name}" is written at {positions} and that is not one definition '
        f'followed by accumulations into it')
      continue
    steps = set(positions)
    between = [position for position in range(positions[0] + 1, positions[-1])
               if position not in steps and _reads(cfg[position], name)]
    if between:
      findings.append(
        f'"{name}" is read at {between}, between the steps that define it')

  for position, action in enumerate(cfg):
    for var in action.term.variables():
      if var.isLocal() and var.name not in written:
        findings.append(f'"{var}" is read at {position} and never written')
  return findings


def _views(cfg):
  """Nothing views a temporary.

  A view names a slice of the same storage, so a temporary that is viewed is
  read and written under two names. What may be substituted for it is then a
  question about both, and nothing here asks it.
  """
  findings = []
  for position, action in enumerate(cfg):
    for var in _operands(action) + [action.result]:
      if var.isView() and var.viewed().isLocal():
        findings.append(f'"{var}" is a view of a temporary, at {position}')
  return findings


def _guards(cfg):
  """A temporary is read only where it was written.

  The whole right-hand side of a guarded assignment is built under that
  guard, so its temporaries are written and read under it and nowhere else.
  Reading one somewhere the guard need not hold would be reading what may
  never have been written, and would want a decision at the join.
  """
  written = {}
  for action in cfg:
    if action.result.isLocal():
      written.setdefault(action.result.name, action.getGuard())

  findings = []
  for position, action in enumerate(cfg):
    guard = action.getGuard()
    for var in action.term.variables():
      if var.isLocal() and var.name in written \
         and not guard.implies(written[var.name]):
        findings.append(
          f'"{var}" is read at {position} under a guard that does not imply '
          f'the one it was written under')
  return findings


def _operands(action):
  return action.term.variableList() if action.isRHSExpression() else [action.term]


def _reads(action, name):
  if action.add and action.result.name == name:
    return True
  return any(var.name == name for var in action.term.variables())
