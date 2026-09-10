from .graph import *


class GraphPass(object):
  """A rewrite of the graph, and how much of it there was.

  Worth counting, because a pass that never rewrites anything is one the
  others have taken over -- and moving such a pass anywhere is moving
  nothing.
  """

  def __init__(self):
    self.rewrites = 0


class MergeScalarMultiplications(GraphPass):
  def visit(self, cfg):
    n = len(cfg)
    i = 1
    while i < n:
      ua = cfg[i]
      if ua.isCopy() and not ua.isCompound() and ua.scalar is not None:
        va = cfg[i-1]
        if not va.isCopy() and not va.isCompound() and ua.copied() == va.result:
          va.scalar = ua.scalar
          va.result = ua.result
          # the merged action now performs ua's store, so it inherits ua's guard
          va.condition = va.getGuard() & ua.getGuard()
          self.rewrites += 1
          del cfg[i]
          i -= 1
          n -= 1
      i += 1
    return cfg

def liveness(cfg):
  """Per position, which variables are live before the statement standing there.

  One entry more than there are statements: the last says what is live once
  the kernel is done, which is nothing.
  """
  live = [None] * len(cfg) + [LiveSet({})]
  for i in reversed(range(len(cfg))):
    action = cfg[i]
    guard = action.getGuard()
    at = live[i+1] - {action.result: guard}
    at = at | {var: guard for var in action.variables()}
    # the guard has to be read to decide the branch, so its variables are
    # live regardless of the outcome
    at = at | {var: Guard.always() for var in action.guardVariables()}
    live[i] = at
  return live

def maySubstitute(statement, when, by, result=True, term=True):
  """Whether `when` may be read from `by` in this statement.

  Rewriting a statement is the business of whoever rewrites it, and so is
  asking whether it may be: what comes out has to be a statement that stands
  up, and that is what is asked, of what would come out.
  """
  operands = [operand.substituted(when, by) for operand in statement.operands]
  layouts = [operand.memoryLayout for operand in operands]
  readable = all(layouts[i].isCompatible(operand.eqspp)
                 for i, operand in enumerate(statement.operands)) \
             and statement.mayReadOperands(layouts)
  writable = statement.result.maySubstitute(when, by)

  rsubs = statement.result.substituted(when, by) if result else statement.result

  # asked of the statement as it stands: what decides whether its value fits a
  # destination -- the indices it is stated over and the entries it has values
  # at -- is not what a substitution changes
  return (not term or readable) and (not result or writable) \
     and statement.resultCompatible(rsubs)


def substituted(statement, when, by, guard=None, result=True, term=True):
  """The statement with `when` read from `by`.

  `guard` is passed only when the substitution redirects the statement's
  *write target* onto a variable another statement writes under that guard;
  the statement then inherits it. A read substitution leaves the guard alone --
  conjoining there would restrict statements that are not themselves
  conditional.
  """
  rsubs = statement.result.substituted(when, by) if result else statement.result
  operands = [operand.substituted(when, by) for operand in statement.operands] \
             if term else statement.operands
  gsubs = statement.condition if guard is None else (statement.getGuard() & guard)
  return ProgramAction(statement.kind, rsubs, operands, statement.indices,
                       statement.eqspp, statement.add, statement.scalar, gsubs,
                       **{name: getattr(statement, name)
                          for name in ProgramAction._FACTS})


def _dropIfEmptied(cfg, position):
  """Drop the statement a substitution has just left saying nothing.

  Substituting a variable by the one it was copied from turns the copy into
  ``x = x``, and that is not a statement -- it is what is left where one was.
  Dropped by whoever made it, so that no other pass ever reasons about a
  graph holding one: something that writes and reads the same name in one
  step is a shape none of them is written for.
  """
  action = cfg[position]
  if not action.isCompound() and action.isCopy() \
     and action.result == action.copied() and action.hasTrivialScalar():
    del cfg[position]
    return True
  return False


def _guardsCompatible(cfg, rng, definition):
  """Every touched action must run at least as restrictively as `definition`.

  Otherwise the substituted variable may be read where it was never written.
  """
  return all(cfg[j].getGuard().implies(definition) for j in rng)

class SubstituteForward(GraphPass):
  def visit(self, cfg):
    live = liveness(cfg)
    i = 0
    while i < len(cfg):
      n = len(cfg)
      ua = cfg[i]

      if not ua.isCompound() \
          and ua.isCopy() \
          and ua.copied().writable \
          and ua.result.isLocal() \
          and (ua.copied(), ua.getGuard()) not in live[i+1] \
          and (ua.hasTrivialScalar() or ua.copied().isLocal()):

        when = ua.result
        by = ua.copied()
        maySubs = all([maySubstitute(cfg[j], when, by) for j in range(i, n)]) \
                  and _guardsCompatible(cfg, range(i, n), ua.getGuard())
        if maySubs:
          for j in range(i, n):
            # a read substitution; the downstream guards stay as they are
            cfg[j] = substituted(cfg[j], when, by)
          self.rewrites += 1
          if _dropIfEmptied(cfg, i):
            # what stands here now is the next statement, not the one after it
            live = liveness(cfg)
            continue
          live = liveness(cfg)
      i += 1

    return cfg

class SubstituteBackward(GraphPass):
  def visit(self, cfg):
    n = len(cfg)
    live = liveness(cfg)
    for i in reversed(range(n)):
      va = cfg[i]
      if not va.isCompound() and va.isCopy() and va.copied().isLocal():
        by = va.result
        found = -1
        for j in range(i):
          if (by, va.getGuard()) not in live[j] and not cfg[j].isCompound() \
             and cfg[j].result == va.copied():
            found = j
            break
        if found >= 0:
          when = cfg[found].result
          maySubs = maySubstitute(cfg[found], when, by, term=False) \
                    and all([maySubstitute(cfg[j], when, by) for j in range(found+1,i+1)]) \
                    and _guardsCompatible(cfg, range(found, i+1), va.getGuard())
          if maySubs:
            # only the producing action changes its write target and hence
            # inherits va's guard; the remaining ones merely read `by`
            cfg[found] = substituted(cfg[found], when, by, va.getGuard(), term=False)
            for j in range(found+1,i+1):
              cfg[j] = substituted(cfg[j], when, by)
            self.rewrites += 1
            _dropIfEmptied(cfg, i)
            live = liveness(cfg)
    return cfg

class MergeActions(GraphPass):
  def visit(self, cfg):
    n = len(cfg)
    i = 0
    while i < n:
      ua = cfg[i]
      if not ua.isCompound():
        found = -1
        V = ua.allVariables()
        for j in range(i+1,n):
          va = cfg[j]
          if va.isCopy() \
              and ua.result == va.copied() \
              and va.result not in V \
              and (ua.hasTrivialScalar() or va.hasTrivialScalar()) \
              and ua.result.isLocal():
            found = j
            break
          elif ua.result in va.allVariables() or ua.result == va.result:
            break
          else:
            V = V | va.allVariables() | {va.result}
        if found >= 0:
          va = cfg[found]
          if maySubstitute(ua, ua.result, va.result, term=False):
            # this action's write target becomes va's, so it inherits va's guard
            cfg[i] = substituted(ua, ua.result, va.result, va.getGuard(), term=False)
            cfg[i].add = va.add
            if not va.hasTrivialScalar():
              cfg[i].scalar = va.scalar
            self.rewrites += 1
            del cfg[found]
            n -= 1
      i += 1
    return cfg
