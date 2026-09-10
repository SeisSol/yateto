from .graph import *


class MergeScalarMultiplications(object):
  def visit(self, cfg):
    n = len(cfg)
    i = 1
    while i < n:
      ua = cfg[i]
      if ua.isRHSVariable() and not ua.isCompound() and ua.scalar is not None:
        va = cfg[i-1]
        if va.isRHSExpression() and not va.isCompound() and ua.term == va.result:
          va.scalar = ua.scalar
          va.result = ua.result
          # the merged action now performs ua's store, so it inherits ua's guard
          va.condition = va.getGuard() & ua.getGuard()
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

def _guardsCompatible(cfg, rng, definition):
  """Every touched action must run at least as restrictively as `definition`.

  Otherwise the substituted variable may be read where it was never written.
  """
  return all(cfg[j].getGuard().implies(definition) for j in rng)

class SubstituteForward(object):
  def visit(self, cfg):
    n = len(cfg)
    live = liveness(cfg)
    for i in range(n):
      ua = cfg[i]

      if not ua.isCompound() \
          and ua.isRHSVariable() \
          and ua.term.writable \
          and ua.result.isLocal() \
          and (ua.term, ua.getGuard()) not in live[i+1] \
          and (ua.hasTrivialScalar() or ua.term.isLocal()):

        when = ua.result
        by = ua.term
        maySubs = all([cfg[j].maySubstitute(when, by) for j in range(i, n)]) \
                  and _guardsCompatible(cfg, range(i, n), ua.getGuard())
        if maySubs:
          for j in range(i, n):
            # a read substitution; the downstream guards stay as they are
            cfg[j] = cfg[j].substituted(when, by)
          live = liveness(cfg)

    return cfg

class SubstituteBackward(object):
  def visit(self, cfg):
    n = len(cfg)
    live = liveness(cfg)
    for i in reversed(range(n)):
      va = cfg[i]
      if not va.isCompound() and va.isRHSVariable() and va.term.isLocal():
        by = va.result
        found = -1
        for j in range(i):
          if (by, va.getGuard()) not in live[j] and not cfg[j].isCompound() \
             and cfg[j].result == va.term:
            found = j
            break
        if found >= 0:
          when = cfg[found].result
          maySubs = cfg[found].maySubstitute(when, by, term=False) \
                    and all([cfg[j].maySubstitute(when, by) for j in range(found+1,i+1)]) \
                    and _guardsCompatible(cfg, range(found, i+1), va.getGuard())
          if maySubs:
            # only the producing action changes its write target and hence
            # inherits va's guard; the remaining ones merely read `by`
            cfg[found] = cfg[found].substituted(when, by, va.getGuard(), term=False)
            for j in range(found+1,i+1):
              cfg[j] = cfg[j].substituted(when, by)
            live = liveness(cfg)
    return cfg

class RemoveEmptyStatements(object):
  def visit(self, cfg):
    n = len(cfg)
    i = 0
    while i < n:
      ua = cfg[i]
      if not ua.isCompound() and ua.isRHSVariable() and ua.result == ua.term and ua.hasTrivialScalar():
        del cfg[i]
        n -= 1
      else:
        i += 1
    return cfg

class MergeActions(object):
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
          if va.isRHSVariable() \
              and ua.result == va.term \
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
          if ua.maySubstitute(ua.result, va.result, term=False):
            # this action's write target becomes va's, so it inherits va's guard
            cfg[i] = ua.substituted(ua.result, va.result, va.getGuard(), term=False)
            cfg[i].add = va.add
            if not va.hasTrivialScalar():
              cfg[i].scalar = va.scalar
            del cfg[found]
            n -= 1
      i += 1
    return cfg
