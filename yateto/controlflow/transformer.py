from .graph import *

class FindLiving(object):   
  def visit(self, cfg):
    cfg[-1].living = set()
    for i in reversed(range(len(cfg)-1)):
      cfg[i].living = (cfg[i+1].living - {cfg[i].action.result}) | cfg[i].action.variables()
    return cfg

class SubstituteForward(object):
  def visit(self, cfg):
    n = len(cfg)-1
    for i in range(n):
      ua = cfg[i].action
      v = cfg[i+1]
      if not ua.isCompound() and ua.isRHSVariable() and ua.term.writable and ua.result.isLocal() and ua.term not in v.living:
        when = ua.result
        by = ua.term
        for j in range(i, n):
          cfg[j].action = cfg[j].action.substituted(when, by)
        cfg = FindLiving().visit(cfg)
    return cfg

class SubstituteBackward(object):
  def visit(self, cfg):
    n = len(cfg)-1
    for i in reversed(range(n)):
      va = cfg[i].action
      if not va.isCompound() and va.isRHSVariable() and va.term.isLocal():
        by = va.result
        found = -1
        for j in range(i):
          u = cfg[j]
          if by not in u.living and not u.action.isCompound() and u.action.result == va.term:
            found = j
            break
        if found >= 0:
          when = u.action.result
          cfg[found].action = cfg[found].action.substituted(when, by, term=False)
          for j in range(found+1,i+1):
            cfg[j].action = cfg[j].action.substituted(when, by)
          cfg = FindLiving().visit(cfg)
    return cfg

class RemoveEmptyStatements(object):
  def visit(self, cfg):
    n = len(cfg)-1
    i = 0
    while i < n:
      ua = cfg[i].action
      if not ua.isCompound() and ua.isRHSVariable() and ua.result == ua.term:
        del cfg[i]
        n -= 1
      else:
        i += 1
    return cfg

class MergeActions(object):
  def visit(self, cfg):
    n = len(cfg)-1
    i = 0
    while i < n:
      ua = cfg[i].action
      if not ua.isCompound():
        found = -1
        V = ua.variables()
        for j in range(i+1,n):
          va = cfg[j].action
          if va.isRHSVariable() and ua.result == va.term and va.result not in V:
            found = j
            break
          elif ua.result in va.variables() or ua.result == va.result:
            break
          else:
            V = V | va.variables()
        if found >= 0:
          va = cfg[j].action
          ua.result = va.result
          ua.add = va.add
          del cfg[j]
          n -= 1
      i += 1
    return FindLiving().visit(cfg)

class ReuseTemporaries(object):
  def visit(self, cfg):
    usedLocals = set()
    n = len(cfg)
    for i in range(n-1):
      u = cfg[i]
      v = cfg[i+1]
      when = u.action.result
      if not when.isGlobal():
        if when not in u.living:
          freeLocals = usedLocals - u.living
          try:
            by = next(iter(freeLocals))
            for j in range(i,n-1):
              cfg[j].action = cfg[j].action.substituted(when, by)
            cfg = FindLiving().visit(cfg)
          except StopIteration:
            pass
        usedLocals = usedLocals | u.action.result.variables()
    return cfg
