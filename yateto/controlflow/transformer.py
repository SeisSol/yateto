from .graph import *

class FindLiving(object):   
  def visit(self, cfg):
    cfg[-1].living = set()
    for i in reversed(range(len(cfg)-1)):
      cfg[i].living = (cfg[i+1].living - {cfg[i].action.result}) | cfg[i].action.variables()
    return cfg

class GreedyReorder(object):
  def visit(self, cfg):
    for i,v in enumerate(cfg):
      if v.action and not v.action.containsExpression():
        V = v.action.variables()
        for j in range(i, 0, -1):
          u = cfg[j-1]
          if u.action.result not in V and v.action.result not in u.action.variables():
            cfg[j], cfg[j-1] = cfg[j-1], cfg[j]
          else:
            break
    return cfg

class EliminateTemporaries(object):    
  def visit(self, cfg):
    n = len(cfg)
    for i in range(n-1):
      u = cfg[i]
      v = cfg[i+1]
      for term in u.action.simpleTerms():
        if term.writable and not u.action.result.isGlobal() and term not in v.living:
          by = term
          when = u.action.result
          for j in range(i,n-1):
            cfg[j].action = cfg[j].action.substituted(when, by)
          cfg = FindLiving().visit(cfg)
          break
    return cfg

class MergeActions(object):
  def visit(self, cfg):
    n = len(cfg)
    i = 0
    while i < n-2:
      u = cfg[i]
      v = cfg[i+1]
      if len(u.action.terms) == 1 and v.action.isPlusEquals() and len(v.action.terms) == 2:
        st = v.action.simpleTerms()
        st.remove(v.action.result)
        if u.action.result in st and v.action.result not in u.action.variables():
          nt = v.action.simpleTerms()
          nt.remove(u.action.result)
          nt.extend(u.action.terms)
          u.action = ProgramAction(v.action.result, *nt)
          del cfg[i+1]
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
      if not when.isGlobal() and when not in u.living:
        freeLocals = usedLocals - u.living
        try:
          by = next(iter(freeLocals))
          for j in range(i,n-1):
            cfg[j].action = cfg[j].action.substituted(when, by)
          cfg = FindLiving().visit(cfg)
        except StopIteration:
          pass
      usedLocals = usedLocals | u.action.localVariables()
    return cfg
