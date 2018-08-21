from functools import singledispatch
from .node import Op

def addIndent(string, indent):
  return '\n'.join([indent + line for line in string.splitlines()])

@singledispatch
def pprint(node, indent=''):
  print(indent + str(node))

@pprint.register(Op)
def _(node, indent=''):
  print(indent + str(node))
  for child in node:
    pprint(child, indent + '  ')
