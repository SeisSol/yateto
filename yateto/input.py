import re
from lxml import etree
from .tensor import Collection, Tensor

def __complain(child):
  raise ValueError('Unknown tag ' + child.tag)

def __parseMatrix(node, clones):
  name = node.get('name')
  rows = int( node.get('rows') )
  columns = int( node.get('columns') )

  matrix = dict()
  for child in node:
    if child.tag == 'entry':
      row = int(child.get('row'))-1
      col = int(child.get('column'))-1
      entry = child.get('value', True)
      matrix[(row, col)] = entry
    else:
      self.__complain(child)

  matrices = dict()
  if name in clones:
    for clone in clones[name]:
      matrices[clone] = Tensor(clone, (rows, columns), matrix)
  else:
    matrices[name] = Tensor(name, (rows, columns), matrix)
  
  return matrices

def parseXMLMatrixFile(xmlFile, clones):
  tree = etree.parse(xmlFile)
  root = tree.getroot()
  
  matrices = dict()
  
  for child in root:
    if child.tag == 'matrix':
      matrices.update( __parseMatrix(child, clones) )
    else:
      __complain(child)
  
  
  maxIndex = dict()
  legalName = re.compile('^([a-zA-Z0-9]+)(\[([0-9]+)\])?$')
  for name, matrix in matrices.items():
    match = legalName.match(name)
    if match == None:
      raise ValueError('Illegal matrix name', name, 'in', xmlFile)
    if match.lastindex == 2:
      baseName = match.group(1)
      maxIndex[baseName] = max(maxIndex.get(baseName, 0), int(match.group(3)))
  
  collection = Collection()
  simpleKeys = matrices.keys() - maxIndex.keys()
  for key in simpleKeys:
    collection.__dict__[key] = matrices[key]
  
  for key, value in maxIndex.items():
    collection.__dict__[key] = [ matrices.get('{}[{}]'.format(key, i), None) for i in range(value+1)]
  
  return collection
