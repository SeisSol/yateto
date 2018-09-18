import re
import json
from lxml import etree
from .tensor import Collection, Tensor

import importlib.util
lxmlSpec = importlib.util.find_spec('lxml')
if lxmlSpec:
  lxml = lxmlSpec.loader.load_module()

def __createCollection(matrices):
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

def __processMatrix(name, rows, columns, entries, clones, transpose, alignStride):  
  if transpose:
    rows, columns = columns, rows

  matrix = dict()
  for entry in entries:
    row = int(entry[0])-1
    col = int(entry[1])-1
    if transpose:
      matrix[(col, row)] = entry[2]
    else:
      matrix[(row, col)] = entry[2]

  matrices = dict()
  if name in clones:
    for clone in clones[name]:
      matrices[clone] = Tensor(clone, (rows, columns), matrix, alignStride=alignStride)
  else:
    matrices[name] = Tensor(name, (rows, columns), matrix, alignStride=alignStride)
  return matrices

def __complain(child):
  raise ValueError('Unknown tag ' + child.tag)

def parseXMLMatrixFile(xmlFile, clones=dict(), transpose=False, alignStride=None):
  if lxmlSpec is None:
    raise RuntimeError('LXML module was not found. parseXMLMatrixFile is unavailable.')

  tree = lxml.etree.parse(xmlFile)
  root = tree.getroot()
  
  matrices = dict()
  
  for node in root:
    if node.tag == 'matrix':
      name = node.get('name')
      rows = int( node.get('rows') )
      columns = int( node.get('columns') )

      entries = list()
      for child in node:
        if child.tag == 'entry':
          row = int(child.get('row'))
          col = int(child.get('column'))
          entry = child.get('value', True)
          entries.append((row,col,entry))
        else:
          __complain(child)

      matrices.update( __processMatrix(name, rows, columns, entries, clones, transpose, alignStride) )
    else:
      __complain(node)

  return __createCollection(matrices)

def parseJSONMatrixFile(jsonFile, clones=dict(), transpose=False, alignStride=None):
  matrices = dict()

  with open(jsonFile) as j:
    content = json.load(j)
    for m in content:
      entries = m['entries']
      if len(next(iter(entries))) == 2:
        entries = [(entry[0], entry[1], True) for entry in entries]
      matrices.update( __processMatrix(m['name'], m['rows'], m['columns'], entries, clones, transpose, alignStride) )

  return __createCollection(matrices)
  
