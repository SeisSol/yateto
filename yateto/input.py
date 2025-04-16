import re
import itertools
import json
from . import Collection, Tensor
from .memory import AlignedCSCMemoryLayout, CSCMemoryLayout, DenseMemoryLayout
from . import aspp
from .util import create_collection
import os
import lzma

import importlib.util
lxmlSpec = importlib.util.find_spec('lxml')
etreeSpec = importlib.util.find_spec('lxml.etree') if lxmlSpec else None
if etreeSpec:
  etree = etreeSpec.loader.load_module()
else:
  etree = importlib.util.find_spec('xml.etree.ElementTree').loader.load_module()

def __transposeMatrix(matrix):
  matrixT = dict()
  for entry,value in matrix.items():
    matrixT[tuple(entry[::-1])] = value
  return matrixT

def __processMatrix(name, shape, entries, clones, transpose, alignStride, namespace=None):
  matrix = dict()

  dim = len(shape)

  # traverse a list of matrix entries and generate a matrix description
  # as a hash table
  for entry in entries:
    # adjust row and column numbers
    index = tuple(int(entry[i]) - 1 for i in range(dim))

    # allocate a matrix element inside of a table
    matrix[index] = entry[-1]

  # allocate an empty hash table to hold tensors (matrices) which are going to be generated
  # using the matrix description
  matrices = dict()

  # create target tensors names
  names = clones[name] if name in clones else [name]

  # generate tensors using description of a give matrix
  for name in names:
    # compute a shape of a tensor (for now, assume transpose == invert dimensions)
    shape = shape[::-1] if transpose(name) else shape
    if len(shape) == 2 and shape[1] == 1: # TODO: remove once all files are converted
      shape = (shape[0],)

    # transpose matrix if it is needed
    mtx = __transposeMatrix(matrix) if transpose(name) else matrix

    # adjust layout description in case if a given matrix is a vector
    if len(shape) == 1: # TODO: remove once all files are converted
      mtx = {(i[0],): val for i,val in mtx.items()}

    # Create an tensor(matrix) using the matrix description and append the hash table
    if name in matrices:
      raise RuntimeError(f'{name} cannot be added to a collection a second time. '
                         'Please, check your input file and rename')

    matrices[name] = Tensor(name=name,
                            shape=tuple(shape),
                            spp=mtx,
                            alignStride=alignStride(name),
                            namespace=namespace)
  return matrices

def __complain(child):
  raise ValueError('Unknown tag ' + child.tag)

def openMaybeCompressed(basefilename):
  if os.path.exists(basefilename):
    return open(basefilename)
  elif os.path.exists(basefilename + '.xz'):
    return lzma.open(basefilename + '.xz')
  else:
    raise FileNotFoundError(basefilename)

def parseXMLMatrixFile(xmlFile, clones=dict(), transpose=lambda name: False, alignStride=lambda name: False, namespace=None):
  with openMaybeCompressed(xmlFile) as file:
    root = etree.fromstring(file.read())
  
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

      matrices.update( __processMatrix(name, (rows, columns), entries, clones, transpose, alignStride, namespace) )
    else:
      __complain(node)

  return create_collection(matrices)

def parseJSONMatrixFile(jsonFile, clones=dict(), transpose=lambda name: False, alignStride=lambda name: False, namespace=None):
  matrices = dict()

  with openMaybeCompressed(jsonFile) as j:
    content = json.load(j)
    for m in content:
      entries = m['entries']
      if 'rows' in m:
        shape = [m['rows']]
        if 'columns' in m:
          shape += [m['columns']]
      else:
        shape = m['shape']
      dim = len(shape)
      if len(next(iter(entries))) == dim:
        entries = [(*entry, True) for entry in entries]
      matrices.update( __processMatrix(m['name'], shape, entries, clones, transpose, alignStride, namespace) )

  return create_collection(matrices)

def memoryLayoutFromFile(xmlFile, db, clones, strict=False):
  with openMaybeCompressed(xmlFile) as file:
    root = etree.fromstring(file.read())
  
  strtobool = ['yes', 'true', '1']
  groups = dict()

  for group in root.findall('group'):
    groupName = group.get('name')
    noMutualSparsityPattern = group.get('noMutualSparsityPattern', '').lower() in strtobool
    groups[groupName] = list()
    for matrix in group:
      if matrix.tag == 'matrix':
        matrixName = matrix.get('name')
        if not db.containsName(matrixName) and strict:
          raise ValueError('Unrecognized matrix name ' + matrixName)
        if len(groups[groupName]) > 0:
          lastMatrixInGroup = groups[groupName][-1]
          if db.byName(lastMatrixInGroup).shape() != db.byName(matrixName).shape():
            raise ValueError('Matrix {} cannot be in the same group as matrix {} due to different shapes.'.format(matrixName, lastMatrixInGroup))
        groups[groupName].append( matrixName )
      else:
        __complain(group)
    # equalize sparsity pattern
    if not noMutualSparsityPattern:
      spp = None
      for matrix in groups[groupName]:
        spp = aspp.add(spp, db.byName(matrix).spp()) if spp is not None else db.byName(matrix).spp()
      for matrix in groups[groupName]:
        db.byName(matrix).setGroupSpp(spp)

  for matrix in root.findall('matrix'):
    group = matrix.get('group')
    name = matrix.get('name')
    sparse = matrix.get('sparse', '').lower() in strtobool
    sparsealigned = matrix.get('sparse', '').lower() == 'aligned'

    if group in groups or name in clones or db.containsName(name):
      blocks = []
      for block in matrix:
        raise NotImplementedError
        if block.tag == 'block':
          startrow = int(block.get('startrow'))
          stoprow = int(block.get('stoprow'))
          startcol = int(block.get('startcol'))
          stopcol = int(block.get('stopcol'))
          blksparse = (block.get('sparse') == None and sparse) or block.get('sparse', '').lower() in strtobool
        else:
          __complain(block)
      names = groups[group] if group in groups else (clones[name] if name in clones else [name])
      for n in names:
        tensor = db.byName(n)
        if sparsealigned:
          if tensor.memoryLayout().alignedStride():
            tensor.setMemoryLayout(AlignedCSCMemoryLayout)
          else:
            tensor.setMemoryLayout(CSCMemoryLayout)
        elif sparse:
          tensor.setMemoryLayout(CSCMemoryLayout)
        else:
          tensor.setMemoryLayout(DenseMemoryLayout, alignStride=tensor.memoryLayout().alignedStride())
    elif strict:
      raise ValueError('Unrecognized matrix name ' + name)
