import re
import itertools
import json
from . import Collection, Tensor, TensorAttributes
from .memory import AlignedCSCMemoryLayout, CSCMemoryLayout, DenseMemoryLayout, PatternMemoryLayout, AlignedPatternMemoryLayout
from . import aspp
from .util import create_collection
import os
import lzma

import importlib.util
lxmlSpec = importlib.util.find_spec('lxml')
etreeSpec = importlib.util.find_spec('lxml.etree') if lxmlSpec else None
if etreeSpec is not None:
  import lxml.etree as etree
else:
  import xml.etree.ElementTree as etree

def __transposeMatrix(matrix):
  matrixT = dict()
  for entry,value in matrix.items():
    matrixT[tuple(entry[::-1])] = value
  return matrixT

def __checkedAttributes(fileName, attrs):
  """The attributes asked for per matrix, checked before anything is read.

  Keyed by the name a matrix has in the file, as `clones` is, so one entry
  covers every tensor made from that matrix -- the family a clone list makes
  of it included. A mapping rather than a function of the name, as
  `transpose` and `alignStride` are: a mapping can be checked against the
  file, and a claim made for a matrix the file does not contain attaches to
  nothing, which nothing would report.
  """
  checked = dict()
  for name, given in (attrs or {}).items():
    try:
      checked[name] = TensorAttributes(given).as_dict()
    except ValueError as error:
      raise ValueError(f'{fileName}, matrix {name!r}: {error}') from error
  return checked

def __checkAllNamed(fileName, attrs, named):
  """Refuses attributes for a matrix the file does not contain."""
  missing = sorted(set(attrs) - set(named))
  if missing:
    raise ValueError(
      '{} contains no matrix named {}, so the attributes given for it would '
      'attach to nothing. Name a matrix as the file does -- a clone by the '
      'matrix it is cloned from; the file has {}.'.format(
        fileName, ', '.join(repr(name) for name in missing), ', '.join(sorted(named))))

def __processMatrix(name, shape, entries, clones, transpose, alignStride, namespace, legacy, attrs=None):
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
    if len(shape) == 2 and shape[1] == 1 and legacy: # TODO: remove once all files are converted
      shape = (shape[0],)

    # transpose matrix if it is needed
    mtx = __transposeMatrix(matrix) if transpose(name) else matrix

    # adjust layout description in case if a given matrix is a vector
    if len(shape) == 1 and legacy: # TODO: remove once all files are converted
      mtx = {(i[0],): val for i,val in mtx.items()}

    # Create an tensor(matrix) using the matrix description and append the hash table
    if name in matrices:
      raise RuntimeError(f'{name} cannot be added to a collection a second time. '
                         'Please, check your input file and rename')

    matrices[name] = Tensor(name=name,
                            shape=tuple(shape),
                            spp=mtx,
                            alignStride=alignStride(name),
                            namespace=namespace,
                            attrs=attrs)
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

def parseXMLMatrixFile(xmlFile, clones=dict(), transpose=lambda name: False, alignStride=lambda name: False, namespace=None, attrs=None):
  """The matrices of an XML file, as a collection of tensors.

  `attrs` maps a matrix's name in the file to the attributes of every tensor
  made from it, as `parseJSONMatrixFile` takes them.
  """
  attrs = __checkedAttributes(xmlFile, attrs)
  with openMaybeCompressed(xmlFile) as file:
    root = etree.fromstring(file.read())

  matrices = dict()
  named = []

  for node in root:
    if node.tag == 'matrix':
      name = node.get('name')
      named.append(name)
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

      matrices.update( __processMatrix(name, (rows, columns), entries, clones, transpose, alignStride, namespace, True, attrs.get(name)) )
    else:
      __complain(node)

  __checkAllNamed(xmlFile, attrs, named)
  return create_collection(matrices)

def parseJSONMatrixFile(jsonFile, clones=dict(), transpose=lambda name: False, alignStride=lambda name: False, namespace=None, attrs=None):
  """The matrices of a JSON file, as a collection of tensors.

  `attrs` maps a matrix's name in the file to the attributes of every tensor
  made from it (see `TensorAttributes`): ``{'star': {'temporal':
  'invariant'}}``, with ``star`` cloned into ``star(0)`` to ``star(2)``,
  gives all three. The parse is where these tensors are declared, so that is
  where they are given, and a name the file does not contain is refused.
  """
  attrs = __checkedAttributes(jsonFile, attrs)
  matrices = dict()
  named = []

  with openMaybeCompressed(jsonFile) as j:
    content = json.load(j)
    for m in content:
      named.append(m['name'])
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
      matrices.update( __processMatrix(m['name'], shape, entries, clones, transpose, alignStride, namespace, False, attrs.get(m['name'])) )

  __checkAllNamed(jsonFile, attrs, named)
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
      for block in matrix:
        # a <matrix> that declares its sparsity block by block is not read
        raise NotImplementedError(f'Block-wise sparsity of matrix "{name}" is not supported.')
      names = groups[group] if group in groups else (clones[name] if name in clones else [name])
      for n in names:
        tensor = db.byName(n)

        # for backwards compatibility reasons, take CSC if we have a proper matrix at hand
        if sparsealigned:
          if tensor.memoryLayout().alignedStride():
            tensor.setMemoryLayout(AlignedCSCMemoryLayout if tensor.rank() == 2 else AlignedPatternMemoryLayout)
          else:
            tensor.setMemoryLayout(CSCMemoryLayout if tensor.rank() == 2 else PatternMemoryLayout)
        elif sparse:
          tensor.setMemoryLayout(CSCMemoryLayout if tensor.rank() == 2 else PatternMemoryLayout)
        else:
          tensor.setMemoryLayout(DenseMemoryLayout, alignStride=tensor.memoryLayout().alignedStride())
    elif strict:
      raise ValueError('Unrecognized matrix name ' + name)
