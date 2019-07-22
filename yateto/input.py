import re
import itertools
import json
from . import Collection, Tensor
from .memory import CSCMemoryLayout, DenseMemoryLayout
from . import aspp

import importlib.util


lxmlSpec = importlib.util.find_spec('lxml')
etreeSpec = importlib.util.find_spec('lxml.etree') if lxmlSpec else None


if etreeSpec:
  etree = etreeSpec.loader.load_module()
else:
  etree = importlib.util.find_spec('xml.etree.ElementTree').loader.load_module()


def __createCollection(matrices):
  """Creates a collection (a table of tables) from a given table of tensors.

  A collection holds tensors in a hierarchical
  i.e. base names followed by tensor groups

  Args:
    matrices (Dict[str, Tensor]): a table of tensors

  Returns:
    Collection (Dict[str, Dict[int, Tensor]]): a table of tables

  Raises:
    ValueError: if a matrix tensor name is illegal according yateto specification.
                See the definition of type.Tensor class.
  """

  collection = Collection()

  for name, matrix in matrices.items():

    if not Tensor.isValidName(name):
      raise ValueError('Illegal matrix name: {}'.format(name))

    base_name = Tensor.getBaseName(name)
    group = Collection.group(name)

    if group is tuple():
      collection[base_name] = matrix

    else:
      if base_name in collection:
        collection[base_name][group] = matrix

      else:
        collection[base_name] = {group: matrix}

  return collection


def __transposeMatrix(matrix):
  """Transposes a matrix

  Args:
    matrix (Dict[Tuple[int, int], float]): sparse matrix specification as a hash table

  Returns:
    Dict[Tuple[int, int], float]: transpose matrix description
  """

  transposed_matrix = dict()

  for entry, value in matrix.items():
    transposed_matrix[(entry[1], entry[0])] = value

  return transposed_matrix


def __processMatrix(name, rows, columns, entries, clones, transpose, alignStride):
  """Creates a tensor (tensors) using passed parameters i.e. a matrix description

  Parameters are retrieved while processing an xml matrix-file. Using parameter 'clones',
  the function allows the user to either change the target name of a tensor or to generate
  multiple tensors using a given matrix description.

  Args:
    name (str): matrix name (which was retrived processing an xml file)
    rows (int): number of rows in a matrix
    columns (int): number of columns in  a matrix
    entries (List[Tuple]): a list of all entries retrieved from a matrix specified an xml file.
                           The list essentially is a coordinate sparse matrix form.
    clones (Dict[str, List[str]]): a table of target tensor names

    transpose (function): function takes a name of a matrix as an input and
                          returns a boolean flag which specifies whether or not
                          yateto has to transpose a matrix

    alignStride (function): function takes a name of a matrix as an input and
                            returns a boolean flag which specifies whether or not
                            yateto has to align a given matrix layout for vectorization

  Returns:
    Dict[str, Tensor]: a table of tensors
  """

  # create an empty hash table
  matrix = dict()

  # traverse a list of matrix entries and generate a matrix description
  # as a hash table
  for entry in entries:

    # adjust row and column numbers
    row = int(entry[0]) - 1
    col = int(entry[1]) - 1

    # allocate a matrix element inside of a table
    matrix[(row, col)] = entry[2]


  # allocate an empty hash table to hold tensors (matrices) which are going to be generated
  # using the matrix description
  matrices = dict()

  # create target tensors names
  names = clones[name] if name in clones else [name]

  # generate tensors using description of a give matrix
  for name in names:

    # compute a shape of a tensor
    shape = (columns, rows) if transpose(name) else (rows, columns)
    if shape[1] == 1:
      shape = (shape[0],)

    # transpose matrix if it is needed
    mtx = __transposeMatrix(matrix) if transpose(name) else matrix

    # adjust layout description in case if a given matrix is a vector
    if len(shape) == 1:
      mtx = {(i[0],): val for i, val in mtx.items()}

    # Create an tensor(matrix) using the matrix description and append the hash table
    matrices[name] = Tensor(name=name,
                            shape=shape,
                            spp=mtx,
                            alignStride=alignStride(name))
  return matrices


def __complain(child):
  """Raises an exception in case of a violation of the xml-matrix format specification

  Args:
    child: a node of an xml parse tree

  Returns:

  """
  raise ValueError('Unknown tag ' + child.tag)


def parseXMLMatrixFile(xml_file,
                       clones=dict(),
                       transpose=lambda name: False,
                       align_stride=lambda name: False):
  """Generates a table of tensors from an xml file containing matrix(tensors) structure and data

  The function reads an xml file and retrieve information about matrices specified in it.
  All matrices are given in sparse coordinate form assuming that the first matrix entry
  has indices 1,1 (instead of 0,0).

  Args:
    xml_file (str): path to a file
    clones (Dict[str, List[str]]): a table of target tensor names
    transpose (function): function takes a name of a matrix as an input and
                          returns a boolean value which specifies whether or not
                          a matrix has to be transposed

    align_stride (function): function takes a name of a matrix as an input and
                             returns a boolean flag which specifies whether or not
                             a given matrix layout has to be aligned for vectorization

  Returns:
    Collection: a table of tensors arranged by their base names and tensor groups
  """

  # parse a give xml file and build a parse tree
  tree = etree.parse(xml_file)
  root = tree.getroot()


  # allocate a dictionary to hold all matrices specified in the file
  # after processing a given xml file
  matrices = dict()

  # traverse a parse tree
  for node in root:

    # check whether a node of a tree describes a matrix.
    # Raise an exception if it is not a case
    if node.tag == 'matrix':

      # retrive name and size of a matrix
      name = node.get('name')
      rows = int(node.get('rows'))
      columns = int(node.get('columns'))

      # allocate a list to hold entries of a matrix specified in a node of a given xml file
      entries = list()

      # traverse children of a matrix node and collect data
      for child in node:

        # Abort execution of a node child is not denoted as a entry i.e. entry of matrix
        if child.tag == 'entry':

          # collect data i.e. row and column numbers as well as a matrix element value
          row = int(child.get('row'))
          col = int(child.get('column'))
          entry = child.get('value', True)

          # put all found matrix elements to a list
          entries.append((row, col, entry))
        else:
          __complain(child)

      # generate a tensor (tensors) using retrieved matrix description
      tensors = __processMatrix(name, rows, columns, entries, clones, transpose, align_stride)

      # append the dictionary with a generated tensor (tensors)
      matrices.update(tensors)

    else:
      __complain(node)

  try:
    collection = __createCollection(matrices)
  except ValueError as error:
    print("ERROR: {} in {}".format(error, xml_file))
    raise

  return collection


def parseJSONMatrixFile(json_file,
                        clones=dict(),
                        transpose=lambda name: False,
                        alignStride=lambda name: False):
  """Generates a table of tensors from an xml file containing matrix(tensors) structure and data

  The function reads an xml file and retrieve information about matrices specified in it.
  All matrices are given in sparse coordinate form assuming that the first matrix entry
  has indices 1,1 (instead of 0,0).


  Args:
    json_file: path to a file
    clones (Dict[str, List[str]]): a table of target tensor names
    transpose (function): function takes a name of a matrix as an input and
                          returns a boolean value which specifies whether or not
                          a matrix has to be transposed

    align_stride (function): function takes a name of a matrix as an input and
                             returns a boolean flag which specifies whether or not
                             a given matrix layout has to be aligned for vectorization

  Returns:
    Collection: a table of tensors arranged by their base names and tensor groups
  """
  matrices = dict()

  with open(json_file) as file:
    content = json.load(file)

    for matrix in content:
      entries = matrix['entries']

      if len(next(iter(entries))) == 2:
        entries = [(entry[0], entry[1], True) for entry in entries]

      tensors = __processMatrix(matrix['name'],
                                matrix['rows'],
                                matrix['columns'],
                                entries,
                                clones,
                                transpose,
                                alignStride)

      matrices.update(tensors)

  return __createCollection(matrices)


def memoryLayoutFromFile(xmlFile, db, clones):
  tree = etree.parse(xmlFile)
  root = tree.getroot()
  strtobool = ['yes', 'true', '1']
  groups = dict()

  for group in root.findall('group'):
    groupName = group.get('name')
    noMutualSparsityPattern = group.get('noMutualSparsityPattern', '').lower() in strtobool
    groups[groupName] = list()

    for matrix in group:
      if matrix.tag == 'matrix':
        matrixName = matrix.get('name')

        if not db.containsName(matrixName):
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
        if sparse:
          tensor.setMemoryLayout(CSCMemoryLayout)
        else:
          tensor.setMemoryLayout(DenseMemoryLayout, alignStride=tensor.memoryLayout().alignedStride())
    else:
      raise ValueError('Unrecognized matrix name ' + name)
