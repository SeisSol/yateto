from typing import Tuple
import re
from .ast.node import Node, IndexedTensor
from numpy import ndarray, zeros
from .memory import DenseMemoryLayout
from . import aspp


class AbstractType(object):
  @classmethod
  def isValidName(cls, name):
    """Checks whether a name is valid for a particular subclass

    NOTE: it is a class methods

    Args:
      name (str): a name

    Returns:
      bool: True if it is valid. Otherwise, False
    """
    return re.match(cls.VALID_NAME, name) is not None


  def name(self):
    """Returns a name of an instance

    NOTE: every subclass of AbstractType must have a member called _name
    Refactoring: get_name

    Returns:
      str: name of a subclass instance
    """
    return self._name


class Scalar(AbstractType):

  # VALID_NAME is a sequence of any alphabet character
  #	  followed by any number of word characters
  VALID_NAME = r'^[a-zA-Z]\w*$'
  
  def __init__(self, name):
    """
    Args:
      name (str): a name of an instance of the class

    Raises:
      ValueError: if name is not valid according to the specification
    """

    if not self.isValidName(name):
      raise ValueError('Scalar name invalid (must match regexp {}): {}'.format(self.VALID_NAME,
                                                                               name))

    self._name = name
  
  def __str__(self):
    return self._name


class Tensor(AbstractType):

  # BASE_NAME is a sequence of any alphabet character
  #	  followed by any number of word characters
  BASE_NAME = r'[a-zA-Z]\w*'


  # GROUP_INDEX is a group containing either 0
  #   or a digit from 1 to 9 followed by any number of digits
  GROUP_INDEX = r'(0|[1-9]\d*)'


  # GROUP_INDICES is the open parentheses followed by
  #     a group containing GROUP_INDEX followed by
  #		      many repeats of the second group
  #			        containing comma "," followd by
  #				          GROUP_INDEX
  # followed by a closed the parentheses
  GROUP_INDICES = r'\(({0}(,{0})*)\)'.format(GROUP_INDEX)


  # VALID_NAME is BASE_NAME optionally ends with
  # a group containing GROUP_INDICES
  VALID_NAME = r'^{}({})?$'.format(BASE_NAME, GROUP_INDICES)


  def __init__(self, name, shape, spp=None, memoryLayoutClass=DenseMemoryLayout, alignStride=False):
    """TODO

    Args:
      name (str): tensor name. The name string must match a regex defined by VALID_NAME
      shape (Tuple[int, int]): shape of a tensor
      spp (TODO): sparsity pattern of a tensor
      memoryLayoutClass (Type[MemoryLayout]):
      alignStride (bool): a flag which tell yateto whether to align a matrix memory layout
                          for vectorization
    """

    # check whether the input specified by the user is correct
    if not isinstance(shape, tuple):
      raise ValueError('shape must be a tuple')
    
    if any(x < 1 for x in shape):
      raise ValueError('shape must not contain entries smaller than 1')
    
    if not self.isValidName(name):
      raise ValueError('Tensor name invalid (must match regexp {}): {}'.format(self.VALID_NAME, name))

    # init data members of an instance
    self._name = name
    self._shape = shape
    self._values = None


    if spp is not None:
      if isinstance(spp, dict):
        if not isinstance(next(iter(spp.values())), bool):
          self._values = spp
        npspp = zeros(shape, dtype=bool, order=aspp.general.NUMPY_DEFAULT_ORDER)
        for multiIndex, value in spp.items():
          npspp[multiIndex] = value
        self._spp = aspp.general(npspp)
      elif isinstance(spp, ndarray) or isinstance(spp, aspp.ASpp):
        if isinstance(spp, ndarray):
          if spp.dtype.kind == 'f':
            nonzeros = spp.nonzero()
            self._values = {entry: str(spp[entry]) for entry in zip(*nonzeros)}
        self._setSparsityPattern(spp)
      else:
        raise ValueError(name, 'Matrix values must be given as dictionary (e.g. {(1,2,3): 2.0} or as numpy.ndarray.')
    else:
      # if the user din't specify a certain sparsity pattern, then
      # create a dense sparsity pattern object with a given shape from assp module
      self._spp = aspp.dense(shape)
    self._groupSpp = self._spp

    # set memory layout for a tensor
    self.setMemoryLayout(memoryLayoutClass, alignStride)


  def setMemoryLayout(self, memoryLayoutClass, alignStride=False):
    self._memoryLayout = memoryLayoutClass.fromSpp(self._groupSpp, alignStride=alignStride)


  def _setSparsityPattern(self, spp, setOnlyGroupSpp=False):
    if spp.shape != self._shape:
      raise ValueError(name, 'The given Matrix\'s shape must match the shape specification.')
    spp = aspp.general(spp) if not isinstance(spp, aspp.ASpp) else spp
    if setOnlyGroupSpp == False:
      self._spp = spp
    self._groupSpp = spp


  def setGroupSpp(self, spp):
    self._setSparsityPattern(spp, setOnlyGroupSpp=True)
    self.setMemoryLayout(self._memoryLayout.__class__, alignStride=self._memoryLayout.alignedStride())


  def __getitem__(self, indexNames: str) -> IndexedTensor:
    """
    Creates and returns an IndexedTensor node
    initialized with the current tensor object and index names

    Args:
      indexNames: a string of tensor indices

    Returns:
      an instance of IndexedTensor
    """
    return IndexedTensor(self, indexNames)
  
  def shape(self) -> Tuple[int]:
    """
    Returns:
      shape of the tensor i.e. sizes of each dimension
    """
    return self._shape


  def memoryLayout(self):
    return self._memoryLayout


  @classmethod
  def getBaseName(cls, name: str):
    """Extracts the original name of the tensor discarding a tensor group if any.

     The original tensor name is all the characters before the tensor group
     which is inside parentheses.

     NOTE: The function is static i.e. the user can use it
     without creating an instance of the class.

     Args:
       name (str): a full name of a tensor including a tensor group if any

     Returns:
       str: a base name of a tensor

     Examples:
       >>> name = "aTensor(1,5,12)"
       >>> base_name = Tensor.getGroup(name)
       >>> print(base_name)
       'aTensor'
     """
    return re.match(cls.BASE_NAME, name).group(0)


  def baseName(self):
    """Extracts the original name of the tensor discarding a tensor group if any.

    The original tensor name is all the characters before the tensor group.
    which is inside parentheses.

    NOTE: The function is internal.

    Returns:
      str: a base name of a tensor

    Examples:
      >>> name = "aTensor(1,5,12)"
      >>> base_name = Tensor.getGroup(name)
      >>> print(base_name)
      'aTensor'
    """
    return self.getBaseName(self._name)


  @classmethod
  def getGroup(cls, name):
    """Extracts a tensor group encoded as a set of integers inside of a tensor name

    A tensor group is encoded inside of parentheses of a tensor name.
    NOTE: The function is static i.e. the user can use it
    without creating an instance of the class.

    Args:
      name (str): a tensor name

    Returns:
      Tuple[int]: a tensor group based on a tensor name

    Examples:
      >>> name = "aTensor"
      >>> group = Tensor.getGroup(name)
      >>> print(group)
      ()
      >>> name = "aTensor(1,5,12)"
      >>> group = Tensor.getGroup(name)
      >>> print(group)
      (1, 5, 12)
    """
    
    matches = re.search(cls.GROUP_INDICES, name)
    if matches:
      # return a tensor group if the name contains any
      return tuple(int(match) for match in re.split(',', matches.group(1)))

    # otherwise, return an empty tuple
    return tuple()


  def group(self):
    """Extracts a tensor group encoded as a set of integers inside of a tensor name

    NOTE: The function is internal.

    Returns:
      Tuple[int]: a tensor group based on a tensor name
    """
    return self.getGroup(self._name)


  def spp(self, groupSpp=True):
    return self._groupSpp if groupSpp else self._spp


  def values(self):
    return self._values


  def __eq__(self, other):
    equal = self._name == other._name
    if equal:
      assert self._shape == other._shape and aspp.array_equal(self._spp, other._spp) and self._memoryLayout == other._memoryLayout
    return equal


  def __hash__(self):
    return hash(self._name)


  def __str__(self):
    return '{}: {}'.format(self._name, self._shape)


class Collection(object):
  """The class represents a table of dictionaries which holds tensors according their
  names and tensor groups

  Each key of a class instance is a string. Each string corresponds to
  a sub-table which holds a table of tensors which belong to the same "class".
  Each key of a sub-table is an integer whereas its value is an instance of the Tensor class.

  NOTE: self is an instance of class Dict i.e. (<class 'dict'>)

  Examples:
    >>> names = ['A_00', 'A_01', 'B_0', 'C']
    >>> shapes = [(4,4), (3,4), (3,2), (3,2)]
    >>> tensors = {}
    >>> for name, shape in zip(names, shapes):
    ...     tensors[name] = Tensor(name, shape)
    ...
    >>> from yateto.input import __createCollection
    >>> collection = __createCollection(Tensors)

  tensor = {'A': {0: <yateto.type.Tensor object at 'A_00'>}, {1: <yateto.type.Tensor object at 'A_01'>}, \
            'B': {0: <yateto.type.Tensor object at 'B_0'>}, \
            'C': <yateto.type.Tensor object at 'C'>}
  """

  def update(self, collection):
    """Concatenates two tables together

    Args:
      collection (Collection): another table of type Collection

    Returns:
      Collection: augmented collection(table)
    """
    self.__dict__.update(collection.__dict__)


  def __getitem__(self, key):
    """Returns a reference to a tensor stored in the collection

    Args:
      key (str): a full tensor name

    Returns:
      Tensor: a reference to a tensor which corresponds to a given name
    """
    return self.__dict__[key]


  def __setitem__(self, key, value):
    """Inserts a new element to a collection

    Args:
      key (str): a base name of a tensor
      value (Dict[int, Tensor]): another tensor table
    """
    self.__dict__[key] = value


  def __contains__(self, key):
    """Check whether collection(table) contains an entry

    Args:
      key (str): a base name of a tensor

    Returns:
      bool: True if a collection contains the key. Otherwise, False
    """
    return key in self.__dict__


  @classmethod
  def group(cls, name):
    """Returns a group of a tensor embedded inside of a full tensor name

    The algorithm is based on a regular expression defined in the Tensor class
    Suggested method name: extract_tensor_group

    Args:
      name (str): a name of a tensor

    Returns:
      [Tuple[int], int]: a tensor group based on a tensor name

    Examples:
      >>> name = "aTensor(1,2,3)"
      >>> Collection.group(name)
      (1, 2, 3)

      >>> name = "aTensor(3)"
      >>> Collection.group(name)
      3


      >>> name = "aTensor"
      >>> Collection.group(name)
      ()
    """
    group = Tensor.getGroup(name)
    return group if len(group) != 1 else group[0]


  def byName(self, name):
    """Returns a tensor instance if it exists inside ot the table.

    Refactoring: get_tensor_by_name.

    Args:
      name (str): a tensor name

    Returns:
      Tensor: a tensor

    Examples:
      >>> names = ['A_00', 'A_01', 'B_0', 'C']
      >>> shapes = [(4,4), (3,4), (3,2), (3,2)]
      >>> tensors = {}
      >>> for name, shape in zip(names, shapes):
      ...     tensors[name] = Tensor(name, shape)
      ...
      >>> from yateto.input import __createCollection
      >>> collection = __createCollection(Tensors)
      >>> collection.byName('A_00')
      <yateto.type.Tensor object at 0x7fcab0ed8588>
      >>> collection.byName('C')
      <yateto.type.Tensor object at 0x7fcab0af5470>

    """
    base_name = Tensor.getBaseName(name)
    group = self.group(name)
    return self[base_name][group] if group is not tuple() else self[base_name]


  def containsName(self, name):
    """Checks whether there is a tensor with the given name by the user

    Args:
      name (str): a tensor name

    Returns:
      bool: True if there is a tensor with such a name. Otherwise, False

    Raises:
      ValueError: if a tensor name is invalid according to the Tensor class specification

    Examples:
      >>> names = ['A_00', 'A_01', 'B_0', 'C']
      >>> shapes = [(4,4), (3,4), (3,2), (3,2)]
      >>> tensors = {}
      >>> for name, shape in zip(names, shapes):
      ...     tensors[name] = Tensor(name, shape)
      ...
      >>> from yateto.input import __createCollection
      >>> collection = __createCollection(Tensors)
      >>> collection.containsName('C')
      True
      >>> collection.containsName('A_00')
      True
      >>> collection.containsName('X')
      False
    """

    if not Tensor.isValidName(name):
      raise ValueError('Invalid name: {}'.format(name))

    base_name = Tensor.getBaseName(name)
    group = self.group(name)
    return base_name in self and (group is tuple() or group in self[base_name])
