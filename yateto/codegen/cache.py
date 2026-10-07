import io
import os
import tempfile

from .code import Cpp

class RoutineGenerator(object):
  def __call__(self, routineName, fileName):
    pass

  def target(self):
    return 'cpu'

  def identity(self):
    """What tells the routine apart from another one of its name, as plain
    data that is the same in every process; None where its code does."""
    return None

class GpuRoutineGenerator(object):
  def __call__(self, routineName, fileName):
    pass

  def target(self):
    return 'gpu'

  def identity(self):
    """See `RoutineGenerator.identity`."""
    return None

class GeneratedRoutine(object):
  """A routine whose code is written already, held as plain data.

  `of` makes one of what a generator writes for a routine. As plain data
  (`asDict` and `fromDict`), the routines one process generates can be handed
  to the process that writes the routines of several -- see
  `GlobalRoutineCache.export`.
  """

  def __init__(self, target, kind, header, code, declaration, identity=None):
    self._target = target
    self._kind = kind
    self._header = header
    self._code = code
    self._declaration = declaration
    self._identity = identity

  @classmethod
  def of(cls, name, generator):
    """The routine `name` as `generator` writes it."""
    if isinstance(generator, cls):
      return generator
    with tempfile.TemporaryDirectory() as directory:
      fileName = os.path.join(directory, 'routine.cpp')
      declaration = generator(name, fileName)
      code = ''
      if os.path.exists(fileName):
        with open(fileName, encoding='utf-8') as file:
          code = file.read()
    with Cpp(io.StringIO()) as cpp:
      generator.header(cpp)
      header = cpp.out.getvalue()
    identity = generator.identity() if hasattr(generator, 'identity') else None
    return cls(generator.target(), RoutineCache.kind(generator), header, code, declaration, identity)

  def asDict(self):
    return {'target': self._target, 'kind': self._kind, 'header': self._header,
            'code': self._code, 'declaration': self._declaration, 'identity': self._identity}

  @classmethod
  def fromDict(cls, data):
    return cls(data['target'], data['kind'], data['header'], data['code'], data['declaration'],
               data.get('identity'))

  def target(self):
    return self._target

  def kind(self):
    """The class of the generator that wrote it."""
    return self._kind

  def header(self, cpp):
    cpp.out.write(self._header)

  def __call__(self, routineName, fileName):
    with open(fileName, 'a', encoding='utf-8') as file:
      file.write(self._code)
    return self._declaration

  def identity(self):
    return self._identity

  def __eq__(self, other):
    """The same routine: by the identity its generator gives it, where both
    have one, else by its code. A generator need not write a routine the same
    way in every process -- PSpaMM assigns the registers of its routines for
    ARM differently from one process to the next."""
    if not isinstance(other, GeneratedRoutine) or \
       (self._target, self._kind, self._declaration) != (other._target, other._kind, other._declaration):
      return False
    if self._identity is not None and other._identity is not None:
      return self._identity == other._identity
    return self._code == other._code

class RoutineCache(object):
  def __init__(self):
    self._routines = dict()
    self._generators = dict()

  @staticmethod
  def kind(generator):
    """What the includes of a routine depend on: the class of the generator
    that writes it. The routines of one kind share the includes of the first."""
    return generator.kind() if isinstance(generator, GeneratedRoutine) else type(generator).__name__

  def addRoutine(self, name, generator):
    if name in self._routines and not self._routines[name] == generator:
      raise RuntimeError(f'`{name}` is already in RoutineCache but the generator is not equal. '
                         f'(That is, a name was given twice for different routines.)')
    self._routines[name] = generator

    generatorName = self.kind(generator)
    if generatorName not in self._generators:
      self._generators[generatorName] = generator

  def routines(self):
    """Name and generator of every routine, in the order they were added."""
    return list(self._routines.items())

  def generate(self, header, cppFileName, gpuFileName):
    with Cpp(gpuFileName) as gpucpp:
      with Cpp(cppFileName) as cpp:
        for generator in self._generators.values():
          if generator.target() == 'gpu':
            generator.header(gpucpp)
          elif generator.target() == 'cpu':
            generator.header(cpp)
          else:
            raise NotImplementedError(f'Unknown target: {generator.target()}')

    for name, generator in self._routines.items():
      if generator.target() == 'gpu':
        declaration = generator(name, gpuFileName)
      elif generator.target() == 'cpu':
        declaration = generator(name, cppFileName)
      else:
        raise NotImplementedError(f'Unknown target: {generator.target()}')
      header(declaration)

class TinytcWriter(GpuRoutineGenerator):
  def __init__(self, signature, source):
    self._source = source
    self._signature = signature

  def __eq__(self, other):
    return self._signature == other._signature

  def header(self, cpp):
    cpp.include('tinytc/tinytc.hpp')
    cpp.include('tinytc/tinytc_sycl.hpp')
    cpp.includeSys('sycl/sycl.hpp')
    cpp.includeSys('stdexcept')
    cpp.includeSys('utility')

  def __call__(self, routineName, fileName):
    with open(fileName, 'a') as f:
      f.write(self._source)

    return self._signature
