from .code import Cpp

class RoutineGenerator(object):
  def __call__(self, routineName, fileName):
    pass

class GpuRoutineGenerator(object):
  def __call__(self, routineName, fileName):
    pass

class RoutineCache(object):
  def __init__(self):
    self._routines = dict()
    self._generators = dict()
  
  def addRoutine(self, name, generator):
    if name in self._routines and not self._routines[name] == generator:
      raise RuntimeError(f'`{name}` is already in RoutineCache but the generator is not equal. '
                         f'(That is, a name was given twice for different routines.)')
    self._routines[name] = generator
    
    generatorName = type(generator).__name__
    if generatorName not in self._generators:
      self._generators[generatorName] = generator
  
  def generate(self, header, cppFileName, gpuFileName):
    with Cpp(gpuFileName) as gpucpp:
      with Cpp(cppFileName) as cpp:
        for generator in self._generators.values():
          if isinstance(generator, GpuRoutineGenerator):
            generator.header(gpucpp)
          else:
            generator.header(cpp)

    for name, generator in self._routines.items():
      if isinstance(generator, GpuRoutineGenerator):
        declaration = generator(name, gpuFileName)
      else:
        declaration = generator(name, cppFileName)
      header(declaration)
