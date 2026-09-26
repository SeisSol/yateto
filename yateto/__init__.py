from .type import *
from .generator import NamespacedGenerator, Generator, simpleParameterSpace, parameterSpaceFromRanges, GlobalRoutineCache
from .arch import useArchitectureIdentifiedBy, deriveArchitecture, HostArchDefinition, DeviceArchDefinition, fixArchitectureGlobal
from .gemm_configuration import *

import builtins as _builtins
import types as _types

def _isForeignModule(name, value):
  """Whether a name bound here is a module `from yateto import *` should skip.

  The star imports above pull in whatever their modules imported, so `math`,
  `re` and `numpy` end up in this namespace -- and so does the submodule
  `type`, which shadows the builtin of that name in the importer's scope.
  """
  if not isinstance(value, _types.ModuleType):
    return False
  return hasattr(_builtins, name) or value.__name__.partition('.')[0] != 'yateto'

__all__ = sorted(name for name, value in list(globals().items())
                 if not name.startswith('_') and not _isForeignModule(name, value))
