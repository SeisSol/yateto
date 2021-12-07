from ..common import *
from ..cache import RoutineGenerator, GpuRoutineGenerator
from ..common import BatchedOperationsAux


# Optional modules
import importlib.util
gf_spec = importlib.util.find_spec('gemmforge')
try:
  if gf_spec:
    gf = gf_spec.loader.load_module()
except:
  raise ('Cannot load gemmforge.')


class CopyScaleAddGenerator(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _formatTerm(self, alpha, term):
    """Generate a sub-string of a term for a source code which is going to be used
    inside of the inner most for-loop

    Args:
      alpha (Union[Scalar, float]): TODO
      term (IndexedTensorDescription): TODO

    Returns:

    Examples:
      >>> from yateto.memory import DenseMemoryLayout
      >>> from yateto.ast.indices import Indices
      >>> from yateto.codegen.common import IndexedTensorDescription
      >>> from yateto.aspp import dense
      >>> from yateto.codegen.copyscaleadd.generic import Generic
      >>> tensor_shape = (5, 6)
      >>> layout = DenseMemoryLayout(shape=tensor_shape)
      >>> indices = Indices(indexNames='ij', shape=tensor_shape)
      >>> description = IndexedTensorDescription(name='A', \
                                                 indices=indices, \
                                                 memoryLayout=layout, \
                                                 eqspp=dense(shape=tensor_shape))
      >>> obj = Generic(arch='dummy', descr=description)
      >>> obj._formatTerm(alpha=3, term=description)
      '3 * A[1*i + 5*j]'
    """

    prefix = ''
    if alpha == 0.0:
      return ''

    if alpha == 1.0:
      prefix = term.name
    else:
      prefix = '{} * {}'.format(alpha, term.name)

    return '{}[{}]'.format(prefix, term.memoryLayout.addressString(term.indices))

  def generate(self, cpp, routineCache):
    """Generates a tensor equation of a form: B = beta * B + alpha * A
    Args:
      cpp (IO): a file stream
      routineCache:

    Returns:

    """
    if (gf_spec):
      d = self._descr  # type: copyscaleadd.Description
      m = d.loopRanges[d.result.indices[0]]
      n = d.loopRanges[d.result.indices[1]]
      alpha = d.alpha

      aux = BatchedOperationsAux(self._arch.typename)
      matrix_a = gf.YatetoInterface.produce_dense_matrix((m, n),
                                                         d.term.memoryLayout.bbox(),
                                                         addressing=aux.deduce_addresing(d.term),
                                                         transpose=False)

      matrix_b = gf.YatetoInterface.produce_dense_matrix((m, n),
                                                         d.result.memoryLayout.bbox(),
                                                         addressing=aux.deduce_addresing(d.result),
                                                         transpose=False)

      try:
        vm = gf.vm_factory(self._arch.name, self._arch.backend, fp_type=self._arch.typename)
        forge_generator = gf.CsaGenerator(vm)
        forge_generator.set(matrix_a, matrix_b, alpha, d.beta)
        routine_name = forge_generator.get_base_name()

        args = [str(alpha),
                aux.deduce_arg(d.term),
                aux.deduce_arg(d.result),
                BatchedOperationsAux.NUM_ELEMENTS_NAME,
                BatchedOperationsAux.FLAGS_NAME,
                BatchedOperationsAux.STREAM_PTR_NAME]
        cpp("{}({});".format(routine_name, ', '.join(args)))

        routineCache.addRoutine(routine_name, GemmForgeWriter(forge_generator, vm.get_headers()))

      except gf.GenerationError as err:
        print("ERROR: {}".format(err))
        raise err

      return m.size() * n.size()

    else:
      raise RuntimeError('gemmforge module is not found. You can install it with pip3. '
                         'e.g., pip3 install gemmforge')


class GemmForgeWriter(GpuRoutineGenerator):
  def __init__(self, forge_generator, headers):
    self._generator = forge_generator
    self._basename = forge_generator.get_base_name()
    self._headers = headers

  def __eq__(self, other):
    if isinstance(other, GemmForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.includes(self._headers)

  def __call__(self, routineName, fileName):
    self._generator.generate()
    declaration = self._generator.get_launcher_header()
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, "a") as file:
      file.write(kernel)
      file.write(launcher)

    return declaration
