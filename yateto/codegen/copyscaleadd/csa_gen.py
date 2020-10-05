from ..common import *
from ..cache import RoutineGenerator, GpuRoutineGenerator
from yateto.type import Tensor
import re


# Optional modules
import importlib.util
gf_spec = importlib.util.find_spec('gemmforge')
gf_dense_spec = importlib.util.find_spec('gemmforge.DenseMatrix') if gf_spec else None
gf_csa_spec = importlib.util.find_spec('gemmforge.CsaGenerator') if gf_spec else None
gf_error_spec = importlib.util.find_spec('gemmforge.GenerationError') if gf_spec else None
gf_arch_spec = importlib.util.find_spec('gemmforge.GemmForgeArch') if gf_spec else None

try:
  if gf_spec:
    DenseMatrix = gf_csa_spec.loader.load_module()
    CsaGenerator = gf_csa_spec.loader.load_module()
    GenerationError = gf_error_spec.loader.load_module()
    GemmForgeArch = gf_arch_spec.loader.load_module()
except:
  pass

class CsaGen(object):
  """Copy-Add-Scale Generator (Csa): B = beta * B + alpha * A

  """
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

      # convert data for gemmforge
      matrix_a = DenseMatrix(num_rows=d.term.memoryLayout._bbox[0].stop,
                             num_cols=d.term.memoryLayout._bbox[1].stop,
                             addressing=deduce_addresing(d.term),
                             bbox=deduce_bbox(m, n, False, d.term.memoryLayout._bbox),
                             transpose=False)

      matrix_b = DenseMatrix(num_rows=d.result.memoryLayout._bbox[0].stop,
                             num_cols=d.result.memoryLayout._bbox[1].stop,
                             addressing=deduce_addresing(d.result),
                             bbox=deduce_bbox(m, n, False, d.result.memoryLayout._bbox),
                             transpose=False)
      try:
        forge_generator = CsaGenerator(GemmForgeArch.produce(self._arch.name, self._arch.sub_name),
                                       self._arch.typename)
        forge_generator.generate(matrix_a, matrix_b, alpha, d.beta)
        routine_name = forge_generator.get_base_name()

        args = [str(alpha),
                deduce_arg(d.term),
                deduce_arg(d.result),
                'NumElements']
        cpp("{}({});".format(routine_name, ', '.join(args)))

        routineCache.addRoutine(routine_name, GemmForgeWriter(forge_generator))

      except GenerationError as err:
        print("ERROR: {}".format(err))
        raise err

      return m.size() * n.size()

    else:
      raise RuntimeError('gemmforge module is not found. You can install it with pip3. '
                         'e.g., pip3 install gemmforge')

def deduce_addresing(term):
  if term.is_compute_constant:
    return 'none'
  temp_variable_name = re.compile(r'_tmp*')
  if temp_variable_name.match(term.name):
    return 'strided'
  else:
    return 'pointer_based'


def deduce_bbox(rows_range, cols_range, is_trans, ml_bbox):
  """Converts yateto memory layoyt (bounding boxes) and ranges to Gemmforge bounding boxes i.e.,
     a box is a list of rows and columns indices where the actual data is located within
     a memory patch

  Args:
    rows_range (loopRanges): a range of rows to operate on
    cols_range (loopRanges): a range of columns to operate on
    is_trans (bool): if true then a GemmForge bonding box needs to be transposed
    ml_bbox (BoundingBox): yateto bounding box (memory layout)

  Returns:
    (list): bounding box in GemmForge format
  """
  if is_trans:
    bbox = [cols_range.start - ml_bbox[0].start,
            rows_range.start - ml_bbox[1].start,
            cols_range.stop - ml_bbox[0].start - 1,
            rows_range.stop - ml_bbox[1].start - 1]
  else:
    bbox = [rows_range.start - ml_bbox[0].start,
            cols_range.start - ml_bbox[1].start,
            rows_range.stop - ml_bbox[0].start - 1,
            cols_range.stop - ml_bbox[1].start - 1]
  return bbox


def deduce_arg(term):
  temp_variable_name = re.compile(r'_tmp*')
  if term.is_compute_constant or temp_variable_name.match(term.name):
    extra_offset = '0'
  else:
    extra_offset = f'ExtraOffset_{term.name}'
  return f'{term.name}, {extra_offset}'


class GemmForgeWriter(GpuRoutineGenerator):
  def __init__(self, forge_generator):
    self._basename = forge_generator.get_base_name()
    self._declaration = forge_generator.get_launcher_header()
    self._launcher = forge_generator.get_launcher()
    self._kernel = forge_generator.get_kernel()

  def __eq__(self, other):
    if isinstance(other, GemmForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.include('gemmgen_aux.h')

  def __call__(self, routineName, fileName):
    with open(fileName, "a") as file:
      file.write(self._kernel)
      file.write(self._launcher)

    return self._declaration