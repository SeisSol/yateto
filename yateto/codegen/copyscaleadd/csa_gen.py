from ..common import *
from ..cache import RoutineGenerator
from gemmforge import DenseMatrix, CsaGenerator, GenerationError
from gemmforge import arch as GemmForgeArch
import re

from yateto.type import Tensor


class CsaGen(object):
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
    """
    d = self._descr  # type: copyscaleadd.Description

    # TODO:  implement initialization in the generator
    #if d.beta == 0.0:
    #  writeBB = boundingBoxFromLoopRanges(d.result.indices, d.loopRanges)
    #  initializeWithZero(cpp, self._arch, d.result, writeBB)

    # calculate FLOP per single operation
    flop = 0
    alpha = d.alpha

    if alpha not in [-1.0, 1.0]:
      flop += 1

    if d.beta == 1.0 and alpha == -1.0:
      alpha = 1.0
      flop += 1

    elif d.beta == 1.0:
      flop += 1

    elif d.beta != 0.0:
      raise NotImplementedError

    # convert data for gemmforge
    matrix_a = DenseMatrix(num_rows=d.term.memoryLayout._bbox[0].stop,
                           num_cols=d.term.memoryLayout._bbox[1].stop,
                           addressing=deduce_addresing(d.term),
                           bbox=deduce_bbox(BoundingBox.fromSpp(d.term.eqspp)),
                           transpose=False)

    matrix_b = DenseMatrix(num_rows=d.result.memoryLayout._bbox[0].stop,
                           num_cols=d.result.memoryLayout._bbox[1].stop,
                           addressing=deduce_addresing(d.result),
                           bbox=deduce_bbox(BoundingBox.fromSpp(d.result.eqspp)),
                           transpose=False)
    try:
      forge_generator = CsaGenerator(GemmForgeArch.produce("nvidia"), self._arch.typename)
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


    return flop * d.term.eqspp.size
    """

    description = self._descr  # type: copyscaleadd.Description

    if description.beta == 0.0:
      # TODO: figure out how to do this trick on gpu
      # writeBB = boundingBoxFromLoopRanges(description.result.indices, description.loopRanges)
      # initializeWithZero(cpp, self._arch, description.result, writeBB)
      pass

    # extract first two leading tensor indices
    # summation of which is going to be computed on GPU
    leading_indices = (description.result.indices[0],
                       description.result.indices[1])  # type: Set[str]

    self.cuda_kernel_indices = \
      description.result.indices.extract(indexNames="".join(leading_indices))  # type: Set[Indices]

    for_loop_index_names = description.result.indices - self.cuda_kernel_indices  # type: Set[str]
    for_loop_indices = \
      description.result.indices.extract(indexNames="".join(for_loop_index_names))

    class CopyScaleAddBody(object):
      def __call__(s):

        flop = 0

        parameters = "{}, {}, ".format(description.loopRanges[self.cuda_kernel_indices[0]].stop,
                                       description.loopRanges[self.cuda_kernel_indices[1]].stop)

        result_data_shift = 0
        term_data_shift = 0
        for i in range(len(description.result.indices)):
          result_data_shift += description.result.memoryLayout.stridei(i) \
                               * description.loopRanges[description.result.indices[i]].start

          term_data_shift += description.term.memoryLayout.stridei(i) \
                             * description.loopRanges[description.term.indices[i]].start

        # append indices w.r.t outer-most (for-loop) loop indices
        if len(description.result.indices) > len(self.cuda_kernel_indices):
          result_data_shift = "{} + {}".format(str(result_data_shift),
                                               description.result.memoryLayout.addressString(
                                                 description.result.indices, for_loop_indices))

          term_data_shift = "{} + {}".format(str(term_data_shift),
                                             description.term.memoryLayout.addressString(
                                               description.term.indices, for_loop_indices))

        parameters += "{}, {} + {}, {}, ".format(description.alpha,
                                                 description.term.name,
                                                 term_data_shift,
                                                 description.term.memoryLayout.stridei(1))

        parameters += "{}, {} + {}, {}, ".format(description.beta,
                                                 description.result.name,
                                                 result_data_shift,
                                                 description.result.memoryLayout.stridei(1))

        if Tensor.getGroup(description.term.name):
          term_group = "({})".format(*Tensor.getGroup(description.term.name))
        else:
          term_group = ""

        term_base_name = Tensor.getBaseName(description.term.name)
        parameters += "ExtraOffset_{}{}, ".format(term_base_name, term_group)

        if Tensor.getGroup(description.result.name):
          result_group = "({})".format(*Tensor.getGroup(description.result.name))
        else:
          result_group = ""

        result_base_name = Tensor.getBaseName(description.result.name)
        parameters += "ExtraOffset_{}{}, ".format(result_base_name, result_group)

        parameters += "NumElements"

        cpp('device.copyAddScale({});'.format(parameters))

        return flop

    return forLoops(cpp=cpp,
                    indexNames=for_loop_indices,
                    ranges=description.loopRanges,
                    body=CopyScaleAddBody(),
                    pragmaSimd=False)


def deduce_addresing(term):
  if term.is_const:
    return 'none'
  temp_variable_name = re.compile(r'_tmp*')
  if temp_variable_name.match(term.name):
    return 'strided'
  else:
    return 'pointer_based'


def deduce_bbox(yateto_bbox):
  gemmforge_bbox = [yateto_bbox[0].start,
                    yateto_bbox[1].start,
                    yateto_bbox[0].stop - 1,
                    yateto_bbox[1].stop - 1]
  return gemmforge_bbox


def deduce_arg(term):
  temp_variable_name = re.compile(r'_tmp*')
  if term.is_const or temp_variable_name.match(term.name):
    extra_offset = '0'
  else:
    extra_offset = f'ExtraOffset_{term.name}'
  return f'{term.name}, {extra_offset}'


class GemmForgeWriter(RoutineGenerator):
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