##
# @file
# This file is part of SeisSol.
#
# @author Carsten Uphoff (c.uphoff AT tum.de, http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
#
# @section LICENSE
# Copyright (c) 2015-2018, SeisSol Group
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# @section DESCRIPTION
#

from .memory import DenseMemoryLayout

class Architecture(object):
  def __init__(self,
               name,
               precision,
               alignment,
               enablePrefetch=False,
               sub_name=None,
               host_name=None):
  #def __init__(self, name, sub_name, precision, alignment, enablePrefetch=False, host_name=None):
    """

    Args:
      name (str): name of the compute (main) architecture.
      sub_name (str): name of sub. architecture type e.g., a model of Nvidia streaming
          multiprocessor (sm_60, sm_61, etc). In case of CPU, the field is equal to None
      precision (str): either 'd' or 's' character which stands for 'double' or 'single' precision
      alignment (int): length of a vector register (unit) in bytes
      enablePrefetch (bool): indicates whether the compute (main) architecture supports
          data prefetching
      host_name (str): name of the host (CPU) architecture. If the code is intentend to be generated
          to CPU-like architecture then the field should be equal to None
    """
    self.name = name
    self.sub_name = sub_name
    self.host_name = host_name

    self.precision = precision.upper()
    if self.precision == 'D':
      self.bytesPerReal = 8
      self.typename = 'double'
      self.epsilon = 2.22e-16
    elif self.precision == 'S':
      self.bytesPerReal = 4
      self.typename = 'float'
      self.epsilon = 1.19e-7
    else:
      raise ValueError('Unknown precision type ' + self.precision)
    self.alignment = alignment
    assert self.alignment % self.bytesPerReal == 0
    self.alignedReals = self.alignment // self.bytesPerReal
    self.enablePrefetch = enablePrefetch
    
    self.uintTypename = 'unsigned'
    self.ulongTypename = 'unsigned long'

    self._tmpStackLimit = 524288

  def setTmpStackLimit(self, tmpStackLimit):
    self._tmpStackLimit = tmpStackLimit

  def alignedLower(self, index):
    return index - index % self.alignedReals

  def alignedUpper(self, index):
    return index + (self.alignedReals - index % self.alignedReals) % self.alignedReals
  
  def alignedShape(self, shape):
    return (self.alignedUpper(shape[0]),) + shape[1:]

  def checkAlignment(self, offset):
    return offset % self.alignedReals == 0
  
  def formatConstant(self, constant):
    return str(constant) + ('f' if self.precision == 'S' else '')

  def onHeap(self, numReals):
    return (numReals * self.bytesPerReal) > self._tmpStackLimit


def _get_name_and_precision(ident):
  return ident[1:], ident[0].upper()


def getArchitectureIdentifiedBy(ident):
  name, precision = _get_name_and_precision(ident)
  arch = {
    'noarch': Architecture(name, precision, 16, False),
    'wsm': Architecture(name, precision, 16, False),
    'snb': Architecture(name, precision, 32, False),
    'hsw': Architecture(name, precision, 32, False),
    'skx': Architecture(name, precision, 64, True),
    'knc': Architecture(name, precision, 64, False),
    'knl': Architecture(name, precision, 64, True), # Libxsmm currently supports prefetch only for KNL kernels
    'rome': Architecture(name, precision, 32, False),
    'thunderx2t99': Architecture(name, precision, 16, False),
    'power9': Architecture(name, precision, 16, False)
  }
  return arch[name]


def getHeterogeneousArchitectureIdentifiedBy(compute_ident, compute_sub_arch=None, host_ident=None):
  compute_name, compute_precision = _get_name_and_precision(compute_ident)
  host_name, host_precision = _get_name_and_precision(host_ident)

  if (compute_precision != host_precision):
    raise ValueError(f'Precision of host and compute arch. must be the same. '
                     f'Given: {host_ident}, {compute_ident}')

  arch = {
    'nvidia': Architecture(compute_name, compute_precision, 64, False, compute_sub_arch, host_name)
  }
  return arch[compute_name]


def useArchitectureIdentifiedBy(compute_ident, compute_sub_arch=None, host_ident=None):
  if not (compute_sub_arch or host_ident):
    arch = getArchitectureIdentifiedBy(compute_ident)

  elif (compute_sub_arch and host_ident):
    arch = getHeterogeneousArchitectureIdentifiedBy(compute_ident, compute_sub_arch, host_ident)
  else:
    raise ValueError(f'given an incomplete set of input parameters: '
                     f'{compute_ident}, {compute_sub_arch}, {host_ident}')

  DenseMemoryLayout.setAlignmentArch(arch)
  return arch
