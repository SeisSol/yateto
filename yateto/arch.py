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
  def __init__(self, name, precision, alignment, enablePrefetch=False, host_name=None):
    self.name = name
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

    self.host_name = host_name

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

def getArchitectureIdentifiedBy(host_ident, compute_ident=None):
  if not compute_ident:
    compute_ident = host_ident

  if host_ident[0].upper() != compute_ident[0].upper():
    raise RuntimeError(f'Precision of host and compute arch. must be the same. '
                       f'Given: {host_ident}, {compute_ident}')

  precision = compute_ident[0].upper()
  compute_name = compute_ident[1:]
  host_name = host_ident[1:]
  arch = {
    'noarch': Architecture(compute_name, precision, 16, False),
    'wsm':    Architecture(compute_name, precision, 16, False),
    'snb':    Architecture(compute_name, precision, 32, False),
    'hsw':    Architecture(compute_name, precision, 32, False),
    'skx':    Architecture(compute_name, precision, 64, True),
    'knc':    Architecture(compute_name, precision, 64, False),
    'knl':    Architecture(compute_name, precision, 64, True), # Libxsmm currently supports prefetch only for KNL kernels
    'nvidia': Architecture(compute_name, precision, 32, False, host_name)
  }
  return arch[compute_name]

def useArchitectureIdentifiedBy(host_ident, compute_ident):
  arch = getArchitectureIdentifiedBy(host_ident, compute_ident)
  DenseMemoryLayout.setAlignmentArch(arch)
  return arch
