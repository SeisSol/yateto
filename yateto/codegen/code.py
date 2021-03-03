##
# @file
# This file is part of SeisSol.
#
# @author Carsten Uphoff (c.uphoff AT tum.de, http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
#
# @section LICENSE
# Copyright (c) 2015, SeisSol Group
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

import sys

class NoScope:
  def __enter__(self):
    pass

  def __exit__(self, type, value, traceback):
    pass
  
class Block:
  def __init__(self, writer, argument, foot = ''):
    self.writer = writer
    self.argument = argument
    self.foot = foot
    
  def __enter__(self):
    space = ' ' if self.argument else ''
    self.writer(self.argument + space + '{')
    self.writer.indent += 1

  def __exit__(self, type, value, traceback):
    self.writer.indent -= 1
    self.writer('}' + self.foot)

class MultiBlock:
  def __init__(self, writer, arguments, foot=None):
    self.writer = writer
    self.arguments = arguments
    if foot is None:
      self.foot = [''] * len(self.arguments)
    else:
      self.foot = foot
  
  def __enter__(self):
    for arg in self.arguments:
      self.writer(arg + ' {')
      self.writer.indent += 1
  
  def __exit__(self, type, value, traceback):
    # Blocks are closed in reverse order, thus reverse footer
    for arg, foot in zip(self.arguments, reversed(self.foot)):
      self.writer.indent -= 1
      self.writer('}' + foot)
    
class HeaderGuard:
  def __init__(self, writer, name):
    self.writer = writer
    self.name = name
    
  def __enter__(self):
    self.writer('#ifndef ' + self.name)
    self.writer('#define ' + self.name)

  def __exit__(self, type, value, traceback):
    self.writer('#endif')
    
class PPIfBlock:
  def __init__(self, writer, name, typ):
    self.writer = writer
    self.name = name
    self.typ = typ
    
  def __enter__(self):
    self.writer('#{} {}'.format(self.typ, self.name))

  def __exit__(self, type, value, traceback):
    self.writer('#endif')

class Cpp:
  def __init__(self, streamOrFileName = sys.stdout):
    self.fileHandle = streamOrFileName
    self.indent = 0
    
  def __enter__(self):
    self.out = open(self.fileHandle, 'w+') if isinstance(self.fileHandle, str) else self.fileHandle
    return self
    
  def __exit__(self, type, value, traceback):
    if self.out is not sys.stdout:
      self.out.close()
    self.out = None
    
  def __call__(self, code):
    indentSpace = self.indent * '  '
    for line in code.splitlines():
      self.out.write(indentSpace + line + '\n')
  
  def emptyline(self):
    self.out.write('\n')
      
  def If(self, argument):
    return Block(self, 'if ({})'.format(argument))
      
  def For(self, argument):
    return Block(self, 'for ({})'.format(argument))
    
  def Namespace(self, name):
    if len(name) == 0:
      return NoScope()
    spaces = name.split('::')
    if len(spaces) == 1:
      foot = ' // namespace {}'.format(name)
      return Block(self, 'namespace ' + name, foot=foot)
    else:
      foot = [' // namespace {}'.format(s) for s in spaces]
      return MultiBlock(self, ['namespace ' + space for space in spaces], foot=foot)

  def AnonymousScope(self):
    return Block(self, '')
    
  def Function(self, name, arguments = '', returnType = 'void', const = False):
    if returnType:
      returnType += ' '
    return Block(self, '{}{}({}){}'.format(returnType, name, arguments, ' const' if const else ''))
    
  def functionDeclaration(self, name, arguments = '', returnType = 'void'):
    return self.__call__('{} {}({});'.format(returnType, name, arguments))

  def Class(self, name):
    return Block(self, 'class ' + name, foot=';')
  
  def classDeclaration(self, name):
    return self.__call__('class {};'.format(name))
  
  def forwardStruct(self, name):
    self.__call__('struct {};'.format(name))

  def Struct(self, name):
    return Block(self, 'struct ' + name, foot=';')
    
  def HeaderGuard(self, name):
    return HeaderGuard(self, name)
    
  def PPIfndef(self, name):
    return PPIfBlock(self, name, 'ifndef')
    
  def PPIf(self, name):
    return PPIfBlock(self, name, 'if')
    
  def label(self, name):
    self.indent -= 1
    self.__call__(name + ':')
    self.indent += 1
    
  def includeSys(self, header):
    self.__call__('#include <{}>'.format(header))

  def include(self, header):
    self.__call__('#include "{}"'.format(header))
    
  def memset(self, name, numberOfValues, typename, offset=0):
    pointer = '&{}[{}]'.format(name, offset) if offset != 0 else name
    self.__call__('memset({}, 0, {} * sizeof({}));'.format(pointer, numberOfValues, typename))
