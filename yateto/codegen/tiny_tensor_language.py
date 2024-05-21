# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Union, List
from ..ast.visitor import Visitor

# Data types

DYNAMIC = -9223372036854775808


def format_mode(mode):
    return '?' if mode == DYNAMIC else str(mode)


class DataType(ABC):
    pass


class IntegerType(Enum):
    i1 = 0,
    i8 = 1,
    i16 = 2,
    i32 = 3,
    i64 = 4,
    index = 5


class FloatingType(Enum):
    f32 = 0,
    f64 = 1


class ScalarType(DataType):

    def __init__(self, ty: Enum):
        self.ty = ty


class MemrefType(DataType):

    def __init__(self, ty: ScalarType, shape: tuple[int], stride: tuple[int]):
        self.ty = ty
        self.shape = shape
        self.stride = stride

    def order(self):
        return len(self.shape)


class GroupType(DataType):

    def __init__(self, ty: MemrefType, offset: Optional[int] = 0):
        self.ty = ty
        self.offset = offset


# Value


class Value(ABC):

    @abstractmethod
    def type(self):
        pass


class IntImmValue(Value):

    def __init__(self, ty: IntegerType, value: int):
        self.ty = ty
        self.value = value

    def type(self):
        return self.ty


class FloatImmValue(Value):

    def __init__(self, ty: FloatingType, value: float):
        self.ty = ty
        self.value = value

    def type(self):
        return self.ty


class LocalValue(Value):

    def __init__(self, ty: DataType, name: str = ""):
        self.ty = ty
        self.name = name

    def type(self):
        return self.ty


# Inst


class Inst(ABC):

    @abstractmethod
    def value(self):
        pass


# Region


class Region(object):

    def __init__(self, body: list[Inst]):
        self.body = body


# Instructions


class Transpose(Enum):
    n = 0,
    t = 1

class Arithmetic(Enum):
    add = 0,
    sub = 1,
    mul = 2,
    div = 3,
    rem = 4


class AxpbyInst(Inst):

    def __init__(self,
                 trans: Transpose,
                 alpha: Value,
                 a: Value,
                 beta: Value,
                 b: Value,
                 atomic: bool = False):
        self.trans = trans
        self.alpha = alpha
        self.a = a
        self.beta = beta
        self.b = b
        self.atomic = atomic

    def value(self):
        return None

class ArithInst(Inst):
    def __init__(self, operation_type: Arithmetic, a: Value, b: Value):
        self.operation_type = operation_type
        self.a = a
        self.b = b
        self.result = LocalValue(a.type())

    def value(self):
        return self.result


#class AllocaInst(ValueInst):
#
#    def __init__(self, ty: MemrefType):
#        self.value = Value(ty)
#
#    def value(self):
#        return self.value
#
class GroupIdInst(Inst):

    def __init__(self):
        self.result = LocalValue(ScalarType(IntegerType.index))

    def value(self):
        return self.result


class LoadInst(Inst):

    def __init__(self, operand: Value, index_list: list[Value]):
        self.operand = operand
        self.index_list = index_list
        self.result = None
        self.result = LocalValue(operand.type().ty)

    def value(self):
        return self.result

class ForInst(Inst):

    def __init__(self, loop_var: Value, start: Value, stop: Value,
                 body: Region):
        self.loop_var = loop_var
        self.start = start
        self.stop = stop
        self.body = body

    def value(self):
        return None

class StoreInst(Inst):

    def __init__(self, data: Value, operand: Value, index_list: list[Value]):
        self.data = data
        self.operand = operand
        self.index_list = index_list
        self.result = LocalValue(operand.type().ty)

    def value(self):
        return self.result


class SubviewInst(Inst):

    def __init__(self, operand: Value, offset_list: list[Value], size_list: list[Value]):
        if not isinstance(operand.type(), MemrefType):
            raise RuntimeError('Subview instruction expects memref type')
        if len(offset_list) != len(size_list) or len(offset_list) != len(operand.type().shape):
            raise RuntimeError('Subview slice list length must match tensor order')

        sliced_modes = [i for i,v in enumerate(size_list) if v is not None]
        shape = []
        stride = []
        for mode in sliced_modes:
            size = DYNAMIC
            if isinstance(size_list[mode], IntImmValue):
                size = size_list[mode].value
                if size == DYNAMIC and isinstance(offset_list[mode], IntImmValue) and operand.type().shape[mode] != DYNAMIC:
                    size = operand.type().shape[mode] - offset_list[mode].value
            shape.append(size)
            stride.append(operand.type().stride[mode])

        self.operand = operand
        self.offset_list = offset_list
        self.size_list = size_list
        self.result = LocalValue(MemrefType(operand.type().ty, shape, stride))

    def value(self):
        return self.result




# Function


class Function(object):

    def __init__(self, name: str, args: list[LocalValue], body: Region):
        self.name = name
        self.args = args
        self.body = body


class RegionBuilder(object):

    def __init__(self):
        self.body = []

    def add(self, inst: Inst):
        self.body.append(inst)
        return inst.value()

    def extend(self, otherRegion: Region):
        for inst in otherRegion.body:
            self.body.append(inst)

    def get_product(self):
        return Region(self.body.copy())


class Traversal(Visitor):

    def generic_visit(self, node):
        raise NotImplementedError(
            f'Traversal for {type(node).__name__} not implemented')

    def visit_IntImmValue(self, node):
        pass

    def visit_FloatImmValue(self, node):
        pass

    def visit_AxpbyInst(self, node):
        self.visit(node.alpha)
        self.visit(node.a)
        self.visit(node.beta)
        self.visit(node.b)

    def visit_ArithInst(self, node):
        self.visit(node.result)
        self.visit(node.a)
        self.visit(node.b)

    def visit_LocalValue(self, node):
        self.visit(node.type())

    def visit_GroupIdInst(self, node):
        self.visit(node.result)

    def visit_LoadInst(self, node):
        self.visit(node.result)
        self.visit(node.operand)
        for idx in node.index_list:
            self.visit(idx)

    def visit_ForInst(self, node):
        self.visit(node.loop_var)
        self.visit(node.start)
        self.visit(node.stop)
        self.visit(node.body)

    def visit_Region(self, node):
        for inst in node.body:
            self.visit(inst)

    def visit_Function(self, node):
        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)

    def visit_StoreInst(self, node):
        self.visit(node.data)
        self.visit(node.operand)
        for idx in node.index_list:
            self.visit(idx)

    def visit_SubviewInst(self, node):
        self.visit(node.result)
        self.visit(node.operand)
        for v in node.offset_list:
            self.visit(v)
        for v in node.size_list:
            if v is not None:
                self.visit(v)

class AssignIdentifiers(Traversal):

    def __init__(self):
        self.val_counter = 0

    def visit_LocalValue(self, node):
        if not node.name:
            node.name = str(self.val_counter)
            self.val_counter += 1


class Dump(Visitor):

    BASE_INDENT = '  '

    def __init__(self):
        self.level = 0

    def generic_visit(self, node):
        raise NotImplementedError(
            f'Dump for {type(node).__name__} not implemented')

    def visit_ScalarType(self, node):
        return f'{node.ty.name}'

    def visit_MemrefType(self, node):
        shape_str = ''
        for s in node.shape:
            shape_str += f'x{format_mode(s)}'
        stride_str = ','.join(format_mode(s) for s in node.stride)
        return f'memref<{self.visit(node.ty)}{shape_str}, strided<{stride_str}>>'

    def visit_GroupType(self, node):
        return f'group<{self.visit(node.ty)}, offset: {format_mode(node.offset)}>'

    def visit_IntImmValue(self, node):
        return f'{format_mode(node.value)}'

    def visit_FloatImmValue(self, node):
        return f'{node.value}'

    def visit_LocalValue(self, node):
        return f'%{node.name}'

    def visit_AxpbyInst(self, node):
        opcode = f'axpby.{node.trans.name}'
        if node.atomic:
            opcode += '.atomic'
        args = (node.alpha, node.a, node.beta, node.b)
        args_str = ', '.join(self.visit(arg) for arg in args)
        type_str = ', '.join(self.visit(arg.type()) for arg in args)
        return f'{opcode} {args_str} : {type_str}'

    def visit_ArithInst(self, node):
        return f'{self.visit(node.value())} = arith.{node.operation_type.name} {self.visit(node.a)}, {self.visit(node.b)} : {self.visit(node.a.type())}'

    def visit_GroupIdInst(self, node):
        return f'{self.visit(node.value())} = group_id'

    def visit_LoadInst(self, node):
        indices = ','.join(self.visit(index) for index in node.index_list)
        return f'{self.visit(node.value())} = load {self.visit(node.operand)}[{indices}] : {self.visit(node.operand.type())}'

    def visit_ForInst(self, node):
        loop_range = f'{self.visit(node.loop_var)}={self.visit(node.start)},{self.visit(node.stop)}'
        return f'for {loop_range} : {self.visit(node.loop_var.type())} {self.visit(node.body)}'

    def visit_StoreInst(self, node):
        indices = ','.join(self.visit(index) for index in node.index_list)
        return f'store {self.visit(node.data)}, {self.visit(node.operand)}[{indices}] : {self.visit(node.operand.type())}'

    def visit_SubviewInst(self, node):
        slice_list = []
        for offset, size in zip(node.offset_list, node.size_list):
            if size is not None:
                slice_list.append(f'{self.visit(offset)}:{self.visit(size)}')
            else:
                slice_list.append(f'{self.visit(offset)}')
        return f'{self.visit(node.value())} = subview {self.visit(node.operand)}[{",".join(slice_list)}] : {self.visit(node.operand.type())}'

    def visit_Region(self, node):
        self.level += 1
        indent = self.BASE_INDENT * self.level
        body_str = f'\n{indent}'.join(self.visit(i) for i in node.body)
        self.level -= 1
        return f'{{\n{indent}{body_str}\n{self.BASE_INDENT * self.level}}}'

    def visit_Function(self, node):
        arg_indent = ' ' * (7 + len(node.name))
        args_str = f',\n{arg_indent}'.join(
            f'{self.visit(arg)}: {self.visit(arg.type())}'
            for arg in node.args)
        return f'func @{node.name}({args_str}) {self.visit(node.body)}'
