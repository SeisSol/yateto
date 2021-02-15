from ...ast.node import LoopOverGEMM
from ..graph import FusedActions, FusedProgramPoint
from abc import ABC, abstractmethod
from typing import Union


class Context:
  def __init__(self):
    self._current_state: Union[State, None] = None
    self._fused_action = None
    self._cfg = []

  def process(self, program_point):
    self._current_state.process(program_point)

  def get_cfg(self):
    return self._cfg

  def get_fused_action(self):
    return self._fused_action

  def change_state(self, state):
    self._current_state = state

  def add_program_point(self, pp):
    self._cfg.append(pp)

  def create_fused_action(self):
    self._fused_action = FusedActions()

  def append_to_fused_action(self, action):
    self._fused_action.add(action)

  @classmethod
  def get_finite_automata(cls):
    context = Context()
    context.change_state(StartState(context))
    return context


class State(ABC):
  def __init__(self, context):
    self._context = context

  @abstractmethod
  def process(self, program_point):
    pass

  @classmethod
  def is_gemm(cls, program_point) -> bool:
    if program_point.action and program_point.action.isRHSExpression():
      node = program_point.action.term.node
      if isinstance(node, LoopOverGEMM):
        return True if node.is_gemm() else False
      else:
        return False
    else:
      return False


class StartState(State):
  def __init__(self, context):
    super().__init__(context)

  def process(self, program_point):
    if State.is_gemm(program_point):
      self._context.create_fused_action()
      self._context.append_to_fused_action(program_point.action)
      self._context.change_state(ProcessState(context=self._context))
    else:
      self._context.add_program_point(program_point)


class ProcessState(State):
  def __init__(self, context):
    super().__init__(context)

  def process(self, program_point):
    if State.is_gemm(program_point):
      self._context.append_to_fused_action(program_point.action)
    else:
      fused_action = self._context.get_fused_action()
      self._context.add_program_point(FusedProgramPoint(fused_action))
      self._context.add_program_point(program_point)
      self._context.change_state(StartState(context=self._context))
