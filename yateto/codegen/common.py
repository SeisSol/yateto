class TensorDescription(object):
  def __init__(self, name, memoryLayout):
    self.name = name
    self.memoryLayout = memoryLayout

class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout):
    super().__init__(name, memoryLayout)
    self.indices = indices

def forLoops(cpp, indices, indexNo, body):
  if indexNo < 0:
    body()
  else:
    with cpp.For('int {0} = 0; {0} < {1}; ++{0}'.format(indices[indexNo], indices.shape()[indexNo])):
      forLoops(cpp, indices, indexNo-1, body)
