# YATeTo

It is **Y**et **A**nother **Te**nsor **To**olbox for discontinuous Galerkin methods and other 
applications. You can find much more information about the package 
[here](https://arxiv.org/abs/1903.11521).

## Installation
```bash
pip install -e .
```

## Usage
```python
from yateto import *

...
def add(g):
  N = 8
  A = Tensor('A', (N, N))
  B = Tensor('B', (N, N, N))
  w = Tensor('w', (N,))
  C = Tensor('C', (N, N))
  
  kernel = C['ij'] <= 2.0 * C['ij'] + A['lj'] * B['ikl'] * w['k']
  g.add(name='kernel', ast=kernel)

# 'd' - double precision; 'hsw' - haswell-like architecture
arch = useArchitectureIdentifiedBy("dhsw")
generator = Generator(arch)
add(generator)
generator.generate(output_dir, GeneratorCollection([LIBXSMM(arch), Eigen(arch)]))
...
```

