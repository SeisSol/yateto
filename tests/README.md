## Testing
Testing is divided into 3 parts: interface, generic and code-gen.

- *interface* - yateto comes with some helper structures that 
external projects can use. The structures are defined 
in the *\<package root\>*/**include** directory and allow a user to copy data from one 
tensor to another if they have the same dimensionality but 
different sizes. For example, a target tensor may be padded along the leading dimension 
to achieve efficient vectorization. This part of the testing is supposed to 
check whether the provided structures operate correctly.

- *generic* - yateto generates an optimized tensor contraction source code based on the
*Loop over GEMM* algorithm. To achieve better performance, the generated source code 
contains calls to optimized GEMM libraries and generators. However, yateto can 
also generate not-optimized source code using simple *for-loops* 
which can be used either for performance comparison or for testing with respect
to an optimized one. This part of the testing is supposed to check whether the 
*generic* implementation of tensor contraction is numerically correct.


- *code-gen* - checks wither an optimized tensor contraction code produces the same
numerical results as the *generic* implementation.

The testing is supposed to only be performed with **GNU** tools and, therefore, 
no platform specific libraries (*like intel MKL*) are included. Moreover, only the
following architectures are supported for testing: *sandy bridge, haswell, skylake, 
ThunderX2*. 

In order to compile tests, make sure that you have **CxxTest** installed and visible
in you current working environment.

## Current status
|    Part   |      Status     |
|:---------:|:---------------:|
| interface |     1 test     |
| generic   | not implemented |
| code-gen  |     2 tests     |


## Running tests manually
### Interface
```console
cd mkdir interface/build && cd interface/build
cmake ..
make
ctest
```

### Code-gen
Code-gen allows a user to test yateto with one of the following GEMM libraries/generators: Eigen, OpenBLAS,  LIBXSMM. Make sure that you have them installed on your system and visible in your current working environment.

##### CMake options
| CMake Variable Name |  Type  |         Allowed Values         |
|:-------------------:|:------:|:------------------------------:|
| ARCH                | string |    snb / hsw / skx / thunderx2t99    |
| EXAMPLES            |  list  | matmul / minimal / matmult;minimal |
| PRECISION           | string |          double / single         |
| VARIANT             | string |     Eigen / OpenBLAS / LIBXSMM     |

##### Default
Uses: **haswell** architecture, **matmul** and **minimal** as examples, **Eigen** 
as a GEMM implementation, **double** precision.
```console
cd mkdir code-gen/build && cd code-gen/build
cmake ..
make
ctest
```
##### A Specific Example
For **haswell** architecture with **single** precision and **libxsmm** 
as a GEMM generator.
```console
cd mkdir code-gen/build && cd code-gen/build
cmake .. -DPRECISION=single -DVARIANT=LIBXSMM
make
ctest
```

## Running tests automatically
The following [pipeline](Jenkinsfile) has been implemented to run the aforementioned tests automatically. As a regular user, you can see results of the last few runs of the pipeline [here](http://vmbungartz10.informatik.tu-muenchen.de/seissol/view/Yateto/job/yateto-codegen/). 

You can trigger the pipeline and thus run all tests if you a member of SeisSol in github. To achive this, please, perform the following steps:

- open this [page](http://vmbungartz10.informatik.tu-muenchen.de/seissol/view/Yateto/job/yateto-codegen/)
- click on `log in` button at the top right corner and follow the authentication procedure
- click on `Build with Parameters` button. You will be forwarded to the next page where you can adjust parameters. We do not recommend to make any changes in `AGENT` and `BUILD_ENV_IMAGE` fields
- click on `Build` to trigger the pipeline. 
- After that, you will see a new flashing entry at the very top of `Build History` field. If you want to see a detail status information about all steps involved in the pipeline then click on a dropdown widget of the flashing entry and select `Console Output`