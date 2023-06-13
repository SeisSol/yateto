.. Copyright (C) 2023 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

.. _descriptor:

===================================
YATeTo immediate language reference
===================================

This document is a draft for an immediate tensor language that sits between the high-level
Einstein notation and the low-level backend-specific code.

The grammar is given in `ABNF syntax <https://www.ietf.org/rfc/rfc5234.txt>`_.

Identifier
==========

Identifiers are either named or unnamed.
Named identifiers are letter followed by letters, underscores, or digits.
Unnamed identifiers are simply numbers.
As in LLVM, local identifiers are prefixed with "%", whereas global identifiers
are prefixed with "@".

.. code:: abnf

    identifier                  = 1*DIGIT / (ALPHA *(ALPHA / DIGIT / "_"))
    local-identifier            = "%" identifier
    global-identifier           = "@" identifier

Constants
=========

.. code:: abnf

    sign                        = "-" / "+"
    integer-constant            = [sign] 1*DIGIT
    floating-constant           = [sign] *DIGIT "." 1*DIGIT ["e" [sign] 1*DIGIT]

Functions
=========

.. code:: abnf

    function-definition         = "define" global-identifier "(" [argument-list] ")" "{" block "}"
    argument-list               = argument ["," argument]
    argument                    = local-identifier ":" type

Types
=====

.. code:: abnf

    type                        = void-type / scalar-type / memref-type / batch-type
    void-type                   = "void"

Scalar types
============

.. code:: abnf

    scalar-type                 = int_type / fp_type
    int_type                    = ("i" / "u") ("8" / "16" / "32" / "64")
    fp_type                     = "f" ("32" / "64")

Scalar types are either integer ("i"), unsigned integer ("u"),
or floating point ("f").
The number behind the scalar type prefix denotes the number of bits,
e.g. "f64" are double precision floating point numbers.

Memref type
===========

.. code:: abnf

    memref-type                 = "memref<" matrix-shape "," leading-dimension, "," offset ">"
    matrix-shape                = scalar-type 2("x" integer-constant)
    leading-dimension           = integer-constant
    offset                      = integer-constant

A memref points to a region of memory that stores a matrix.
The underlying scalar type and the matrix shape is given by the ``matrix-shape`` rule.
The leading dimension is the distance in number of scalars between rows.

.. admonition:: TODO

    - Memref should be extended to arbitrary order tensors
    - Clarify offset
    - Do we need a value type?

Batch type
==========

.. code:: abnf

    batch-type                  = "batch<" (scalar-type / memref-type) "," batch-layout ">"
    batch-layout                = strided-layout / pointers-layout
    strided-layout              = "strided<" integer-constant ">"
    pointers-layout             = "pointers"

The batch type describes a batch of scalars or memrefs.
The batch is either given in a single memory region with a fixed
distance between items (strided layout) or a pointer to each item is given (pointers layout).

Instructions
============

.. code:: abnf

    block                       = *instruction
    instruction                 = value-instruction / axpy-instruction / barrier-instruction / lifetime-stop-instruction / matmul_inst
    value-instruction           = local-identifier "=" (alloca-instruction / get-work-item-instruction / submatrix-instruction)

Alloca
------

.. code:: abnf

    alloca-instruction          = "alloca" memref-type

Overview
........

The alloca instruction allocates temporary memory that is freed automatically.

Arguments
.........

The argument is the type of the returned value.

Get work item
-------------

.. code:: abnf

    get-work-item-instruction   = "get_work_item" local-identifier ["," local-identifier]

Overview
........

Get work item fetches an item from a batch.

Arguments
.........

The first operand must have the batch type.
The optional second operand must be an integer scalar type and is used to specify
an offset.

Submatrix
---------

.. code:: abnf

    submatrix-instruction       = "submatrix" local-identifier "[" slice "," slice "]"
    slice                       = integer-constant ":" integer-constant

Overview
........

The submatrix instruction returns a view on a matrix.

Arguments
.........

The local identifier must have memref type.
The instruction returns a value with memref type with appropriate size and offset.
Slices are given as [to:from), i.e. to is included and from is excluded.


Axpy
----

.. code:: abnf

    axpy-instruction            = "axpy" floating-constant "," local-identifier "," local-identifier

Overview
........

Axpy is analoguous to the BLAS-1 operation with the same name.

Arguments
.........

Axpy implements

.. math::

    B := \alpha A + B

The first argument gives :math:`\alpha`.
The second and the third argument must have memref type and give A and B, respectively.

Matrix muliplication
--------------------

.. code:: abnf

    matmul_inst                 = floating-constant "," local-identifier "," local-identifier "," floating-constant "," local-identifier

Overview
........

Matmul is analoguous to the GEMM BLAS-3 operation.

Arguments
.........

Matmul implements

.. math::

    C := \alpha A B + \beta C

The first argument gives :math:`\alpha` and the fourth argument gives :math:`\beta`.
The second, the third, and the fifth argument must have memref type and give
A, B, and C, respectively.

Additional instructions
-----------------------

.. code:: abnf

    barrier-instruction         = "barrier"
    lifetime-stop-instruction   = "lifetime_stop" local-identifier

Sample code
===========

The following sample implementes the kernel

.. math::

    D := 5 A B C + D \text{ with }
        A \in \mathbb{R}^{16\times 8},
        B \in \mathbb{R}^{8\times 8},
        C \in \mathbb{R}^{8\times 16},
        D \in \mathbb{R}^{16\times 16}

where B and C are constant matrices and A and D are matrix batches.

.. code::

   func @fused_kernel(%A: batch<memref<f32x16x8,16,0>,pointers>,
                      %B: memref<f32x8x8,8,0>,
                      %C: memref<f32x8x16,8,0>,
                      %D: batch<memref<f32x16x16,16,0>,strided<256>>) {
     %0 = get_work_item %A
     %1 = get_work_item %D
     %tmp0 = alloca memref<f32x16x8,16,0>
     matmul 1.0, %0, %B, 0.0, %tmp0
     matmul 5.0, %tmp0, %C, 1.0, %1
   }
