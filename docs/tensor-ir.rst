.. Copyright (C) 2023 Intel Corporation
   SPDX-License-Identifier: BSD-3-Clause

.. _descriptor:

===================================
YATeTo immediate language reference
===================================

This document is a draft for an immediate tensor language that sits between the high-level
Einstein notation and the low-level back-end-specific code.

The grammar is given in `ABNF syntax <https://www.ietf.org/rfc/rfc5234.txt>`_.

Core rules
==========

White space is used to separate tokens, where a token is either an identifier,
a literal, a keyword, or characters such as punctuation or delimiters.
Otherwise, white space has no meaning.

Comments start with ``;`` and stop at the end of the line (``\\n``). 

Identifier
==========

Identifiers are either named or unnamed.
Named identifiers are letter followed by letters, underscores, or digits.
Unnamed identifiers are simply numbers.
As in LLVM, local identifiers are prefixed with ``%``, whereas global identifiers
are prefixed with ``@``.

.. code:: abnf

    identifier                  = 1*DIGIT / (ALPHA *(ALPHA / DIGIT / "_"))
    local-identifier            = "%" identifier
    global-identifier           = "@" identifier

Index notation
==============

.. code:: abnf

   index                        = ALPHA
   indices                      = 1*index / "_"
   map-arity-1                  = "{" indices "to" indices "}"
   map-arity-2                  = "{" indices "," indices "to" indices "}"
   loop-index                   = "[" index "]"
   fused-indices                = "(" 1*index ")"
   indices-with-mods            = 1*(index / loop-index / fused-indices ) / "_"
   map-arity-2-with-mods        = "{" indices-with-mods "," indices-with-mods "to" indices-with-mods "}"

Tensor operations are specified using index notation.
The indices can be arbitrarily chosen but need to consistent among operands.

For example, a copy instruction could have the *map-arity-1* "{ji to ij}" that can
be thought of as the copy B[i,j] = A[j,i].
Here, a transpose operation is fused inside the copy as the order of indices i,j is switched on the right-hand-side.

The index notation can be augmented with modifiers.
For example, the tensor contraction

.. math::
    C_{ijnk} = \sum_m A_{ikm} B_{mjn}

is, for example, described as loop-over-GEMM with "{i[k]m, m(jn) to i(jn)[k]}".
Here, "[.]" means that an index is looped-over (not part of the GEMM) and
"(.)" means that two or more indices are treated as a single index.


The "_"-symbol is used to omit indices, that is, for 0-dimensional tensors.

Constants
=========

.. code:: abnf

    sign                        = "-" / "+"
    integer-constant            = [sign] 1*DIGIT
    hexdigit                    = DIGIT / ALPHA
    floating-constant           = [sign] *DIGIT "." 1*DIGIT ["e" [sign] 1*DIGIT]
    mantissa-dec                = *DIGIT "." 1*DIGIT | 1*DIGIT "."
    mantissa-hex                = *hexdigit "." 1*hexdigit | 1*hexdigit "."
    exponent                    = [sign] 1*DIGIT
    floating-constant-dec       = [sign] (mantissa-dec ["e" exponent] | 1*DIGIT "e" exponent)
    floating-constant-hex       = [sign] "0x" (mantissa-hex ["p" exponent] | 1*hexdigit "p" exponent)
    floating-constant           = floating-constant-dec | floating-constant-hex

Integer constants must lie in the range :math:`-2^63+1,\dots,2^63-1`.

Floating point constants are given in C syntax and expected to be in the range of double precision numbers.
The hexadecimal floating point syntax is supported, too.
`strtod <https://en.cppreference.com/w/c/string/byte/strtof>`_ can be used for parsing floating
point numbers.

Functions
=========

.. code:: abnf

    function-definition         = "define" global-identifier "(" [argument-list] ")" region
    argument-list               = argument *("," argument)
    argument                    = local-identifier ":" type

Regions
=======

.. code:: abnf

    region                      = "{" *instruction "}"

A region is an ordered list of instructions.
An instruction might contain a region.
Regions have access to values from its enclosing region, but the enclosing region does not have access to 
values assigned in the region.

Types
=====

.. code:: abnf

    type                        = void-type / scalar-type / memref-type / group-type
    void-type                   = "void"

Scalar types
------------

.. code:: abnf

    scalar-type                 = integer-type / floating-type
    integer-type                = ("i" / "u") ("8" / "16" / "32" / "64")
    floating-type               = "f" ("32" / "64")

Scalar types are either integer ("i"), unsigned integer ("u"),
or floating point ("f").
The number behind the scalar type prefix denotes the number of bits,
e.g. "f64" are double precision floating point numbers.

Memref type
-----------

.. code:: abnf

    memref-type                 = "memref<" tensor-shape ["," memory-layout] ">"
    tensor-shape                = scalar-type *("x" integer-constant)

A memref points to a region of memory that stores a tensor.
The underlying scalar type and the tensor shape is given by the ``tensor-shape`` rule.

The tensor can have order 0. E.g. ``memref<f32>`` can be thought of as a pointer to a single precision float.
A vector is a tensor of order 1, e.g. ``memref<f64x4>``.
A matrix is a tensor of order 2, e.g. ``memref<f64x4x4>``.
A tensor of order n is given by ``memref<f32xs_1x...xs_n>``.


The default memory layout is the packed dense layout.
E.g. the memory layout of ``memref<f32x5x6x7>`` is ``strided<1,5,30>``.
We note that ``memref<f32x5x6x7>`` and ``memref<f32x5x6x7,strided<1,5,30>>``
are the same type.


.. admonition:: Discussion

    - Do we need a tensor value type?

Memory layout
.............

.. code:: abnf

    memory-layout               = strided-layout

Strided layout
~~~~~~~~~~~~~~

.. code:: abnf

    strided-layout              = "strided<" [integer-list] ">"
    integer-list                = integer-constant *("," integer-constant)

The strided layout is a sequence of integers :math:`S_1,S_2,...,S_n`, where *n* must be equal
to the order of the tensor.
The strided layout is defined as the map

.. math::

    (i_1,i_2,...,i_n) \mapsto i_1 S_1 + i_2 S_2 + ... + i_n S_n

We further impose the following restriction for a tensor with shape :math:`s_1\times s_2 \times ... \times s_n`:

* :math:`1 \leq S_1`
* :math:`\forall i \in [2,n]: S_{i-1}s_{i-1} \leq S_i`

Therefore, we have the "column-major" layout.
The default packed dense layout is given by

* :math:`1 = S_1`
* :math:`\forall i \in [2,n]: S_{i-1}s_{i-1} = S_i`

Group type
----------

.. code:: abnf

    group-type                  = "group<" memref-type "," group-layout ">"
    group-layout                = distance-layout / pointer-layout
    distance-layout             = "distance<" integer-constant ">"
    pointers-layout             = "pointers"

The group type describes a group of memrefs.
The group is either given in a single memory region with a fixed
distance between items (distance layout) or a pointer to each item is given (pointers layout).

.. admonition:: Discussion

    - Instead of ``group<..., distance<...>>`` one could use tensors with dynamic size.
      E.g. instead of ``group<tensor<f32x4>, distance<4>>`` one might use
      ``memref<f32x4x?>``. That would be nice from a conceptual point of view but then
      we would need do deal with tensors with potentially unknown size in every instruction.

Instructions
============

.. code:: abnf

    instruction                 = value-instruction
                                  / axpby-instruction
                                  / barrier-instruction
                                  / lifetime-stop-instruction
                                  / log-instruction
                                  / for-instruction
                                  / product-instruction
                                  / sum-instruction
    value-instruction           = local-identifier "=" (alloca-instruction / get-work-item-instruction / subview-instruction)

Alloca
------

.. code:: abnf

    alloca-instruction          = "alloca" ":" memref-type

Overview
........

The alloca instruction allocates temporary memory that is freed automatically at the end of the block that contains the alloca.

Arguments
.........

The argument is the type of the returned value.

Get work item
-------------

.. code:: abnf

    get-work-item-instruction   = "get_work_item" local-identifier ["," integer-type local-identifier] ":" group-type

Overview
........

Get work item fetches an item from a batch.

Arguments
.........

The first operand must have the batch type.
The optional second operand must be an integer scalar type and is used to specify
an offset.

Subview
-------

.. code:: abnf

    subview-instruction         = "subview" local-identifier "[" [index-or-slice-list] "]" ":" memref-type "to" memref-type
    index-or-slice-list         = index-or-slice *("," index-or-slice)
    index-or-slice              = integer-type local-identifier | integer-constant | slice
    slice                       = [integer-constant] ":" [integer-constant]

Overview
........

The subview instruction returns a view on a tensor.

Arguments
.........

The local identifier must have the left-hand memref type and the instruction returns the right-hand memref type.
Slices are given as [to:from), i.e. to is included and from is excluded.


Axpby
-----

.. code:: abnf

    axpby-instruction           = "axpby" map-arity-1 "," floating-constant "," local-identifier "," local-identifier ":" memref-type "to" memref-type

Overview
........

Axpby implements

.. math::

    B[\pi_B(I)] := \alpha A[\pi_A(I)] + \beta B[\pi_B(I)]

Arguments
.........

The first argument gives the index map that defines the indices I
as well as the permutation :math:`\pi_A, \pi_B`.
Note that the input and output indices in the index map must be equal
up to permutation.
The second argument gives :math:`\alpha`.
The third and the fourth argument must have memref type and give A and B, respectively.
The number of indices must be equal to the order of A and B.

Loop-over-GEMM
--------------

.. code:: abnf

    log-instruction          = "log" map-arity-2-with-mods "," floating-constant "," local-identifier "," local-identifier "," floating-constant "," local-identifier ":" memref-type "," memref-type "to" memref-type

Overview
........

Loop-over-GEMM implements the well-known GEMM BLAS-3 operation
wrapped in loops.

Arguments
.........

The loop-over-GEMM operation implements

.. math::

    C[\pi_C(I_m\cup I_n)] := \alpha \sum_{I_k}
        A[\pi_A(I_m \cup I_k)] B[\pi_B(I_k \cup I_n)]
        + \beta C[\pi_C(I_m\cup I_n)]

The permuations and index sets are given by the index map (first argument).
The index map defines the three sets :math:`I_A, I_B, I_C` and we have

.. math::

   I_{common} = I_A \cap I_B \cap I_C

   I_m = I_A \cap I_C \setminus I_{common}
   
   I_n = I_B \cap I_C \setminus I_{common}

   I_k = I_A \cap I_B \setminus I_{common}

.. admonition:: Todo

   Specify modifiers.

The second argument gives :math:`\alpha` and the fifth argument gives :math:`\beta`.
The third, the fourth, and the sixth argument must have memref type and give
A, B, and C, respectively.

For
---

.. code:: abnf

    for-instruction        = "for" integer-type local_identifier "=" integer-constant "to" integer-constant region

Overview
........

It's a for loop.

The loop's range [from; to) is given by the first integer constant and second integer constant.
The trip count is stored in the local identifier.

Product
-------

.. code:: abnf

    product-instruction          = "product" map-arity-2 "," floating-constant "," local-identifier "," local-identifier "," floating-constant "," local-identifier ":" memref-type "," memref-type "to" memref-type

Overview
........

Product multiplies two tensors without reduction (sum over index).

Arguments
.........

The product operation implements

.. math::

    C[\pi_C(I_C)] := \alpha
        A[\pi_A(I_A)] B[\pi_B(I_B)]
        + \beta C[\pi_C(I_C)]

The permuations and index sets are given by the index map (first argument).
The index map defines the three sets :math:`I_A, I_B, I_C` and it
is required that

.. math::

   I_C = I_A \cup I_B

The second argument gives :math:`\alpha` and the fifth argument gives :math:`\beta`.
The third, the fourth, and the sixth argument must have memref type and give
A, B, and C, respectively.

Sum
---

.. code:: abnf

    sum-instruction          = "sum" map-arity-1 "," floating-constant "," local-identifier "," floating-constant "," local-identifier ":" memref-type "to" memref-type

Overview
........

Sum over indices.

Arguments
.........

The sum operation implements

.. math::

    B[\pi_B(I_B)] := \alpha \sum_{I_s}
        A[\pi_A(I_A)] + \beta B[\pi_B(I_B)]

The permuations and index sets are given by the index map (first argument).
The index map defines the two sets :math:`I_A, I_B` and we require

.. math::

   I_B \subset I_A

   I_{s} = I_A \setminus I_B

The second argument gives :math:`\alpha` and the fourth argument gives :math:`\beta`.
The third and the fifth argument must have memref type and give
A and B, respectively.


Additional instructions
-----------------------

.. code:: abnf

    barrier-instruction         = "barrier"
    lifetime-stop-instruction   = "lifetime_stop" local-identifier

Sample code
===========

The following sample implements the kernel

.. math::

    D := 5 A B C + D \text{ with }
        A \in \mathbb{R}^{16\times 8},
        B \in \mathbb{R}^{8\times 8},
        C \in \mathbb{R}^{8\times 16},
        D \in \mathbb{R}^{16\times 16}

where B and C are constant matrices and A and D are matrix batches.

.. code::

   define @fused_kernel(%A: group<memref<f32x16x8>,pointers>,
                        %B: memref<f32x8x8>,
                        %C: memref<f32x8x16>,
                        %D: group<memref<f32x16x16>,distance<256>>) {
     %0 = get_work_item %A : group<memref<f32x16x8>,pointers> 
     %1 = get_work_item %D : group<memref<f32x16x16>,distance<256>> 
     %tmp0 = alloca : memref<f32x16x8>
     log {ik, kj to ij} 1.0, %0, %B, 0.0, %tmp0
        : memref<f32x16x8>, memref<f32x8x8> to memref<f32x8x16> 
     log {ik, kj to ij} 5.0, %tmp0, %C, 1.0, %1
        : memref<f32x16x8>, memref<f32x8x16> to memref<f32x16x16>
   }
