#!/usr/bin/env python3
"""Kernels with a batch (diagonal) index: an index that occurs in the result
*and* in both operands of a contraction.

This is the shape produced by SeisSol's fused simulations, where every tensor
carries a leading simulation index. yateto's ast/log.py already excludes such
indices from m/n/k (`Icommon`) and turns them into loop indices; these kernels
make sure the code generator agrees.
"""

from yateto import Tensor


def add(g):
    S = 8
    N = 20
    P = 9

    # --- 1. batch index only in the result and both operands ------------------
    Q = Tensor('Q', (S, N, P))
    M = Tensor('M', (N, N))
    R = Tensor('R', (S, P, P))
    g.add('batchedGram', R['sIJ'] <= M['ij'] * Q['siI'] * Q['sjJ'])

    # --- 2. same, but the batch index is not leading --------------------------
    Q2 = Tensor('Q2', (N, P, S))
    R2 = Tensor('R2', (P, P, S))
    g.add('batchedGramTrailing', R2['IJs'] <= M['ij'] * Q2['iIs'] * Q2['jJs'])

    # --- 3. two batch indices -------------------------------------------------
    Q3 = Tensor('Q3', (S, 3, N, P))
    R3 = Tensor('R3', (S, 3, P, P))
    g.add('batchedGram2', R3['stIJ'] <= M['ij'] * Q3['stiI'] * Q3['stjJ'])

    # --- 4. batch index with a degenerate (size-1) contraction result ---------
    V = Tensor('V', (S, N))
    W = Tensor('W', (S,))
    g.add('batchedDot', W['s'] <= V['si'] * V['si'])
