# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from pycg import exp
from nksr.ext import sparse_solve


class SparseMatrixBlock:
    def __init__(self, a_p: torch.Tensor, a_i: torch.Tensor, a_j: torch.Tensor, a_x: torch.Tensor):
        self.a_p = a_p
        self.a_i = a_i
        self.a_j = a_j
        self.a_x = a_x


class SparseMatrix:
    def __init__(self, n_block_size: int):
        self.blocks = {}
        self.n_block_size = n_block_size

        self.block_size = [0 for _ in range(self.n_block_size)]
        self.inv_diag = [None for _ in range(self.n_block_size)]

    def add_block(self, pos_i: int, pos_j: int, size_i: int, size_j: int,
                  a_i: torch.Tensor, a_j: torch.Tensor, a_x: torch.Tensor):
        assert pos_i <= pos_j, "Only upper triangular part is allowed."
        if pos_i == pos_j:
            self.inv_diag[pos_i] = 1.0 / a_x[a_i == a_j]
        self.block_size[pos_i] = size_i
        self.block_size[pos_j] = size_j
        self.blocks[(pos_i, pos_j)] = SparseMatrixBlock(
            a_p=sparse_solve.ind2ptr(a_i, size_i),
            a_i=a_i if a_x.requires_grad else None,
            a_j=a_j, a_x=a_x
        )

    def solve(self, rhs: dict, pcg_conf):
        rhs_vec = torch.cat([rhs[d] for d in range(self.n_block_size) if d in rhs])
        if rhs_vec.requires_grad:
            ax_vec = PCGSolver.assemble_symblk(self)
        else:
            ax_vec = None
        x_vec = PCGSolver.apply(self, ax_vec, rhs_vec, pcg_conf)
        return {d: x_vec[sum(self.block_size[:d]): sum(self.block_size[:d + 1])]
                for d in range(self.n_block_size)}

    def _solve(self, rhs: torch.Tensor, pcg_conf):
        csr_p, csr_j, csr_x = {}, {}, {}
        for (di, dj), blk in self.blocks.items():
            csr_p[(di, dj)] = blk.a_p
            csr_j[(di, dj)] = blk.a_j
            csr_x[(di, dj)] = blk.a_x
        block_ptr = [sum(self.block_size[:d]) for d in range(self.n_block_size + 1)]

        res, cg_iter = sparse_solve.solve_pcg(
            csr_p, csr_j, csr_x, block_ptr,
            rhs, torch.cat([t for t in self.inv_diag if t is not None]),
            pcg_conf.tol, pcg_conf.max_iter, False
        )

        if 0 < pcg_conf.max_iter == cg_iter:
            exp.logger.warning(f"Symblk-CG-Jacobi-Cuda does not converge in {pcg_conf.max_iter} iterations.")
            exp.global_var_manager.set('skip_backward', True)

        if pcg_conf.verbose:
            exp.logger.info(f"PCG Iteration ended in {cg_iter}")

        return res


class PCGSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a: SparseMatrix, a_x: torch.Tensor, b: torch.Tensor, conf):
        with torch.no_grad():
            x = a._solve(b.detach(), conf)
        ctx.save_for_backward(x)
        ctx.a = a
        ctx.conf = conf
        return x

    @staticmethod
    def backward(ctx, grad_x):
        x,  = ctx.saved_tensors
        a: SparseMatrix = ctx.a
        with torch.no_grad():
            grad_b = a._solve(grad_x, ctx.conf)
            grad_a_x = []
            key_seqs = sorted(list(a.blocks.keys()))
            for (di, dj) in key_seqs:
                sub_a = a.blocks[(di, dj)]
                ind_i = (sub_a.a_i + sum(a.block_size[:di])).long()
                ind_j = (sub_a.a_j + sum(a.block_size[:dj])).long()
                if di == dj:
                    grad_a_x.append(-grad_b[ind_i] * x[ind_j])
                else:
                    grad_a_x.append(-grad_b[ind_i] * x[ind_j] - grad_b[ind_j] * x[ind_i])
            grad_a_x = torch.cat(grad_a_x)
        return None, grad_a_x, grad_b, None

    @staticmethod
    def assemble_symblk(a: SparseMatrix):
        """ This is used in junction with the backward pass! """
        key_seqs = sorted(list(a.blocks.keys()))
        asm_x = []
        for (di, dj) in key_seqs:
            if di != dj:
                # Just make sure no repetitive reps.
                assert (dj, di) not in a.blocks.keys()
            asm_x.append(a.blocks[(di, dj)].a_x)
        return torch.cat(asm_x)
