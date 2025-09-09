#include <torch/extension.h>

std::pair<torch::Tensor, int> solve_pcg_cuda(
        py::dict Ap, py::dict Aj, py::dict Ax, const std::vector<int64_t>& block_ptr,
        const torch::Tensor& b,
        const torch::Tensor& inv_diag_A,
        const float tol, const int max_iter, const bool res_fix);

std::pair<torch::Tensor, int> solve_pcg_cpu(
        py::dict Ap, py::dict Aj, py::dict Ax, const std::vector<int64_t>& block_ptr,
        const torch::Tensor& b,
        const torch::Tensor& inv_diag_A,
        const float tol, const int max_iter, const bool res_fix);

torch::Tensor ind2ptr_cpu(torch::Tensor ind, int64_t M);
torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M);

torch::Tensor ptr2ind_cpu(torch::Tensor ptr, int64_t E);
torch::Tensor ptr2ind_cuda(torch::Tensor ptr, int64_t E);

// Let's dispatch device outside, due to possibly different interfaces.
void pybind_sparse_solve(py::module& m) {
    m.def("solve_pcg", [](py::dict Ap, py::dict Aj, py::dict Ax, const std::vector<int64_t>& block_ptr,
                          const torch::Tensor& b,
                          const torch::Tensor& inv_diag_A,
                          const float tol, const int max_iter, const bool res_fix) {
        if (b.device().is_cuda()) {
            return solve_pcg_cuda(Ap, Aj, Ax, block_ptr, b, inv_diag_A, tol, max_iter, res_fix);
        } else {
            return solve_pcg_cpu(Ap, Aj, Ax, block_ptr, b, inv_diag_A, tol, max_iter, res_fix);
        }
    }, "Solve sparse matrix using ConjGrad.");
    m.def("ind2ptr", [](torch::Tensor ind, int64_t M) {
        if (M == -1) {
            M = ind.max().item().toLong() + 2;
        }
        if (ind.device().is_cuda()) {
            return ind2ptr_cuda(ind, M);
        } else {
            return ind2ptr_cpu(ind, M);
        }
    }, "Convert row indices to pointer.", py::arg("ind"), py::arg("M") = -1);
    m.def("ptr2ind", [](torch::Tensor ptr, int64_t E) {
        if (E == -1) {
            E = ptr[-1].item().toLong();
        }
        if (ptr.device().is_cuda()) {
            return ptr2ind_cuda(ptr, E);
        } else {
            return ptr2ind_cpu(ptr, E);
        }
    }, "Convert pointer to row indices.", py::arg("ptr"), py::arg("E") = -1);
}
