#include <ATen/Parallel.h>
#include <torch/torch.h>
#include <Eigen/Sparse>
#include <pybind11/pytypes.h>

namespace py = pybind11;

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")
#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")

typedef std::pair<int, int> BlockInds;
typedef Eigen::VectorXf EVector;
typedef Eigen::Map<EVector> EVectorMap;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> ESpMat;


template <typename IntegerT>
torch::Tensor dispatch_ind2ptr_cpu(torch::Tensor ind, int64_t M) {
    CHECK_CPU(ind);
    auto out = torch::empty(M + 1, ind.options());
    auto ind_data = ind.accessor<IntegerT, 1>();
    auto out_data = out.accessor<IntegerT, 1>();

    int64_t numel = ind.numel();

    if (numel == 0)
        return out.zero_();

    for (IntegerT i = 0; i <= ind_data[0]; i++)
        out_data[i] = 0;

    int64_t grain_size = at::internal::GRAIN_SIZE;
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
        IntegerT idx = ind_data[begin], next_idx;
        for (IntegerT i = begin; i < std::min(end, numel - 1); i++) {
            next_idx = ind_data[i + 1];
            for (; idx < next_idx; idx++)
                out_data[idx + 1] = i + 1;
        }
    });

    for (IntegerT i = ind_data[numel - 1] + 1; i < M + 1; i++)
        out_data[i] = numel;

    return out;
}

template <typename IntegerT>
torch::Tensor dispatch_ptr2ind_cpu(torch::Tensor ptr, int64_t E) {
    CHECK_CPU(ptr);
    auto out = torch::empty(E, ptr.options());
    auto ptr_data = ptr.accessor<IntegerT, 1>();
    auto out_data = out.accessor<IntegerT, 1>();

    int64_t numel = ptr.numel();

    int64_t grain_size = at::internal::GRAIN_SIZE;
    at::parallel_for(0, numel - 1, grain_size, [&](int64_t begin, int64_t end) {
        IntegerT idx = ptr_data[begin], next_idx;
        for (IntegerT i = begin; i < end; i++) {
            next_idx = ptr_data[i + 1];
            for (IntegerT e = idx; e < next_idx; e++)
                out_data[e] = i;
            idx = next_idx;
        }
    });

    return out;
}

torch::Tensor ind2ptr_cpu(torch::Tensor ind, int64_t M) {
    torch::Tensor out;
    AT_DISPATCH_INDEX_TYPES(ind.scalar_type(), "dispatch_ind2ptr_cpu", [&]() {
        out = dispatch_ind2ptr_cpu<index_t>(ind, M);
    });
    return out;
}

torch::Tensor ptr2ind_cpu(torch::Tensor ptr, int64_t E) {
    torch::Tensor out;
    AT_DISPATCH_INDEX_TYPES(ptr.scalar_type(), "dispatch_ptr2ind_cpu", [&]() {
        out = dispatch_ptr2ind_cpu<index_t>(ptr, E);
    });
    return out;
}

static inline void symblk_matmul(
        EVector& d_Ax, EVector& d_x, int n_block,
        std::map<BlockInds, ESpMat*>& matA_map, const std::vector<int64_t>& block_ptr) {
    d_Ax.setZero();
    for (int i = 0; i < n_block; ++i) {
        for (int j = 0; j < n_block; ++j) {
            BlockInds ij(i, j), ji(j, i);
            if (matA_map.find(ij) != matA_map.end()) {
                d_Ax.segment(block_ptr[i], block_ptr[i + 1] - block_ptr[i]) +=
                        *matA_map.at(ij) * d_x.segment(block_ptr[j], block_ptr[j + 1] - block_ptr[j]);
                continue;
            }
            if (matA_map.find(ji) != matA_map.end()) {
                d_Ax.segment(block_ptr[i], block_ptr[i + 1] - block_ptr[i]) +=
                        matA_map.at(ji)->transpose() * d_x.segment(block_ptr[j], block_ptr[j + 1] - block_ptr[j]);
            }
        }
    }
}

std::pair<torch::Tensor, int> solve_pcg_cpu(
        py::dict Ap, py::dict Aj, py::dict Ax, const std::vector<int64_t>& block_ptr,
        const torch::Tensor& b,
        const torch::Tensor& inv_diag_A,
        const float tol, const int max_iter, const bool res_fix) {

    CHECK_CONTIGUOUS(b); CHECK_CPU(b); CHECK_IS_FLOAT(b);
    CHECK_CONTIGUOUS(inv_diag_A); CHECK_CPU(inv_diag_A); CHECK_IS_FLOAT(inv_diag_A);

    int n_block = block_ptr.size() - 1;
    int64_t N = b.size(0);
    int sqrt_n = (int) std::ceil(std::sqrt((double) N));

    const EVectorMap eigen_b(b.data_ptr<float>(), N);
    const EVectorMap eigen_inv_A(inv_diag_A.data_ptr<float>(), N);
    float b_norm = eigen_b.norm();
    float atol = tol * b_norm;

    EVector eigen_x = EVector::Zero(N);
    EVector eigen_d_p = EVector::Zero(N);
    EVector eigen_d_z = EVector::Zero(N);
    EVector eigen_d_r = eigen_b;
    EVector eigen_Ax = EVector::Zero(N);

    std::map<BlockInds, ESpMat*> matA_map;
    for (auto it = Ap.begin(); it != Ap.end(); ++it) {
        auto tp = it->first.cast<py::tuple>();
        BlockInds ind = std::make_pair(tp[0].cast<int>(), tp[1].cast<int>());

        int dim_i = block_ptr[ind.first + 1] - block_ptr[ind.first];
        int dim_j = block_ptr[ind.second + 1] - block_ptr[ind.second];
        torch::Tensor ap_tensor = Ap[tp].cast<torch::Tensor>();
        torch::Tensor aj_tensor = Aj[tp].cast<torch::Tensor>();
        torch::Tensor ax_tensor = Ax[tp].cast<torch::Tensor>();
        CHECK_CONTIGUOUS(ap_tensor); CHECK_CPU(ap_tensor); CHECK_IS_INT(ap_tensor);
        CHECK_CONTIGUOUS(aj_tensor); CHECK_CPU(aj_tensor); CHECK_IS_INT(aj_tensor);
        CHECK_CONTIGUOUS(ax_tensor); CHECK_CPU(ax_tensor); CHECK_IS_FLOAT(ax_tensor);

        auto* a_mat = new ESpMat(dim_i, dim_j);
        a_mat->resizeNonZeros(ax_tensor.size(0));
        at::parallel_for(0, ax_tensor.size(0), at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < std::min(end, ax_tensor.size(0)); i++) {
                a_mat->data().index(i) = aj_tensor.data_ptr<int>()[i];
                a_mat->data().value(i) = ax_tensor.data_ptr<float>()[i];
            }
        });
        at::parallel_for(0, ap_tensor.size(0), at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < std::min(end, ap_tensor.size(0)); i++) {
                a_mat->outerIndexPtr()[i] = ap_tensor.data_ptr<int>()[i];
            }
        });
        matA_map[ind] = a_mat;
    }

    // [r = b - matvec(x)]
    symblk_matmul(eigen_Ax, eigen_x, n_block, matA_map, block_ptr);
    eigen_d_r -= eigen_Ax;

    int iters = 0;
    float rho = 0.0f;
    float rho1;

    while (max_iter < 0 || iters < max_iter) {
        // [z = psolve(r)]
        eigen_d_z = eigen_inv_A.cwiseProduct(eigen_d_r);
        rho1 = rho;
        // [rho = dot(r, z)]
        rho = eigen_d_r.dot(eigen_d_z);
        if (iters == 0) {
            // [p = z]
            eigen_d_p = eigen_d_z;
        } else {
            float betap = rho / rho1;
            // [p = z + beta * p]
            eigen_d_p = eigen_d_z + betap * eigen_d_p;
        }
        // [q = matvec(p)]
        symblk_matmul(eigen_Ax, eigen_d_p, n_block, matA_map, block_ptr);
        // [alpha = rho / dot(p, q)]
        float alpha = rho / eigen_d_p.dot(eigen_Ax);
        // [x = x + alpha * p]
        eigen_x += alpha * eigen_d_p;
        if ((iters + 1) % sqrt_n == 0 && res_fix) {
            // [r = b - matvec(x)]
            symblk_matmul(eigen_Ax, eigen_x, n_block, matA_map, block_ptr);
            eigen_d_r = eigen_b - eigen_Ax;
        } else {
            eigen_d_r -= alpha * eigen_Ax;
        }
        iters ++;
        // [resid = nrm2(r)]
        if (eigen_d_r.norm() <= atol) {
            break;
        }
    }

    for (const auto& it : matA_map) {
        delete it.second;
    }

    torch::Tensor x_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    EVectorMap(x_tensor.data_ptr<float>(), N) = eigen_x;

    return std::make_pair(x_tensor, iters);
}
