#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <time.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

void cusparseSafeCall(cusparseStatus_t err);
void cublasSafeCall(cublasStatus_t err);

typedef std::pair<int, int> BlockInds;
#define THREADS 256

template <typename ScalarT>
__global__ void ind2ptr_kernel(const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> ind_data,
                               torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> out_data,
                               int64_t M, int64_t numel) {

    ScalarT thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_idx == 0) {
        for (ScalarT i = 0; i <= ind_data[0]; i++)
            out_data[i] = 0;
    } else if (thread_idx < numel) {
        for (ScalarT i = ind_data[thread_idx - 1]; i < ind_data[thread_idx]; i++)
            out_data[i + 1] = thread_idx;
    } else if (thread_idx == numel) {
        for (ScalarT i = ind_data[numel - 1] + 1; i < M + 1; i++)
            out_data[i] = numel;
    }
}

template <typename ScalarT>
__global__ void ptr2ind_kernel(const torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> ptr_data,
                               torch::PackedTensorAccessor32<ScalarT, 1, torch::RestrictPtrTraits> out_data,
                               int64_t E, int64_t numel) {

    ScalarT thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_idx < numel) {
        ScalarT idx = ptr_data[thread_idx], next_idx = ptr_data[thread_idx + 1];
        for (ScalarT i = idx; i < next_idx; i++) {
            out_data[i] = thread_idx;
        }
    }
}

__global__ static void apply_jacobi(const float *a, const float *b, float *res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = a[i] * b[i];
    }
}

static inline void symblk_matmul(
        cusparseHandle_t cusparseHandle,
        float* d_Ax, size_t N, int n_block,
        std::map<BlockInds, cusparseSpMatDescr_t>& matA_map,
        std::vector<cusparseDnVecDescr_t>& vecx,
        std::vector<cusparseDnVecDescr_t>& vecAx,
        unsigned char* buffer,
        const std::vector<int64_t>& block_ptr) {
    float one = 1.0;

    cudaMemset(d_Ax, 0, sizeof(float) * N);
    for (int i = 0; i < n_block; ++i) {
        for (int j = 0; j < n_block; ++j) {
            BlockInds ij(i, j), ji(j, i);
            if (matA_map.find(ij) != matA_map.end()) {
                cusparseSafeCall(cusparseSpMV(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &one, matA_map.at(ij), vecx[j], &one, vecAx[i], CUDA_R_32F,
                        CUSPARSE_SPMV_ALG_DEFAULT, buffer));
                continue;
            }
            if (matA_map.find(ji) != matA_map.end()) {
                cusparseSafeCall(cusparseSpMV(
                        cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                        &one, matA_map.at(ji), vecx[j], &one, vecAx[i], CUDA_R_32F,
                        CUSPARSE_SPMV_ALG_DEFAULT, buffer));
            }
        }
    }
}

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M) {
    CHECK_CUDA(ind);
    cudaSetDevice(ind.get_device());

    auto out = torch::empty({M + 1}, ind.options());

    if (ind.numel() == 0)
        return out.zero_();

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_INDEX_TYPES(ind.scalar_type(), "ind2ptr_kernel", [&]() {
        ind2ptr_kernel<index_t><<<(ind.numel() + 2 + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                ind.packed_accessor32<index_t, 1, torch::RestrictPtrTraits>(),
                out.packed_accessor32<index_t, 1, torch::RestrictPtrTraits>(),
                M, ind.numel()
                );
    });

    return out;
}

torch::Tensor ptr2ind_cuda(torch::Tensor ptr, int64_t E) {
    CHECK_CUDA(ptr);
    cudaSetDevice(ptr.get_device());

    auto out = torch::empty({E}, ptr.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_INDEX_TYPES(ptr.scalar_type(), "ptr2ind_kernel", [&]() {
        ptr2ind_kernel<index_t><<<(ptr.numel() - 1 + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                ptr.packed_accessor32<index_t, 1, torch::RestrictPtrTraits>(),
                out.packed_accessor32<index_t, 1, torch::RestrictPtrTraits>(),
                E, ptr.numel() - 1
                );
    });

    return out;
}

/**
 * Preconditioned Conjugate Gradient with Jacobi(diagonal)-style preconds.
 * @param Ap torch.Tensor: CSR compressed row of A
 * @param Aj torch.Tensor: CSR column indices of A
 * @param Ax torch.Tensor: CSR values of A
 * @param b torch.Tensor: RHS
 * @param inv_diag_A torch.Tensor: inverse of A only diagonal part.
 * @param tol float: tolerance for convergence.
 * @param max_iter int: (-1) to allow running forever.
 * @param res_fix bool: whether or not to do periodical residual computation fixes.
 * @return solution x (torch.Tensor) and converged iterations (int)
 */
std::pair<torch::Tensor, int> solve_pcg_cuda(
        py::dict Ap, py::dict Aj, py::dict Ax, const std::vector<int64_t>& block_ptr,
        const torch::Tensor& b,
        const torch::Tensor& inv_diag_A,
        const float tol, const int max_iter, const bool res_fix) {

    CHECK_CONTIGUOUS(b); CHECK_CUDA(b); CHECK_IS_FLOAT(b);
    CHECK_CONTIGUOUS(inv_diag_A); CHECK_CUDA(inv_diag_A); CHECK_IS_FLOAT(inv_diag_A);

    int n_block = block_ptr.size() - 1;
    int N = b.size(0);
    int sqrt_n = (int) std::ceil(std::sqrt((double) N));

    // Determine tolerance -- atol
    float b_norm = torch::linalg_norm(b).item<float>();
    float atol = tol * b_norm;

    // Constant for cublas
    float one = 1.0;
    float neg_one = -1.0;

    // Memory allocation
    torch::Tensor x_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor d_p_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor d_z_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor d_r_tensor = torch::clone(b);
    torch::Tensor Ax_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    float* d_x = x_tensor.data_ptr<float>();
    float* d_p = d_p_tensor.data_ptr<float>();
    float* d_z = d_z_tensor.data_ptr<float>();
    float* d_r = d_r_tensor.data_ptr<float>();
    float* d_Ax = Ax_tensor.data_ptr<float>();
    float* d_b = b.data_ptr<float>();
    const float* d_inv_diag_A = inv_diag_A.data_ptr<float>();

    cusparseHandle_t cusparseHandle = at::cuda::getCurrentCUDASparseHandle();
    cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();

    // Create sparse tensors and vectors
    std::map<BlockInds, cusparseSpMatDescr_t> matA_map;
    std::vector<cusparseDnVecDescr_t> vecx(n_block, nullptr);
    std::vector<cusparseDnVecDescr_t> vecp(n_block, nullptr);
    std::vector<cusparseDnVecDescr_t> vecAx(n_block, nullptr);
    size_t bufferSize = 0;

    for (auto it = Ap.begin(); it != Ap.end(); ++it) {
        auto tp = it->first.cast<py::tuple>();
        BlockInds ind = std::make_pair(tp[0].cast<int>(), tp[1].cast<int>());

        int dim_i = block_ptr[ind.first + 1] - block_ptr[ind.first];
        int dim_j = block_ptr[ind.second + 1] - block_ptr[ind.second];
        torch::Tensor ap_tensor = Ap[tp].cast<torch::Tensor>();
        torch::Tensor aj_tensor = Aj[tp].cast<torch::Tensor>();
        torch::Tensor ax_tensor = Ax[tp].cast<torch::Tensor>();
        CHECK_CONTIGUOUS(ap_tensor); CHECK_CUDA(ap_tensor); CHECK_IS_INT(ap_tensor);
        CHECK_CONTIGUOUS(aj_tensor); CHECK_CUDA(aj_tensor); CHECK_IS_INT(aj_tensor);
        CHECK_CONTIGUOUS(ax_tensor); CHECK_CUDA(ax_tensor); CHECK_IS_FLOAT(ax_tensor);
        cusparseSpMatDescr_t matA = nullptr;
        cusparseSafeCall(cusparseCreateCsr(
                &matA,
                dim_i,
                dim_j,
                ax_tensor.size(0),
                ap_tensor.data_ptr<int>(), aj_tensor.data_ptr<int>(), ax_tensor.data_ptr<float>(),
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        matA_map[ind] = matA;

        if (vecx[ind.first] == nullptr) {
            cusparseSafeCall(cusparseCreateDnVec(&vecx[ind.first], dim_i, d_x + block_ptr[ind.first], CUDA_R_32F));
            cusparseSafeCall(cusparseCreateDnVec(&vecp[ind.first], dim_i, d_p + block_ptr[ind.first], CUDA_R_32F));
            cusparseSafeCall(cusparseCreateDnVec(&vecAx[ind.first], dim_i, d_Ax + block_ptr[ind.first], CUDA_R_32F));
        }

        if (vecx[ind.second] == nullptr) {
            cusparseSafeCall(cusparseCreateDnVec(&vecx[ind.second], dim_j, d_x + block_ptr[ind.second], CUDA_R_32F));
            cusparseSafeCall(cusparseCreateDnVec(&vecp[ind.second], dim_j, d_p + block_ptr[ind.second], CUDA_R_32F));
            cusparseSafeCall(cusparseCreateDnVec(&vecAx[ind.second], dim_j, d_Ax + block_ptr[ind.second], CUDA_R_32F));
        }

        size_t curBufferSize = 0;
        cusparseSafeCall(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecx[ind.second],
                &one, vecAx[ind.first], CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &curBufferSize));
        bufferSize = std::max(bufferSize, curBufferSize);
        if (ind.first != ind.second) {
            cusparseSafeCall(cusparseSpMV_bufferSize(
                    cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, &one, matA, vecx[ind.first],
                    &one, vecAx[ind.second], CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &curBufferSize));
            bufferSize = std::max(bufferSize, curBufferSize);
        }
    }
    torch::Tensor cusparse_buffer_tensor = torch::empty({(int)bufferSize}, torch::dtype(torch::kByte).device(torch::kCUDA));
    auto* buffer = cusparse_buffer_tensor.data_ptr<unsigned char>();

    /* Begin CG */
    // [r = b - matvec(x)]
    // vecAx = alpha * matA @ vecx + beta * vecAx = matA @ vecx
    symblk_matmul(cusparseHandle, d_Ax, N, n_block, matA_map, vecx, vecAx, buffer, block_ptr);

    // d_r = alpham1 * d_Ax + d_r
    cublasSafeCall(cublasSaxpy(cublasHandle, N, &neg_one, d_Ax, 1, d_r, 1));

    int iters = 0;
    float rho = 0.0;
    float rho1;

    while (max_iter < 0 || iters < max_iter) {
        // [z = psolve(r)]
        {
            dim3 dimBlock = dim3(256);
            dim3 dimGrid = dim3((N + dimBlock.x - 1) / dimBlock.x);
            apply_jacobi<<<dimGrid, dimBlock>>>(d_inv_diag_A, d_r, d_z, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        rho1 = rho;

        // [rho = cublas.dotc(r, z)]
        cublasSafeCall(cublasSdot(cublasHandle, N, d_r, 1, d_z, 1, &rho));

        if (iters == 0) {
            // [p = z]
            cublasSafeCall(cublasScopy(cublasHandle, N, d_z, 1, d_p, 1));
        } else {
            float betap = rho / rho1;
            // [p = z + beta * p]
            cublasSafeCall(cublasSscal(cublasHandle, N, &betap, d_p, 1));
            cublasSafeCall(cublasSaxpy(cublasHandle, N, &one, d_z, 1, d_p, 1));
        }

        // [q = matvec(p)], where q --> Ax
        symblk_matmul(cusparseHandle, d_Ax, N, n_block, matA_map, vecp, vecAx, buffer, block_ptr);

        float alpha, neg_alpha, dot;
        // [alpha = rho / cublas.dotc(p, q)]
        cublasSafeCall(cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot));
        alpha = rho / dot;

        // [x = x + alpha * p]
        cublasSafeCall(cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

        if ((iters + 1) % sqrt_n == 0 && res_fix) {
            // [r = b - matvec(x)]
            symblk_matmul(cusparseHandle, d_Ax, N, n_block, matA_map, vecx, vecAx, buffer, block_ptr);
            cublasSafeCall(cublasScopy(cublasHandle, N, d_b, 1, d_r, 1));
            cublasSafeCall(cublasSaxpy(cublasHandle, N, &neg_one, d_Ax, 1, d_r, 1));
        } else {
            // [r = r - alpha * q]
            neg_alpha = -alpha;
            cublasSafeCall(cublasSaxpy(cublasHandle, N, &neg_alpha, d_Ax, 1, d_r, 1));
        }

        iters++;

        // [resid = cublas.nrm2(r)]
        float resid;
        cublasSafeCall(cublasSnrm2(cublasHandle, N, d_r, 1, &resid));

//        cudaDeviceSynchronize();
//        printf("iteration = %3d, residual = %e\n", iters, sqrt(resid));
//        fflush(0);

        if (resid <= atol) {
            break;
        }

    }

    for (const auto& it : matA_map) {
        cusparseSafeCall(cusparseDestroySpMat(it.second));
    }
    for (const auto& it : vecx) {
        if (it) cusparseSafeCall(cusparseDestroyDnVec(it));
    }
    for (const auto& it : vecp) {
        if (it) cusparseSafeCall(cusparseDestroyDnVec(it));
    }
    for (const auto& it : vecAx) {
        if (it) cusparseSafeCall(cusparseDestroyDnVec(it));
    }

    return std::make_pair(x_tensor, iters);
}
