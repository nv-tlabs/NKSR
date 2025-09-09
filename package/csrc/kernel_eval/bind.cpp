#include <torch/extension.h>
#include "SparseFeatureIndexGrid.h"

#include "keval.h"

#define CHECK_DEVICE(a, b) TORCH_CHECK(a.device() == b.device(), #a " and " #b " are not on the same device!")

template <typename GridT>
void makeBinding(pybind11::module_& m) {
    m.def("kernel_evaluation", [](std::shared_ptr<GridT> grid,
            const torch::Tensor& query, const torch::Tensor& queryKernel,
            const torch::Tensor& gridKernel, const torch::Tensor& gridAlpha,
            const torch::Tensor& gradKernelQuery, bool grad) {
        CHECK_DEVICE((*grid), query);
        CHECK_DEVICE((*grid), queryKernel);
        CHECK_DEVICE((*grid), gridKernel);
        CHECK_DEVICE((*grid), gridAlpha);
        CHECK_DEVICE((*grid), gradKernelQuery);
        TORCH_CHECK(gradKernelQuery.dim() == 3);
        TORCH_CHECK(gridAlpha.size(0) == grid->numVoxels(), "grid_alpha must have number of voxels");
        return KernelEvaluation<typename GridT::BufferT>::apply(
                grid->nanovdbGrid(), grid->primalTransform(), grad,
                query, queryKernel, gridKernel, gridAlpha, gradKernelQuery);
    }, py::arg("grid"), py::arg("query"), py::arg("query_kernel"), py::arg("grid_kernel"),
          py::arg("grid_alpha"), py::arg("grad_kernel_query"), py::arg("grad") = false);

    m.def("matrix_building", [](std::shared_ptr<GridT> gridI, std::shared_ptr<GridT> gridJ,
                                const torch::Tensor& ptsPos,
                                const torch::Tensor& ptsKernelI, const torch::Tensor& ptsKernelJ,
                                const torch::Tensor& iKernel, const torch::Tensor& jKernel,
                                const torch::Tensor& gradPtsKernelPosI, const torch::Tensor& gradPtsKernelPosJ,
                                const torch::Tensor& indexMap,
                                bool grad, int64_t numEntries) {
        CHECK_DEVICE((*gridI), (*gridJ));
        CHECK_DEVICE((*gridI), ptsPos);
        CHECK_DEVICE((*gridI), ptsKernelI);
        CHECK_DEVICE((*gridI), ptsKernelJ);
        CHECK_DEVICE((*gridI), iKernel);
        CHECK_DEVICE((*gridI), jKernel);
        CHECK_DEVICE((*gridI), gradPtsKernelPosI);
        CHECK_DEVICE((*gridI), gradPtsKernelPosJ);
        CHECK_DEVICE((*gridI), indexMap);
        TORCH_CHECK(iKernel.size(0) == gridI->numVoxels(), "iKernel does not have correct size!");
        TORCH_CHECK(jKernel.size(0) == gridJ->numVoxels(), "jKernel does not have correct size!");
        return MatrixBuilding<typename GridT::BufferT>::apply(
                gridI->nanovdbGrid(), gridJ->nanovdbGrid(), gridI->primalTransform(), gridJ->primalTransform(),
                ptsPos,
                ptsKernelI, ptsKernelJ,
                iKernel, jKernel,
                gradPtsKernelPosI, gradPtsKernelPosJ,
                indexMap, grad, numEntries);
    }, py::arg("grid_i"), py::arg("grid_j"), py::arg("pts_pos"),
          py::arg("pts_kernel_i"), py::arg("pts_kernel_j"),
          py::arg("i_kernel"), py::arg("j_kernel"),
          py::arg("grad_pts_kernel_pos_i"), py::arg("grad_pts_kernel_pos_j"),
          py::arg("index_map"), py::arg("grad"), py::arg("num_entries"));

    m.def("k_building", [](std::shared_ptr<GridT> grid,
                           const torch::Tensor& kernel,
                           const torch::Tensor& indexMap,
                           int64_t numEntries) {
              CHECK_DEVICE((*grid), kernel);
              CHECK_DEVICE((*grid), indexMap);
              TORCH_CHECK(kernel.size(0) == grid->numVoxels(), "Kernel does not have correct size!");
              return KBuilding<typename GridT::BufferT>::apply(
                      grid->nanovdbGrid(), grid->primalTransform(), kernel, indexMap, numEntries);
          }, py::arg("grid"), py::arg("kernel"), py::arg("index_map"), py::arg("num_entries"));

    m.def("rhs_evaluation", [](std::shared_ptr<GridT> grid,
                               const torch::Tensor& pts, const torch::Tensor& ptsKernel,
                               const torch::Tensor& gridKernel, const torch::Tensor& gradKernelPts,
                               const torch::Tensor& ptsData) {
              CHECK_DEVICE((*grid), pts);
              CHECK_DEVICE((*grid), ptsKernel);
              CHECK_DEVICE((*grid), gridKernel);
              CHECK_DEVICE((*grid), gradKernelPts);
              TORCH_CHECK(gradKernelPts.dim() == 3);
              TORCH_CHECK(gridKernel.size(0) == grid->numVoxels(), "grid_kernel must have number of voxels");
              return RhsEvaluation<typename GridT::BufferT>::apply(
                      grid->nanovdbGrid(), grid->primalTransform(),
                      pts, ptsKernel, gridKernel, gradKernelPts, ptsData);
          }, py::arg("grid"), py::arg("pts"), py::arg("pts_kernel"), py::arg("grid_kernel"),
          py::arg("grad_kernel_pts"), py::arg("pts_data"));

    m.def("qg_building", [](std::shared_ptr<GridT> grid,
                               const torch::Tensor& pts, const torch::Tensor& ptsKernel,
                               const torch::Tensor& gridKernel, const torch::Tensor& gradKernelPts,
                               bool grad) {
              CHECK_DEVICE((*grid), pts);
              CHECK_DEVICE((*grid), ptsKernel);
              CHECK_DEVICE((*grid), gridKernel);
              CHECK_DEVICE((*grid), gradKernelPts);
              TORCH_CHECK(gradKernelPts.dim() == 3);
              TORCH_CHECK(gridKernel.size(0) == grid->numVoxels(), "grid_kernel must have number of voxels");
              return QgBuilding<typename GridT::BufferT>::apply(
                      grid->nanovdbGrid(), grid->primalTransform(),
                      pts, ptsKernel, gridKernel, gradKernelPts, grad);
          }, py::arg("grid"), py::arg("pts"), py::arg("pts_kernel"), py::arg("grid_kernel"),
          py::arg("grad_kernel_pts"), py::arg("grad"));

    m.def("csr_matrix_multiplication", [](
            std::shared_ptr<GridT> grid_i, std::shared_ptr<GridT> grid_j,
            const torch::Tensor& coordsI, const torch::Tensor& coordsJ,
            const torch::Tensor& iValue, const torch::Tensor& jValue,
            const torch::Tensor& iRowPtr, const torch::Tensor& jRowPtr,
            const torch::Tensor& iColInds, const torch::Tensor& jColInds,
            const torch::Tensor& indexMap, int64_t numEntries) {
        CHECK_DEVICE((*grid_j), (*grid_j));
        TORCH_CHECK(coordsI.size(0) == grid_i->numVoxels());
        TORCH_CHECK(coordsJ.size(0) == grid_j->numVoxels());
        TORCH_CHECK(iValue.size(0) == iColInds.size(0));
        TORCH_CHECK(jValue.size(0) == jColInds.size(0));
        return CsrMatrixMultiplication::apply(
                grid_i->primalTransform(),
                grid_j->primalTransform(),
                coordsI, coordsJ, iValue, jValue, iRowPtr, jRowPtr,
                iColInds, jColInds, indexMap, numEntries);
    });

    m.def("build_coo_indexer", [](std::shared_ptr<GridT> grid_i, std::shared_ptr<GridT> grid_j) -> torch::Tensor {
        CHECK_DEVICE((*grid_j), (*grid_j));
        return buildCOOIndexer<typename GridT::BufferT>(*grid_i, *grid_j);
    });
}

void pybind_kernel_eval(py::module& m) {
    makeBinding<SparseFeatureIndexGrid<nanovdb::HostBuffer>>(m);
    makeBinding<SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>>(m);
}
