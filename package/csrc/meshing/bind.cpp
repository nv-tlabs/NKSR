#include <torch/extension.h>
#include <pybind11/stl.h>
#include "meshing.h"
#include "grid_builder.h"

template <typename GridT>
void makeBinding(pybind11::module_& m) {
    m.def("primal_cube_graph", [](std::shared_ptr<GridT> primal, std::shared_ptr<GridT> dual) -> torch::Tensor {
        return primalCubeGraph<typename GridT::BufferT>(*primal, *dual);
    });
    m.def("dual_cube_graph", [](const std::vector<std::shared_ptr<GridT>>& primalGrids, std::shared_ptr<GridT> corner) -> torch::Tensor {
       return dualCubeGraph<typename GridT::BufferT>(primalGrids, *corner);
    });
    m.def("build_flattened_grid", [](const std::shared_ptr<GridT>& thisGrid,
                                     const torch::optional<std::shared_ptr<GridT>>& childGrid,
                                     bool conforming) -> std::shared_ptr<GridT> {
        return buildFlattenedGrid<typename GridT::BufferT>(thisGrid, childGrid, conforming);
    });
    m.def("build_joint_dual_grid", [](const std::vector<std::shared_ptr<GridT>>& primalGrids) -> std::shared_ptr<GridT> {
        return buildJointDualGrid<typename GridT::BufferT>(primalGrids);
    });
}


void pybind_meshing(py::module& m) {
    makeBinding<SparseFeatureIndexGrid<nanovdb::HostBuffer>>(m);
    makeBinding<SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>>(m);

    m.def("marching_cubes", [](const torch::Tensor &cubeCornerInds,
                               const torch::Tensor &cornerPos,
                               const torch::Tensor &cornerValue) {
        if (cubeCornerInds.device().is_cuda()) {
            return MarchingCubesCUDA(cubeCornerInds, cornerPos, cornerValue);
        } else {
            return MarchingCubesCPU(cubeCornerInds, cornerPos, cornerValue);
        }
    }, "Marching cubes on a prebuilt graph.");
}
