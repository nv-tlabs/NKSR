#include <torch/extension.h>
#include "pcproc.h"

void pybind_pcproc(py::module& m) {
    m.def("nearest_neighbours", &nearestNeighbours, "KNN computation (CUDA) distance is squared L2.",
          py::arg("xyz"), py::arg("knn"));
    m.def("estimate_normals_knn", &estimateNormalsKNN, "Estimate point cloud normals (CUDA)",
          py::arg("xyz"), py::arg("knn_dist"), py::arg("knn_indices"), py::arg("max_radius") = -1.0);
}
