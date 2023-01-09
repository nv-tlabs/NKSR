#include <torch/extension.h>
using namespace pybind11::literals;

std::vector<torch::Tensor> sdf_from_points(const torch::Tensor& queries,
                              const torch::Tensor& ref_xyz,
                              const torch::Tensor& ref_normal,
                              int nb_points, float stdv, bool compute_grad, bool imls, int adaptive_knn);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdf_from_points", &sdf_from_points,
          "Compute sdf value from reference points.",
          "queries"_a, "ref_xyz"_a, "ref_normal"_a, "nb_points"_a, "stdv"_a,
          "compute_grad"_a=false, "imls"_a=false, "adaptive_knn"_a=0
    );
}
