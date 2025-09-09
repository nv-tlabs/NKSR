#include <torch/extension.h>

static uint div_up(const uint a, const uint b) {
    return (a + b - 1) / b;
}

std::vector<torch::Tensor> nearestNeighbours(
        torch::Tensor xyz,
        int knn
);

torch::Tensor estimateNormalsKNN(
        torch::Tensor xyz,
        torch::Tensor knnDist,
        torch::Tensor knnIndices,
        float maxRadius
);
