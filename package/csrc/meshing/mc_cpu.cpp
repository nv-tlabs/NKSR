#include "meshing.h"
#include "mc_data.h"

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")
#define CHECK_IS_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be a long tensor")

std::vector<torch::Tensor> MarchingCubesCPU(
        const torch::Tensor &cubeCornerInds,    // (M, 8) long
        const torch::Tensor &cornerPos,         // (N, 3) float
        const torch::Tensor &cornerValue) {     // (N, )  float

    CHECK_CPU(cubeCornerInds); CHECK_IS_LONG(cubeCornerInds);
    CHECK_CPU(cornerPos); CHECK_IS_FLOAT(cornerPos);
    CHECK_CPU(cornerValue); CHECK_IS_FLOAT(cornerValue);

    int64_t numCubes = cubeCornerInds.size(0);
    int64_t grainSize = at::internal::GRAIN_SIZE;

    const auto& opt = torch::TensorOptions().device(cubeCornerInds.device());
    torch::Tensor vertexCounts = torch::empty(numCubes, opt.dtype(torch::kLong));

    // We need accessor even for CPU tensors because direct Tensor subscripting has extra huge overhead.
    const auto& cubeCornerIndsAccessor = cubeCornerInds.accessor<int64_t, 2>();
    const auto& cornerValueAccessor = cornerValue.accessor<float, 1>();
    const auto& cornerPosAccessor = cornerPos.accessor<float, 2>();
    auto vertexCountsAccessor = vertexCounts.accessor<int64_t, 1>();

    if (numCubes > 0) {
        at::parallel_for(0, numCubes, grainSize, [&](int64_t begin, int64_t end) {
            float sdfVals[8];
            for (size_t cubeIdx = begin; cubeIdx < std::min(end, numCubes); cubeIdx++) {
                for (int i = 0; i < 8; ++i) {
                    sdfVals[i] = cornerValueAccessor[cubeCornerIndsAccessor[cubeIdx][i]];
                }
                int cubeType = get_cube_type(sdfVals);
                vertexCountsAccessor[cubeIdx] = numVertsTable[cubeType];
            }
        });
    }

    torch::Tensor countCsum = torch::cumsum(vertexCounts, 0);
    int64_t nTriangles = 0;
    if (numCubes > 0) {
        nTriangles = countCsum[-1].item<int64_t>() / 3;
        countCsum = torch::roll(countCsum, torch::IntList(1));
        countCsum[0] = 0;
    }
    const auto& countCsumAccessor = countCsum.accessor<int64_t, 1>();

    // Generate triangles
    torch::Tensor triangles = torch::empty({nTriangles, 3, 3}, opt.dtype(torch::kFloat32));
    torch::Tensor vertIdx = torch::empty({nTriangles, 3, 2}, opt.dtype(torch::kLong));
    auto trianglesAccessor = triangles.accessor<float, 3>();
    auto vertIdxAccessor = vertIdx.accessor<int64_t, 3>();

    if (nTriangles > 0) {
        at::parallel_for(0, numCubes, grainSize, [&](int64_t begin, int64_t end) {
            float sdfVals[8];
            int64_t pointIds[8];
            nanovdb::Vec3f points[8];
            nanovdb::Vec4f vertList[12];

            for (int64_t cubeIdx = begin; cubeIdx < std::min(end, numCubes); cubeIdx++) {
                for (int i = 0; i < 8; ++i) {
                    pointIds[i] = cubeCornerIndsAccessor[cubeIdx][i];
                    sdfVals[i] = cornerValueAccessor[pointIds[i]];
                    for (int d = 0; d < 3; ++d) {
                        points[i][d] = cornerPosAccessor[pointIds[i]][d];
                    }
                }
                int cubeType = get_cube_type(sdfVals);
                int edgeConfig = edgeTable[cubeType];
                if (edgeConfig == 0)
                    continue;
                fill_vert_list(vertList, edgeConfig, points, sdfVals);

                // Write triangles to array.
                for (int i = 0; triangleTable[cubeType][i] != -1; i += 3) {
                    int64_t triangle_id = countCsumAccessor[cubeIdx] / 3 + i / 3;
                    for (int vi = 0; vi < 3; ++vi) {
                        int vlid = triangleTable[cubeType][i + vi];
                        for (int d = 0; d < 3; ++d) {
                            trianglesAccessor[triangle_id][vi][d] = vertList[vlid][d];
                        }
                        int64_t vid0 = pointIds[e2iTable[vlid][0]];
                        int64_t vid1 = pointIds[e2iTable[vlid][1]];
                        if (vid0 < vid1) {
                            std::swap(vid0, vid1);
                        }
                        vertIdxAccessor[triangle_id][vi][0] = vid0;
                        vertIdxAccessor[triangle_id][vi][1] = vid1;
                    }
                }
            }
        });
    }

    // Merge triangles by detecting the same vertex position.
    const auto &unqRet = torch::unique_dim(vertIdx.view({-1, 2}), 0, false, true);
    const torch::Tensor& unqVertIdx = std::get<0>(unqRet);
    const torch::Tensor& unqTriangles = std::get<1>(unqRet);

    torch::Tensor vertices = torch::zeros({unqVertIdx.size(0), 3}, opt.dtype(torch::kFloat32));
    vertices.index_put_({unqTriangles}, triangles.view({-1, 3}));

    return {vertices, unqTriangles.view({-1, 3}), unqVertIdx};

}
