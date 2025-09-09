#pragma once

#include "PytorchDeviceBuffer.h"
#include "PytorchTensorReadAccessor.h"
#include "CudaDeviceUtils.h"
#include "IndexGridBuilders.h"
#include "VoxelCoordTransform.h"
#include "ActiveVoxelIterator.h"
#include "TrilinearInterpolationIterator.h"
#include "Printing.h"


template <typename AccT>
inline int64_t voxelIndex(const AccT& acc, const nanovdb::Coord ijk) {
    return acc.getValue(ijk) - static_cast<int64_t>(1);
}


namespace fvdb {

inline std::vector<int64_t> getShapeButReplaceFirstDim(int64_t s0, const torch::Tensor& sRest) {
    std::vector<int64_t> outSize(sRest.dim());
    outSize[0] = s0;
    for (size_t i = 1; i < outSize.size(); i += 1) {
        outSize[i] = sRest.size(i);
    }
    return outSize;
}

} // namespace fvdb