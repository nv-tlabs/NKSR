#pragma once

#include <nanovdb/util/GridHandle.h>

#include <torch/extension.h>

#include "utils/Utils.h"


namespace fvdb {

template <typename Buffertype>
void dispatchDownsampleGridMaxPoolGrad(const nanovdb::GridHandle<Buffertype>& coarseGridHdl,
                                       const nanovdb::GridHandle<Buffertype>& fineGridHdl,
                                       const torch::Tensor& fineData,
                                       const torch::Tensor& coarseGradOut,
                                       torch::Tensor& outFineGradIn,
                                       unsigned poolingFactor,
                                       unsigned nThreadsX = 64,
                                       unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchDownsampleGridMaxPool(const nanovdb::GridHandle<BufferType>& fineGrid,
                                   const nanovdb::GridHandle<BufferType>& coarseGrid,
                                   torch::Tensor& fineData,
                                   torch::Tensor& outCoarseData,
                                   unsigned poolingFactor,
                                   unsigned nThreadsX = 64,
                                   unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchUpsampleGridNearest(const nanovdb::GridHandle<BufferType>& coarseGrid,
                                 const nanovdb::GridHandle<BufferType>& fineGrid,
                                 torch::Tensor& coarseData,
                                 torch::Tensor& outFineData,
                                 unsigned upsamplingFactor,
                                 unsigned nThreadsX = 128,
                                 unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchUpsampleGridNearestGrad(const nanovdb::GridHandle<BufferType>& fineGrid,
                                     const nanovdb::GridHandle<BufferType>& coarseGrid,
                                     torch::Tensor& fineData,
                                     torch::Tensor& outCoarseData,
                                     unsigned upsamplingFactor,
                                     unsigned nThreadsX = 128,
                                     unsigned nThreadsY = 4);

template <typename BufferType>
void dispatchIjkToIndex(const nanovdb::GridHandle<BufferType>& gridBuf,
                        const torch::Tensor& coords,
                        torch::Tensor& outIndex,
                        unsigned nThreadsX = 128);


template <typename BufferType>
void dispatchPointsInGrid(const nanovdb::GridHandle<BufferType>& gridBuf,
                          const VoxelCoordTransform& transform,
                          const torch::Tensor& points, torch::Tensor& outMask,
                          unsigned nThreadsX = 128);


template <typename BufferType>
void dispatchSampleGridTrilinear(const nanovdb::GridHandle<BufferType>& gridBuf,
                                 const torch::Tensor& points,
                                 const torch::Tensor& gridData,
                                 const VoxelCoordTransform& transform,
                                 torch::Tensor& outFeatures,
                                 unsigned nThreadsX = 128, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSampleGridTrilinearWithGrad(const nanovdb::GridHandle<BufferType>& gridBuf,
                                         const torch::Tensor& points,
                                         const torch::Tensor& gridData,
                                         const VoxelCoordTransform& transform,
                                         torch::Tensor& outFeatures,
                                         torch::Tensor& outGradFeatures,
                                         unsigned nThreadsX = 64, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSampleGridTrilinearWithGradGrad(const nanovdb::GridHandle<BufferType>& gridBuf,
                                             const torch::Tensor& points,
                                             const torch::Tensor& gradOutFeatures,
                                             const torch::Tensor& gradOutGradFeatures,
                                             const VoxelCoordTransform& transform,
                                             torch::Tensor& outGridData,
                                             unsigned nThreadsX = 64, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSplatIntoGridTrilinear(const nanovdb::GridHandle<BufferType>& gridBuf,
                                    const torch::Tensor& points,
                                    const torch::Tensor& gradOut,
                                    const VoxelCoordTransform& transform,
                                    torch::Tensor& outGrad,
                                    unsigned nthreadsX = 128, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSplatIntoGridTrilinearWithCounts(const nanovdb::GridHandle<BufferType>& gridBuf,
                                              const torch::Tensor& points,
                                              const torch::Tensor& pointPrimalData,
                                              const VoxelCoordTransform& transform,
                                              torch::Tensor& outPrimalData,
                                              torch::Tensor& outCounts,
                                              unsigned nThreadsX = 128, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSampleGridBezier(const nanovdb::GridHandle<BufferType>& gridBuf,
                              const torch::Tensor& points,
                              const torch::Tensor& gridData,
                              const VoxelCoordTransform& transform,
                              torch::Tensor& outFeatures,
                              unsigned nThreadsX = 128, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSampleGridBezierWithGrad(const nanovdb::GridHandle<BufferType>& gridBuf,
                                      const torch::Tensor& points,
                                      const torch::Tensor& gridData,
                                      const VoxelCoordTransform& transform,
                                      torch::Tensor& outFeatures,
                                      torch::Tensor& outGradFeatures,
                                      unsigned nThreadsX = 128, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSampleGridBezierWithGradGrad(const nanovdb::GridHandle<BufferType>& gridBuf,
                                          const torch::Tensor& points,
                                          const torch::Tensor& gradOutFeatures,
                                          const torch::Tensor& gradOutGradFeatures,
                                          const VoxelCoordTransform& transform,
                                          torch::Tensor& outGridData,
                                          unsigned nThreadsX = 128, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchSplatIntoGridBezier(const nanovdb::GridHandle<BufferType>& gridBuf,
                                 const torch::Tensor& points,
                                 const torch::Tensor& gradOut,
                                 const VoxelCoordTransform& transform,
                                 torch::Tensor& outGrad,
                                 unsigned nthreadsX = 128, unsigned nThreadsY = 4);


template <typename BufferType>
void dispatchActiveGridCoords(const nanovdb::GridHandle<BufferType>& gridBuf,
                              torch::Tensor& outGridCoords,
                              unsigned nThreadsX = 256);


template <typename BufferType>
void dispatchTransformPointsToGrid(const torch::Tensor& points,
                                   const VoxelCoordTransform& transform,
                                   torch::Tensor& outCoords,
                                   unsigned nThreadsX = 256);


template <typename BufferType>
void dispatchTransformPointsToGridGrad(const torch::Tensor& gradOut,
                                       const VoxelCoordTransform& transform,
                                       torch::Tensor& outGradIn,
                                       unsigned nThreadsX = 256);


template <typename BufferType>
void dispatchInvTransformPointsToGrid(const torch::Tensor& points,
                                      const VoxelCoordTransform& transform,
                                      torch::Tensor& outCoords,
                                      unsigned nThreadsX = 256);


template <typename BufferType>
void dispatchInvTransformPointsToGridGrad(const torch::Tensor& gradOut,
                                          const VoxelCoordTransform& transform,
                                          torch::Tensor& outGradIn,
                                          unsigned nThreadsX = 256);

} // namespace fvdb