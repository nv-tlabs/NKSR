#pragma once

#include "platform.h"
#include <nanovdb/NanoVDB.h>

template <int N, typename ScalarT>
struct NNIterator {
    NNIterator() = delete;

    // Public types
    using value_type = nanovdb::Coord;

    _CPU_AND_GPU_CODE_ explicit NNIterator(const nanovdb::Vec3<ScalarT>& p, int init = 0)
    : NNIterator(p.round(), init) {}

    _CPU_AND_GPU_CODE_ explicit NNIterator(const nanovdb::Coord& p, int init = 0) {
        static_assert(N % 2 == 1, "N should be odd.");
        mCount = 0;
        mCenter = p;
        mCoord = DeltaFromCount(init) + mCenter;
    }

    _CPU_AND_GPU_CODE_ static int32_t CountFromDelta(const nanovdb::Coord& delta) {
        int32_t dx = delta.x() + N / 2;
        int32_t dy = delta.y() + N / 2;
        int32_t dz = delta.z() + N / 2;
        return dx * N * N + dy * N + dz;
    }

    _CPU_AND_GPU_CODE_ static nanovdb::Coord DeltaFromCount(int32_t count) {
        const int32_t dz = count % N - N / 2;
        const int32_t dy = (count / N) % N - N / 2;
        const int32_t dx = count / (N * N) - N / 2;
        return {dx, dy, dz};
    }

    _CPU_AND_GPU_CODE_
    inline const NNIterator& operator++() {
        mCount += 1;
        if (!isValid()) {
            return *this;
        }
        mCoord = DeltaFromCount(mCount) + mCenter;
        return *this;
    }

    _CPU_AND_GPU_CODE_
    NNIterator operator++(int) {
        NNIterator tmp = *this; ++(*this); return tmp;
    }

    // Dereferencable.
    _CPU_AND_GPU_CODE_
    inline constexpr const value_type& operator*() const {
        return mCoord;
    }

    _CPU_AND_GPU_CODE_
    inline constexpr const value_type* operator->() const {
        return (const value_type*) &mCoord;
    }

    // Equality / inequality.
    _CPU_AND_GPU_CODE_
    inline constexpr bool operator==(const NNIterator& rhs) const {
        return mCenter == rhs.mCenter && mCount == rhs.mCount;
    }

    _CPU_AND_GPU_CODE_
    inline constexpr bool operator!=(const NNIterator& rhs) const {
        return !(*this == rhs);
    }

    _CPU_AND_GPU_CODE_
    inline constexpr bool isValid() {
        return mCount < total();
    }

    _CPU_AND_GPU_CODE_
    static inline constexpr int total() {
        return N * N * N;
    }

    _CPU_AND_GPU_CODE_
    inline int getCount() const {
        return mCount;
    }

private:
    int32_t mCount = 0;
    nanovdb::Coord mCoord;      // Current integer coordinates
    nanovdb::Coord mCenter;      // Center integer coordinates
};


struct OctChildrenIterator {
    OctChildrenIterator() = delete;

    // Public types
    using value_type = nanovdb::Coord;

    _CPU_AND_GPU_CODE_ explicit OctChildrenIterator(const nanovdb::Coord& p) {
        mCount = 0;
        mCenter = (p >> 1) << 1;
        mCoord = DeltaFromCount(0) + mCenter;
    }

    _CPU_AND_GPU_CODE_ static nanovdb::Coord DeltaFromCount(int32_t count) {
        const int32_t dz = count % 2;
        const int32_t dy = (count / 2) % 2;
        const int32_t dx = count / 4;
        return {dx, dy, dz};
    }

    _CPU_AND_GPU_CODE_
    inline const OctChildrenIterator& operator++() {
        mCount += 1;
        if (!isValid()) {
            return *this;
        }
        mCoord = DeltaFromCount(mCount) + mCenter;
        return *this;
    }

    _CPU_AND_GPU_CODE_
    OctChildrenIterator operator++(int) {
        OctChildrenIterator tmp = *this; ++(*this); return tmp;
    }

    // Dereferencable.
    _CPU_AND_GPU_CODE_
    inline constexpr const value_type& operator*() const {
        return mCoord;
    }

    _CPU_AND_GPU_CODE_
    inline constexpr const value_type* operator->() const {
        return (const value_type*) &mCoord;
    }

    // Equality / inequality.
    _CPU_AND_GPU_CODE_
    inline bool operator==(const OctChildrenIterator& rhs) const {
        return mCenter == rhs.mCenter && mCount == rhs.mCount;
    }

    _CPU_AND_GPU_CODE_
    inline bool operator!=(const OctChildrenIterator& rhs) const {
        return !(*this == rhs);
    }

    _CPU_AND_GPU_CODE_
    inline constexpr bool isValid() {
        return mCount < total();
    }

    _CPU_AND_GPU_CODE_
    static inline constexpr int total() {
        return 8;
    }

    _CPU_AND_GPU_CODE_
    inline int getCount() const {
        return mCount;
    }

private:
    int32_t mCount = 0;
    nanovdb::Coord mCoord;      // Current integer coordinates
    nanovdb::Coord mCenter;      // Center integer coordinates
};


// From Face ID to its axis (fix_axis_small/big, fix_axis, iter_axis0, iter_axis1) and MC accumulate indices.
_CPU_AND_GPU_CONSTANT_ int faceAxisTable[6][4] = {
        {0, 0, 1, 2}, {1, 0, 1, 2}, {0, 1, 2, 0}, {1, 1, 2, 0}, {0, 2, 0, 1}, {1, 2, 0, 1}
};
_CPU_AND_GPU_CONSTANT_ int faceAccIndsTable[6][4] = {
        {1, 2, 5, 6}, {0, 3, 4, 7}, {2, 3, 6, 7}, {0, 1, 5, 4}, {4, 5, 6, 7}, {0, 1, 2, 3}
};

// From Edge ID to its axis and MC accumulate indices.
_CPU_AND_GPU_CONSTANT_ int edgeAxisTable[12][5] = {
        {0, 0, 1, 2, 0}, {0, 1, 1, 2, 0}, {1, 0, 1, 2, 0}, {1, 1, 1, 2, 0},
        {0, 0, 2, 0, 1}, {0, 1, 2, 0, 1}, {1, 0, 2, 0, 1}, {1, 1, 2, 0, 1},
        {0, 0, 0, 1, 2}, {0, 1, 0, 1, 2}, {1, 0, 0, 1, 2}, {1, 1, 0, 1, 2},
};
_CPU_AND_GPU_CONSTANT_ int edgeAccIndsTable[12][2] = {
        {6, 7}, {2, 3}, {4, 5}, {0, 1},
        {5, 6}, {4, 7}, {1, 2}, {0, 3},
        {2, 6}, {1, 5}, {3, 7}, {0, 4}
};

// From Corner ID to corresponding values
_CPU_AND_GPU_CONSTANT_ int cornerAxisTable[8][3] = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
};
_CPU_AND_GPU_CONSTANT_ int cornerAccIndsTable[8] = {
        6, 2, 5, 1, 7, 3, 4, 0
};


template <int Stride>
struct CubeFaceIterator {

    CubeFaceIterator() = delete;

    // Public types
    using value_type = nanovdb::Coord;

    _CPU_AND_GPU_CODE_ explicit CubeFaceIterator(const nanovdb::Coord& p) {
        static_assert(Stride && ((Stride & (Stride - 1)) == 0), "Stride should be power of 2.");
        mCount = 0;
        mBaseStrided = nanovdb::Coord(p.x() * Stride, p.y() * Stride, p.z() * Stride);
        updateCoords();
    }

    _CPU_AND_GPU_CODE_ inline void updateCoords() {
        if (isFace()) {
            int cubeIdx = mCount - numCorners() - numEdges();
            int faceIdx = cubeIdx / ((Stride - 1) * (Stride - 1));
            int faceInnerIdx = cubeIdx % ((Stride - 1) * (Stride - 1));
            mCoord[faceAxisTable[faceIdx][1]] = faceAxisTable[faceIdx][0] * Stride;
            mCoord[faceAxisTable[faceIdx][2]] = faceInnerIdx % (Stride - 1) + 1;
            mCoord[faceAxisTable[faceIdx][3]] = int(faceInnerIdx / (Stride - 1)) + 1;
            mCoord += mBaseStrided;
            mLocalIdx = faceIdx;
        } else if (isEdge()) {
            int cubeIdx = mCount - numCorners();
            int edgeIdx = cubeIdx / (Stride - 1);
            int edgeInnerIdx = cubeIdx % (Stride - 1);
            mCoord[edgeAxisTable[edgeIdx][2]] = edgeAxisTable[edgeIdx][0] * Stride;
            mCoord[edgeAxisTable[edgeIdx][3]] = edgeAxisTable[edgeIdx][1] * Stride;
            mCoord[edgeAxisTable[edgeIdx][4]] = edgeInnerIdx + 1;
            mCoord += mBaseStrided;
            mLocalIdx = edgeIdx;
        } else {
            mCoord[0] = cornerAxisTable[mCount][0] * Stride;
            mCoord[1] = cornerAxisTable[mCount][1] * Stride;
            mCoord[2] = cornerAxisTable[mCount][2] * Stride;
            mCoord += mBaseStrided;
            mLocalIdx = mCount;
        }
    }

    _CPU_AND_GPU_CODE_ inline constexpr int getAccCount() const {
        if (isCorner()) return 1;
        if (isEdge()) return 2;
        return 4;
    }

    _CPU_AND_GPU_CODE_ inline constexpr int getAccInds(const int idx) const {
        if (isCorner()) return cornerAccIndsTable[mLocalIdx];
        if (isEdge()) return edgeAccIndsTable[mLocalIdx][idx];
        return faceAccIndsTable[mLocalIdx][idx];
    }

    _CPU_AND_GPU_CODE_
    inline const CubeFaceIterator& operator++() {
        mCount += 1;
        if (!isValid()) {
            return *this;
        }
        updateCoords();
        return *this;
    }

    _CPU_AND_GPU_CODE_
    CubeFaceIterator operator++(int) {
        CubeFaceIterator tmp = *this; ++(*this); return tmp;
    }

    // Dereferencable.
    _CPU_AND_GPU_CODE_
    inline constexpr const value_type& operator*() const {
        return mCoord;
    }

    _CPU_AND_GPU_CODE_
    inline constexpr const value_type* operator->() const {
        return (const value_type*) &mCoord;
    }

    // Equality / inequality.
    _CPU_AND_GPU_CODE_
    inline constexpr bool operator==(const CubeFaceIterator& rhs) const {
        return mBaseStrided == rhs.mBaseStrided && mCount == rhs.mCount;
    }

    _CPU_AND_GPU_CODE_
    inline constexpr bool operator!=(const CubeFaceIterator& rhs) const {
        return !(*this == rhs);
    }

    // Iterator special attributes
    _CPU_AND_GPU_CODE_ inline constexpr int numCorners() const {
        return 8;
    }

    _CPU_AND_GPU_CODE_ inline constexpr int numEdges() const {
        return (Stride - 1) * 12;
    }

    _CPU_AND_GPU_CODE_ inline constexpr int numFaces() const {
        return (Stride - 1) * (Stride - 1) * 6;
    }

    _CPU_AND_GPU_CODE_ inline constexpr bool isValid() const {
        return mCount < numCorners() + numEdges() + numFaces();
    }

    _CPU_AND_GPU_CODE_ inline constexpr bool isCorner() const {
        return mCount < numCorners() && mCount >= 0;
    }

    _CPU_AND_GPU_CODE_ inline constexpr bool isEdge() const {
        return mCount < numCorners() + numEdges() && !isCorner();
    }

    _CPU_AND_GPU_CODE_ inline constexpr bool isFace() const {
        return mCount >= numCorners() + numEdges() && isValid();
    }

    _CPU_AND_GPU_CODE_
    inline int getCount() const {
        return mCount;
    }

private:
    int32_t mCount = 0;
    int32_t mLocalIdx = 0;
    nanovdb::Coord mCoord;      // Current integer coordinates
    nanovdb::Coord mBaseStrided;      // Center integer coordinates
};
