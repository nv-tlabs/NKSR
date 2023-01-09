#ifndef FLANN_CUDA_KD_TREE_BUILDER_H_
#define FLANN_CUDA_KD_TREE_BUILDER_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include "cutil_math.h"
#include <cstdlib>
#include <map>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

namespace tinyflann {

    // Distance types.
    struct CudaL1;
    struct CudaL2;

    // Parameters
    struct KDTreeCuda3dIndexParams {
        int leaf_max_size = 64;
    };

    struct SearchParams
    {
        SearchParams(int checks_ = 32, float eps_ = 0.0, bool sorted_ = true ) :
                checks(checks_), eps(eps_), sorted(sorted_)
        {
            max_neighbors = -1;
            use_heap = true;
        }

        // how many leafs to visit when searching for neighbours (-1 for unlimited)
        int checks;
        // search for eps-approximate neighbours (default: 0)
        float eps;
        // only for radius search, require neighbours sorted by distance (default: true)
        bool sorted;
        // maximum number of neighbors radius search should return (-1 for unlimited)
        int max_neighbors;
        // use a heap to manage the result set (default: FLANN_Undefined)
        bool use_heap;
    };

    template <typename Distance>
    class KDTreeCuda3dIndex
    {
    public:
        int visited_leafs;
        KDTreeCuda3dIndex(const float* input_data, size_t input_count,
                          const KDTreeCuda3dIndexParams& params = KDTreeCuda3dIndexParams())
                : dataset_(input_data), leaf_count_(0), visited_leafs(0), node_count_(0), current_node_count_(0) {
            size_ = input_count;
            leaf_max_size_ = params.leaf_max_size;
            gpu_helper_=0;
        }

        /**
         * Standard destructor
         */
        ~KDTreeCuda3dIndex() {
            clearGpuBuffers();
        }

        /**
         * Builds the index
         */
        void buildIndex() {
            leaf_count_ = 0;
            node_count_ = 0;
            uploadTreeToGpu();
        }

        /**
         * queries: (N, p) flattened float cuda array where only first 3 elements are used.
         * n_query: N
         * n_query_stride: p
         * indices: (N, knn) int cuda array
         * dists: (N, knn) float cuda array
         */
        void knnSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, size_t knn, const SearchParams& params = SearchParams()) const;
        int radiusSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, float radius, const SearchParams& params = SearchParams()) const;

    private:

        void uploadTreeToGpu();
        void clearGpuBuffers();

    private:

        struct GpuHelper;
        GpuHelper* gpu_helper_;

        const float* dataset_;
        int leaf_max_size_;
        int leaf_count_;
        int node_count_;
        //! used by convertTreeToGpuFormat
        int current_node_count_;
        size_t size_;

    };   // class KDTreeCuda3dIndex


} // namespace all


#endif