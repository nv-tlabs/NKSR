#include "kdtree_cuda.cuh"
// #define THRUST_DEBUG 1
#include <cuda.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector_types.h>
#include "cutil_math.h"
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <vector_types.h>

#include <limits>

__device__ __forceinline__
float infinity()
{
    return __int_as_float(0x7f800000);
}

#ifndef INFINITY
#define INFINITY infinity()
#endif

namespace tinyflann
{
    namespace cuda {

        template <class T>
        __device__ __host__ void swap( T& x, T& y ) {
            T t=x; x=y; y=t;
        }

        template <class GreaterThan, class RandomAccessIterator, class RandomAccessIterator2>
        __host__ __device__ void
        sift_down( RandomAccessIterator key, RandomAccessIterator2 value, size_t begin, size_t length, GreaterThan c = GreaterThan() )
        {

            while( 2*begin+1 < length ) {
                size_t left = 2*begin+1;
                size_t right = 2*begin+2;
                size_t largest=begin;
                if((left < length)&& c(key[left], key[largest]) ) largest=left;

                if((right < length)&& c(key[right], key[largest]) ) largest=right;

                if( largest != begin ) {
                    cuda::swap( key[begin], key[largest] );
                    cuda::swap( value[begin], value[largest] );
                    begin=largest;
                }
                else return;
            }
        }

//! creates a max-heap in the array beginning at begin of length "length"
//! if c is a less-than comparator, it will create a min-heap
        template <class GreaterThan, class RandomAccessIterator, class RandomAccessIterator2>
        __host__ __device__ void
        make_heap( RandomAccessIterator key,  RandomAccessIterator2 value, size_t length, GreaterThan c = GreaterThan() )
        {
            int i=length/2-1;
            while( i>=0 ) {
                sift_down( key, value, i, length, c );
                i--;
            }
        }

        template< typename DistanceType >
        struct GreaterThan
        {
            __device__
            bool operator()(DistanceType a, DistanceType b)
            {
                return a>b;
            }
        };

//! result set for the 1nn search. Doesn't do any global memory accesses on its own,
        template<typename DistanceType>
        struct SingleResultSet {
            int bestIndex;
            DistanceType bestDist;
            const DistanceType epsError;

            __device__ __host__ SingleResultSet(DistanceType eps) : bestIndex(-1), bestDist(INFINITY), epsError(eps) {}

            __device__
            inline float
            worstDist() {
                return bestDist;
            }

            __device__
            inline void
            insert(int index, DistanceType dist) {
                if (dist <= bestDist) {
                    bestIndex = index;
                    bestDist = dist;
                }
            }

            DistanceType *resultDist;
            int *resultIndex;

            __device__
            inline void
            setResultLocation(DistanceType *dists, int *index, int thread, int stride) {
                resultDist = dists + thread * stride;
                resultIndex = index + thread * stride;
                if (stride != 1) {
                    for (int i = 1; i < stride; i++) {
                        resultDist[i] = INFINITY;
                        resultIndex[i] = -1;
                    }
                }
            }

            __device__
            inline void
            finish() {
                resultDist[0] = bestDist;
                resultIndex[0] = bestIndex;
            }
        };

        template<typename DistanceType, bool useHeap>
        struct KnnResultSet {
            int foundNeighbors;
            DistanceType largestHeapDist;
            int maxDistIndex;
            const int k;
            const bool sorted;
            const DistanceType epsError;


            __device__ __host__
            KnnResultSet(int knn, bool sortResults, DistanceType eps) : foundNeighbors(0), largestHeapDist(INFINITY),
                                                                        k(knn), sorted(sortResults), epsError(eps) {}

            __device__
            inline DistanceType
            worstDist() {
                return largestHeapDist;
            }

            __device__
            inline void
            insert(int index, DistanceType dist) {
                if (foundNeighbors < k) {
                    resultDist[foundNeighbors] = dist;
                    resultIndex[foundNeighbors] = index;
                    if (foundNeighbors == k - 1) {
                        if (useHeap) {
                            make_heap(resultDist, resultIndex, k, GreaterThan<DistanceType>());
                            largestHeapDist = resultDist[0];
                        } else {
                            findLargestDistIndex();
                        }

                    }
                    foundNeighbors++;
                } else if (dist < largestHeapDist) {
                    if (useHeap) {
                        resultDist[0] = dist;
                        resultIndex[0] = index;
                        sift_down(resultDist, resultIndex, 0, k, GreaterThan<DistanceType>());
                        largestHeapDist = resultDist[0];
                    } else {
                        resultDist[maxDistIndex] = dist;
                        resultIndex[maxDistIndex] = index;
                        findLargestDistIndex();
                    }

                }
            }

            __device__
            void
            findLargestDistIndex() {
                largestHeapDist = resultDist[0];
                maxDistIndex = 0;
                for (int i = 1; i < k; i++)
                    if (resultDist[i] > largestHeapDist) {
                        maxDistIndex = i;
                        largestHeapDist = resultDist[i];
                    }
            }

            float *resultDist;
            int *resultIndex;

            __device__ inline void setResultLocation(DistanceType *dists, int *index, int thread, int stride) {
                resultDist = dists + stride * thread;
                resultIndex = index + stride * thread;
                for (int i = 0; i < stride; i++) {
                    resultDist[i] = INFINITY;
                    resultIndex[i] = -1;
                }
            }

            __host__ __device__ inline void finish() {
                if (sorted) {
                    if (!useHeap) make_heap(resultDist, resultIndex, k, GreaterThan<DistanceType>());
                    for (int i = k - 1; i > 0; i--) {
                        cuda::swap(resultDist[0], resultDist[i]);
                        cuda::swap(resultIndex[0], resultIndex[i]);
                        sift_down(resultDist, resultIndex, 0, i, GreaterThan<DistanceType>());
                    }
                }
            }
        };

        template<typename DistanceType>
        struct CountingRadiusResultSet {
            int count_;
            DistanceType radius_sq_;
            int max_neighbors_;

            __device__ __host__
            CountingRadiusResultSet(DistanceType radius, int max_neighbors) : count_(0), radius_sq_(radius),
                                                                              max_neighbors_(max_neighbors) {}

            __device__
            inline DistanceType
            worstDist() {
                return radius_sq_;
            }

            __device__
            inline void
            insert(int index, float dist) {
                if (dist < radius_sq_) {
                    count_++;
                }
            }

            int *resultIndex;

            __device__
            inline void
            setResultLocation(DistanceType * /*dists*/, int *count, int thread, int stride) {
                resultIndex = count + thread * stride;
            }

            __device__
            inline void
            finish() {
                if ((max_neighbors_ <= 0) || (count_ <= max_neighbors_)) resultIndex[0] = count_;
                else resultIndex[0] = max_neighbors_;
            }
        };

// Difference to RadiusKnnResultSet: Works like KnnResultSet, doesn't pack the results densely (as the RadiusResultSet does)
        template<typename DistanceType, bool useHeap>
        struct KnnRadiusResultSet {
            int foundNeighbors;
            DistanceType largestHeapDist;
            int maxDistIndex;
            const int k;
            const bool sorted;
            const DistanceType epsError;
            const DistanceType radius_sq;


            __device__ __host__
            KnnRadiusResultSet(int knn, bool sortResults, DistanceType eps, DistanceType radius) : foundNeighbors(0),
                                                                                                   largestHeapDist(
                                                                                                           radius),
                                                                                                   k(knn),
                                                                                                   sorted(sortResults),
                                                                                                   epsError(eps),
                                                                                                   radius_sq(radius) {}

            __device__
            inline DistanceType
            worstDist() {
                return largestHeapDist;
            }

            __device__
            inline void
            insert(int index, DistanceType dist) {
                if (dist < largestHeapDist) {
                    if (foundNeighbors < k) {
                        resultDist[foundNeighbors] = dist;
                        resultIndex[foundNeighbors] = index;
                        if (foundNeighbors == k - 1) {
                            if (useHeap) {
                                make_heap(resultDist, resultIndex, k, GreaterThan<DistanceType>());
                                largestHeapDist = resultDist[0];
                            } else {
                                findLargestDistIndex();
                            }
                        }
                        foundNeighbors++;
                    } else { //if( dist < largestHeapDist )
                        if (useHeap) {
                            resultDist[0] = dist;
                            resultIndex[0] = index;
                            sift_down(resultDist, resultIndex, 0, k, GreaterThan<DistanceType>());
                            largestHeapDist = resultDist[0];
                        } else {
                            resultDist[maxDistIndex] = dist;
                            resultIndex[maxDistIndex] = index;
                            findLargestDistIndex();
                        }
                    }
                }
            }

            __device__
            void
            findLargestDistIndex() {
                largestHeapDist = resultDist[0];
                maxDistIndex = 0;
                for (int i = 1; i < k; i++)
                    if (resultDist[i] > largestHeapDist) {
                        maxDistIndex = i;
                        largestHeapDist = resultDist[i];
                    }
            }

            DistanceType *resultDist;
            int *resultIndex;

            __device__ inline void setResultLocation(DistanceType *dists, int *index, int thread, int stride) {
                resultDist = dists + stride * thread;
                resultIndex = index + stride * thread;
                for (int i = 0; i < stride; i++) {
                    resultDist[i] = INFINITY;
                    resultIndex[i] = -1;
                }
            }

            __device__ inline void finish() {
                if (sorted) {
                    if (!useHeap) make_heap(resultDist, resultIndex, k, GreaterThan<DistanceType>());
                    for (int i = k - 1; i > 0; i--) {
                        cuda::swap(resultDist[0], resultDist[i]);
                        cuda::swap(resultIndex[0], resultIndex[i]);
                        sift_down(resultDist, resultIndex, 0, i, GreaterThan<DistanceType>());
                    }
                }
            }
        };
    }
}


namespace tinyflann {

    namespace cuda {
        namespace kd_tree_builder_detail {
//! normal node: contains the split dimension and value
//! leaf node: left == index of first points, right==index of last point +1
            struct SplitInfo {
                union {
                    struct {
                        // begin of child nodes
                        int left;
                        // end of child nodes
                        int right;
                    };
                    struct {
                        int split_dim;
                        float split_val;
                    };
                };

            };

            struct IsEven {
                typedef int result_type;

                __device__
                int operator()(int i) {
                    return (i & 1) == 0;
                }
            };

            struct SecondElementIsEven {
                __host__ __device__
                bool operator()(const thrust::tuple<int, int> &i) {
                    return (thrust::get<1>(i) & 1) == 0;
                }
            };

//! just for convenience: access a float4 by an index in [0,1,2]
//! (casting it to a float* and accessing it by the index is way slower...)
            __host__ __device__
            float get_value_by_index(const float4 &f, int i) {
                switch (i) {
                    case 0:
                        return f.x;
                    case 1:
                        return f.y;
                    default:
                        return f.z;
                }

            }

//! mark a point as belonging to the left or right child of its current parent
//! called after parents are split
            struct MovePointsToChildNodes {
                MovePointsToChildNodes(int *child1, SplitInfo *splits, float *x, float *y, float *z, int *ox, int *oy,
                                       int *oz, int *lrx, int *lry, int *lrz)
                        : child1_(child1), splits_(splits), x_(x), y_(y), z_(z), ox_(ox), oy_(oy), oz_(oz), lrx_(lrx),
                          lry_(lry), lrz_(lrz) {}

                //  int dim;
                //  float threshold;
                int *child1_;
                SplitInfo *splits_;

                // coordinate values
                float *x_, *y_, *z_;
                // owner indices -> which node does the point belong to?
                int *ox_, *oy_, *oz_;
                // temp info: will be set to 1 of a point is moved to the right child node, 0 otherwise
                // (used later in the scan op to separate the points of the children into continuous ranges)
                int *lrx_, *lry_, *lrz_;

                __device__
                void operator()(const thrust::tuple<int, int, int, int> &data) {
                    int index = thrust::get<0>(data);
                    int owner = ox_[index]; // before a split, all points at the same position in the index array have the same owner
                    int point_ind1 = thrust::get<1>(data);
                    int point_ind2 = thrust::get<2>(data);
                    int point_ind3 = thrust::get<3>(data);
                    int leftChild = child1_[owner];
                    int split_dim;
                    float dim_val1, dim_val2, dim_val3;
                    SplitInfo split;
                    lrx_[index] = 0;
                    lry_[index] = 0;
                    lrz_[index] = 0;
                    // this element already belongs to a leaf node -> everything alright, no need to change anything
                    if (leftChild == -1) {
                        return;
                    }
                    // otherwise: load split data, and assign this index to the new owner
                    split = splits_[owner];
                    split_dim = split.split_dim;
                    switch (split_dim) {
                        case 0:
                            dim_val1 = x_[point_ind1];
                            dim_val2 = x_[point_ind2];
                            dim_val3 = x_[point_ind3];
                            break;
                        case 1:
                            dim_val1 = y_[point_ind1];
                            dim_val2 = y_[point_ind2];
                            dim_val3 = y_[point_ind3];
                            break;
                        default:
                            dim_val1 = z_[point_ind1];
                            dim_val2 = z_[point_ind2];
                            dim_val3 = z_[point_ind3];
                            break;

                    }


                    int r1 = leftChild + (dim_val1 > split.split_val);
                    ox_[index] = r1;
                    int r2 = leftChild + (dim_val2 > split.split_val);
                    oy_[index] = r2;
                    oz_[index] = leftChild + (dim_val3 > split.split_val);

                    lrx_[index] = (dim_val1 > split.split_val);
                    lry_[index] = (dim_val2 > split.split_val);
                    lrz_[index] = (dim_val3 > split.split_val);
                    //                      return thrust::make_tuple( r1, r2, leftChild+(dim_val > split.split_val) );
                }
            };

//! used to update the left/right pointers and aabb infos after the node splits
            struct SetLeftAndRightAndAABB {
                int maxPoints;
                int nElements;

                SplitInfo *nodes;
                int *counts;
                int *labels;
                float4 *aabbMin;
                float4 *aabbMax;
                const float *x, *y, *z;
                const int *ix, *iy, *iz;

                __host__ __device__
                void operator()(int i) {
                    int index = labels[i];
                    int right;
                    int left = counts[i];
                    nodes[index].left = left;
                    if (i < nElements - 1) {
                        right = counts[i + 1];
                    } else { // index==nNodes
                        right = maxPoints;
                    }
                    nodes[index].right = right;
                    aabbMin[index].x = x[ix[left]];
                    aabbMin[index].y = y[iy[left]];
                    aabbMin[index].z = z[iz[left]];
                    aabbMax[index].x = x[ix[right - 1]];
                    aabbMax[index].y = y[iy[right - 1]];
                    aabbMax[index].z = z[iz[right - 1]];
                }
            };


//! - decide whether a node has to be split
//! if yes:
//! - allocate child nodes
//! - set split axis as axis of maximum aabb length
            struct SplitNodes {
                int maxPointsPerNode;
                int *node_count;
                int *nodes_allocated;
                int *out_of_space;
                int *child1_;
                int *parent_;
                SplitInfo *splits;

                __device__
                void operator()(
                        thrust::tuple<int &, int &, SplitInfo &, float4 &, float4 &, int> node) // float4: aabbMin, aabbMax
                {
                    int &parent = thrust::get<0>(node);
                    int &child1 = thrust::get<1>(node);
                    SplitInfo &s = thrust::get<2>(node);
                    const float4 &aabbMin = thrust::get<3>(node);
                    const float4 &aabbMax = thrust::get<4>(node);
                    int my_index = thrust::get<5>(node);
                    bool split_node = false;
                    // first, each thread block counts the number of nodes that it needs to allocate...
                    __shared__ int block_nodes_to_allocate;
                    if (threadIdx.x == 0) block_nodes_to_allocate = 0;
                    __syncthreads();

                    // don't split if all points are equal
                    // (could lead to an infinite loop, and doesn't make any sense anyway)
                    bool all_points_in_node_are_equal =
                            aabbMin.x == aabbMax.x && aabbMin.y == aabbMax.y && aabbMin.z == aabbMax.z;

                    int offset_to_global = 0;

                    // maybe this could be replaced with a reduction...
                    if ((child1 == -1) && (s.right - s.left > maxPointsPerNode) &&
                        !all_points_in_node_are_equal) { // leaf node
                        split_node = true;
                        offset_to_global = atomicAdd(&block_nodes_to_allocate, 2);
                    }

                    __syncthreads();
                    __shared__ int block_left;
                    __shared__ bool enough_space;
                    // ... then the first thread tries to allocate this many nodes...
                    if (threadIdx.x == 0) {
                        block_left = atomicAdd(node_count, block_nodes_to_allocate);
                        enough_space = block_left + block_nodes_to_allocate < *nodes_allocated;
                        // if it doesn't succeed, no nodes will be created by this block
                        if (!enough_space) {
                            atomicAdd(node_count, -block_nodes_to_allocate);
                            *out_of_space = 1;
                        }
                    }

                    __syncthreads();
                    // this thread needs to split it's node && there was enough space for all the nodes
                    // in this block.
                    //(The whole "allocate-per-block-thing" is much faster than letting each element allocate
                    // its space on its own, because shared memory atomics are A LOT faster than
                    // global mem atomics!)
                    if (split_node && enough_space) {
                        int left = block_left + offset_to_global;

                        splits[left].left = s.left;
                        splits[left].right = s.right;
                        splits[left + 1].left = 0;
                        splits[left + 1].right = 0;

                        // split axis/position: middle of longest aabb extent
                        float4 aabbDim = aabbMax - aabbMin;
                        int maxDim = 0;
                        float maxDimLength = aabbDim.x;
                        float4 splitVal = (aabbMax + aabbMin);
                        splitVal *= 0.5f;
                        for (int i = 1; i <= 2; i++) {
                            float val = get_value_by_index(aabbDim, i);
                            if (val > maxDimLength) {
                                maxDim = i;
                                maxDimLength = val;
                            }
                        }
                        s.split_dim = maxDim;
                        s.split_val = get_value_by_index(splitVal, maxDim);

                        child1_[my_index] = left;
                        splits[my_index] = s;

                        parent_[left] = my_index;
                        parent_[left + 1] = my_index;
                        child1_[left] = -1;
                        child1_[left + 1] = -1;
                    }
                }
            };


//! computes the scatter target address for the split operation, see Sengupta,Harris,Zhang,Owen: Scan Primitives for GPU Computing
//! in my use case, this is about 2x as fast as thrust::partition
            struct set_addr3 {
                const int *val_, *f_;

                int npoints_;

                __device__
                int operator()(int id) {
                    int nf = f_[npoints_ - 1] + (val_[npoints_ - 1]);
                    int f = f_[id];
                    int t = id - f + nf;
                    return val_[id] ? f : t;
                }
            };

//! converts a float4 point (xyz) to a tuple of three float vals (used to separate the
//! float4 input buffer into three arrays in the beginning of the tree build)
            struct pointxyz_to_px_py_pz {
                __device__
                thrust::tuple<float, float, float> operator()(const float4 &val) {
                    return thrust::make_tuple(val.x, val.y, val.z);
                }
            };
        } // namespace kd_tree_builder_detail

    } // namespace cuda


    class CudaKdTreeBuilder {
    public:
        CudaKdTreeBuilder(const thrust::device_vector<float4> &points, int max_leaf_size)
                : /*out_of_space_(1,0),node_count_(1,1),*/ max_leaf_size_(max_leaf_size) {
            points_ = &points;
            int prealloc = points.size() / max_leaf_size_ * 16;
            allocation_info_.resize(3);
            allocation_info_[NodeCount] = 1;
            allocation_info_[NodesAllocated] = prealloc;
            allocation_info_[OutOfSpace] = 0;

            child1_ = new thrust::device_vector<int>(prealloc, -1);
            parent_ = new thrust::device_vector<int>(prealloc, -1);
            cuda::kd_tree_builder_detail::SplitInfo s;
            s.left = 0;
            s.right = 0;
            splits_ = new thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo>(prealloc, s);
            s.right = points.size();
            (*splits_)[0] = s;

            aabb_min_ = new thrust::device_vector<float4>(prealloc);
            aabb_max_ = new thrust::device_vector<float4>(prealloc);

            index_x_ = new thrust::device_vector<int>(points_->size());
            index_y_ = new thrust::device_vector<int>(points_->size());
            index_z_ = new thrust::device_vector<int>(points_->size());

            owners_x_ = new thrust::device_vector<int>(points_->size(), 0);
            owners_y_ = new thrust::device_vector<int>(points_->size(), 0);
            owners_z_ = new thrust::device_vector<int>(points_->size(), 0);

            leftright_x_ = new thrust::device_vector<int>(points_->size(), 0);
            leftright_y_ = new thrust::device_vector<int>(points_->size(), 0);
            leftright_z_ = new thrust::device_vector<int>(points_->size(), 0);

            tmp_index_ = new thrust::device_vector<int>(points_->size());
            tmp_owners_ = new thrust::device_vector<int>(points_->size());
            tmp_misc_ = new thrust::device_vector<int>(points_->size());

            points_x_ = new thrust::device_vector<float>(points_->size());
            points_y_ = new thrust::device_vector<float>(points_->size());
            points_z_ = new thrust::device_vector<float>(points_->size());
            delete_node_info_ = false;
        }

        ~CudaKdTreeBuilder() {
            if (delete_node_info_) {
                delete child1_;
                delete parent_;
                delete splits_;
                delete aabb_min_;
                delete aabb_max_;
                delete index_x_;
            }

            delete index_y_;
            delete index_z_;
            delete owners_x_;
            delete owners_y_;
            delete owners_z_;
            delete points_x_;
            delete points_y_;
            delete points_z_;
            delete leftright_x_;
            delete leftright_y_;
            delete leftright_z_;
            delete tmp_index_;
            delete tmp_owners_;
            delete tmp_misc_;
        }

        //! build the tree
        //! general idea:
        //! - build sorted lists of the points in x y and z order (to be able to compute tight AABBs in O(1) )
        //! - while( nodes to split exist )
        //!    - split non-child nodes along longest axis if the number of points is > max_points_per_node
        //!    - for each point: determine whether it is in a node that was split. If yes, mark it as belonging to the left or right child node of its current parent node
        //!    - reorder the points so that the points of a single node are continuous in the node array
        //!    - update the left/right pointers and AABBs of all nodes
        void buildTree() {
            //              std::cout<<"buildTree()"<<std::endl;
            //              sleep(1);
            //              Util::Timer stepTimer;
            thrust::transform(points_->begin(), points_->end(), thrust::make_zip_iterator(
                    thrust::make_tuple(points_x_->begin(), points_y_->begin(), points_z_->begin())),
                              cuda::kd_tree_builder_detail::pointxyz_to_px_py_pz());

            thrust::counting_iterator<int> it(0);
            thrust::copy(it, it + points_->size(), index_x_->begin());

            thrust::copy(index_x_->begin(), index_x_->end(), index_y_->begin());
            thrust::copy(index_x_->begin(), index_x_->end(), index_z_->begin());

            thrust::device_vector<float> tmpv(points_->size());

            // create sorted index list -> can be used to compute AABBs in O(1)
            thrust::copy(points_x_->begin(), points_x_->end(), tmpv.begin());

            thrust::sort_by_key(tmpv.begin(), tmpv.end(), index_x_->begin());

            thrust::copy(points_y_->begin(), points_y_->end(), tmpv.begin());
            thrust::sort_by_key(tmpv.begin(), tmpv.end(), index_y_->begin());
            thrust::copy(points_z_->begin(), points_z_->end(), tmpv.begin());
            thrust::sort_by_key(tmpv.begin(), tmpv.end(), index_z_->begin());


            (*aabb_min_)[0] = make_float4((*points_x_)[(*index_x_)[0]], (*points_y_)[(*index_y_)[0]],
                                          (*points_z_)[(*index_z_)[0]], 0);

            (*aabb_max_)[0] = make_float4((*points_x_)[(*index_x_)[points_->size() - 1]],
                                          (*points_y_)[(*index_y_)[points_->size() - 1]],
                                          (*points_z_)[(*index_z_)[points_->size() - 1]], 0);
#ifdef PRINT_DEBUG_TIMING
            cudaDeviceSynchronize();
        std::cout<<" initial stuff:"<<stepTimer.elapsed()<<std::endl;
        stepTimer.restart();
#endif
            int last_node_count = 0;
            for (int i = 0;; i++) {
                cuda::kd_tree_builder_detail::SplitNodes sn;

                sn.maxPointsPerNode = max_leaf_size_;
                sn.node_count = thrust::raw_pointer_cast(&allocation_info_[NodeCount]);
                sn.nodes_allocated = thrust::raw_pointer_cast(&allocation_info_[NodesAllocated]);
                sn.out_of_space = thrust::raw_pointer_cast(&allocation_info_[OutOfSpace]);
                sn.child1_ = thrust::raw_pointer_cast(&(*child1_)[0]);
                sn.parent_ = thrust::raw_pointer_cast(&(*parent_)[0]);
                sn.splits = thrust::raw_pointer_cast(&(*splits_)[0]);

                thrust::counting_iterator<int> cit(0);
                thrust::for_each(thrust::make_zip_iterator(
                        thrust::make_tuple(parent_->begin(), child1_->begin(), splits_->begin(), aabb_min_->begin(),
                                           aabb_max_->begin(), cit)),
                                 thrust::make_zip_iterator(thrust::make_tuple(parent_->begin() + last_node_count,
                                                                              child1_->begin() + last_node_count,
                                                                              splits_->begin() + last_node_count,
                                                                              aabb_min_->begin() + last_node_count,
                                                                              aabb_max_->begin() + last_node_count,
                                                                              cit + last_node_count)),
                                 sn);
                // copy allocation info to host
                thrust::host_vector<int> alloc_info = allocation_info_;

                if (last_node_count == alloc_info[NodeCount]) { // no more nodes were split -> done
                    break;
                }
                last_node_count = alloc_info[NodeCount];

                // a node was un-splittable due to a lack of space
                if (alloc_info[OutOfSpace] == 1) {
                    resize_node_vectors(alloc_info[NodesAllocated] * 2);
                    alloc_info[OutOfSpace] = 0;
                    alloc_info[NodesAllocated] *= 2;
                    allocation_info_ = alloc_info;
                }
#ifdef PRINT_DEBUG_TIMING
                cudaDeviceSynchronize();
            std::cout<<" node split:"<<stepTimer.elapsed()<<std::endl;
            stepTimer.restart();
#endif

                // foreach point: point was in node that was split?move it to child (leaf) node : do nothing
                cuda::kd_tree_builder_detail::MovePointsToChildNodes sno(thrust::raw_pointer_cast(&(*child1_)[0]),
                                                                         thrust::raw_pointer_cast(&(*splits_)[0]),
                                                                         thrust::raw_pointer_cast(&(*points_x_)[0]),
                                                                         thrust::raw_pointer_cast(&(*points_y_)[0]),
                                                                         thrust::raw_pointer_cast(&(*points_z_)[0]),
                                                                         thrust::raw_pointer_cast(&(*owners_x_)[0]),
                                                                         thrust::raw_pointer_cast(&(*owners_y_)[0]),
                                                                         thrust::raw_pointer_cast(&(*owners_z_)[0]),
                                                                         thrust::raw_pointer_cast(&(*leftright_x_)[0]),
                                                                         thrust::raw_pointer_cast(&(*leftright_y_)[0]),
                                                                         thrust::raw_pointer_cast(&(*leftright_z_)[0])
                );
                thrust::counting_iterator<int> ci0(0);
                thrust::for_each(thrust::make_zip_iterator(
                        thrust::make_tuple(ci0, index_x_->begin(), index_y_->begin(), index_z_->begin())),
                                 thrust::make_zip_iterator(
                                         thrust::make_tuple(ci0 + points_->size(), index_x_->end(), index_y_->end(),
                                                            index_z_->end())), sno);

#ifdef PRINT_DEBUG_TIMING
                cudaDeviceSynchronize();
            std::cout<<" set new owners:"<<stepTimer.elapsed()<<std::endl;
            stepTimer.restart();
#endif

                // move points around so that each leaf node's points are continuous
                separate_left_and_right_children(*index_x_, *owners_x_, *tmp_index_, *tmp_owners_, *leftright_x_);
                std::swap(tmp_index_, index_x_);
                std::swap(tmp_owners_, owners_x_);
                separate_left_and_right_children(*index_y_, *owners_y_, *tmp_index_, *tmp_owners_, *leftright_y_,
                                                 false);
                std::swap(tmp_index_, index_y_);
                separate_left_and_right_children(*index_z_, *owners_z_, *tmp_index_, *tmp_owners_, *leftright_z_,
                                                 false);
                std::swap(tmp_index_, index_z_);

#ifdef PRINT_DEBUG_TIMING
                cudaDeviceSynchronize();
            std::cout<<" split:"<<stepTimer.elapsed()<<std::endl;
            stepTimer.restart();
#endif
                // calculate new AABB etc
                update_leftright_and_aabb(*points_x_, *points_y_, *points_z_, *index_x_, *index_y_, *index_z_,
                                          *owners_x_, *splits_, *aabb_min_, *aabb_max_);
#ifdef PRINT_DEBUG_TIMING
                cudaDeviceSynchronize();
            std::cout<<" update_leftright_and_aabb:"<<stepTimer.elapsed()<<std::endl;
            stepTimer.restart();
            print_vector(node_count_);
#endif

            }
        }

        template<class Distance>
        friend
        class KDTreeCuda3dIndex;

    protected:


        //! takes the partitioned nodes, and sets the left-/right info of leaf nodes, as well as the AABBs
        void
        update_leftright_and_aabb(const thrust::device_vector<float> &x, const thrust::device_vector<float> &y,
                                  const thrust::device_vector<float> &z,
                                  const thrust::device_vector<int> &ix, const thrust::device_vector<int> &iy,
                                  const thrust::device_vector<int> &iz,
                                  const thrust::device_vector<int> &owners,
                                  thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo> &splits,
                                  thrust::device_vector<float4> &aabbMin, thrust::device_vector<float4> &aabbMax) {
            thrust::device_vector<int> *labelsUnique = tmp_owners_;
            thrust::device_vector<int> *countsUnique = tmp_index_;
            // assume: points of each node are continuous in the array

            // find which nodes are here, and where each node's points begin and end
            int unique_labels =
                    thrust::unique_by_key_copy(owners.begin(), owners.end(), thrust::counting_iterator<int>(0),
                                               labelsUnique->begin(), countsUnique->begin()).first -
                    labelsUnique->begin();

            // update the info
            cuda::kd_tree_builder_detail::SetLeftAndRightAndAABB s;
            s.maxPoints = x.size();
            s.nElements = unique_labels;
            s.nodes = thrust::raw_pointer_cast(&(splits[0]));
            s.counts = thrust::raw_pointer_cast(&((*countsUnique)[0]));
            s.labels = thrust::raw_pointer_cast(&((*labelsUnique)[0]));
            s.x = thrust::raw_pointer_cast(&x[0]);
            s.y = thrust::raw_pointer_cast(&y[0]);
            s.z = thrust::raw_pointer_cast(&z[0]);
            s.ix = thrust::raw_pointer_cast(&ix[0]);
            s.iy = thrust::raw_pointer_cast(&iy[0]);
            s.iz = thrust::raw_pointer_cast(&iz[0]);
            s.aabbMin = thrust::raw_pointer_cast(&aabbMin[0]);
            s.aabbMax = thrust::raw_pointer_cast(&aabbMax[0]);

            thrust::counting_iterator<int> it(0);
            thrust::for_each(it, it + unique_labels, s);
        }

        //! Separates the left and right children of each node into continuous parts of the array.
        //! More specifically, it seperates children with even and odd node indices because nodes are always
        //! allocated in pairs -> child1==child2+1 -> child1 even and child2 odd, or vice-versa.
        //! Since the split operation is stable, this results in continuous partitions
        //! for all the single nodes.
        //! (basically the split primitive according to sengupta et al)
        //! about twice as fast as thrust::partition
        void separate_left_and_right_children(thrust::device_vector<int> &key_in, thrust::device_vector<int> &val_in,
                                              thrust::device_vector<int> &key_out, thrust::device_vector<int> &val_out,
                                              thrust::device_vector<int> &left_right_marks,
                                              bool scatter_val_out = true) {
            thrust::device_vector<int> *f_tmp = &val_out;
            thrust::device_vector<int> *addr_tmp = tmp_misc_;

            thrust::exclusive_scan( /*thrust::make_transform_iterator(*/
                    left_right_marks.begin() /*,cuda::kd_tree_builder_detail::IsEven*/
                    /*())*/, /*thrust::make_transform_iterator(*/
                    left_right_marks.end() /*,cuda::kd_tree_builder_detail::IsEven*/
                    /*())*/, f_tmp->begin());
            cuda::kd_tree_builder_detail::set_addr3 sa;
            sa.val_ = thrust::raw_pointer_cast(&left_right_marks[0]);
            sa.f_ = thrust::raw_pointer_cast(&(*f_tmp)[0]);
            sa.npoints_ = key_in.size();
            thrust::counting_iterator<int> it(0);
            thrust::transform(it, it + val_in.size(), addr_tmp->begin(), sa);

            thrust::scatter(key_in.begin(), key_in.end(), addr_tmp->begin(), key_out.begin());
            if (scatter_val_out) thrust::scatter(val_in.begin(), val_in.end(), addr_tmp->begin(), val_out.begin());
        }

        //! allocates additional space in all the node-related vectors.
        //! new_size elements will be added to all vectors.
        void resize_node_vectors(size_t new_size) {
            size_t add = new_size - child1_->size();
            child1_->insert(child1_->end(), add, -1);
            parent_->insert(parent_->end(), add, -1);
            cuda::kd_tree_builder_detail::SplitInfo s;
            s.left = 0;
            s.right = 0;
            splits_->insert(splits_->end(), add, s);
            float4 f;
            aabb_min_->insert(aabb_min_->end(), add, f);
            aabb_max_->insert(aabb_max_->end(), add, f);
        }


        const thrust::device_vector<float4> *points_;

        // tree data, those are stored per-node

        //! left child of each node. (right child==left child + 1, due to the alloc mechanism)
        //! child1_[node]==-1 if node is a leaf node
        thrust::device_vector<int> *child1_;
        //! parent node of each node
        thrust::device_vector<int> *parent_;
        //! split info (dim/value or left/right pointers)
        thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo> *splits_;
        //! min aabb value of each node
        thrust::device_vector<float4> *aabb_min_;
        //! max aabb value of each node
        thrust::device_vector<float4> *aabb_max_;

        enum AllocationInfo {
            NodeCount = 0,
            NodesAllocated = 1,
            OutOfSpace = 2
        };
        // those were put into a single vector of 3 elements so that only one mem transfer will be needed for all three of them
        //  thrust::device_vector<int> out_of_space_;
        //  thrust::device_vector<int> node_count_;
        //  thrust::device_vector<int> nodes_allocated_;
        thrust::device_vector<int> allocation_info_;

        int max_leaf_size_;

        // coordinate values of the points
        thrust::device_vector<float> *points_x_, *points_y_, *points_z_;
        // indices
        thrust::device_vector<int> *index_x_, *index_y_, *index_z_;
        // owner node
        thrust::device_vector<int> *owners_x_, *owners_y_, *owners_z_;
        // contains info about whether a point was partitioned to the left or right child after a split
        thrust::device_vector<int> *leftright_x_, *leftright_y_, *leftright_z_;
        thrust::device_vector<int> *tmp_index_, *tmp_owners_, *tmp_misc_;
        bool delete_node_info_;
    };


    namespace KdTreeCudaPrivate {
        template<typename GPUResultSet, typename Distance>
        __device__ void searchNeighbors(const cuda::kd_tree_builder_detail::SplitInfo *splits,
                             const int *child1,
                             const int *parent,
                             const float4 *aabbLow,
                             const float4 *aabbHigh, const float4 *elements, const float4 &q, GPUResultSet &result,
                             const Distance &distance = Distance()) {

            bool backtrack = false;
            int lastNode = -1;
            int current = 0;

            cuda::kd_tree_builder_detail::SplitInfo split;
            while (true) {
                if (current == -1) break;
                split = splits[current];

                float diff1;
                if (split.split_dim == 0) diff1 = q.x - split.split_val;
                else if (split.split_dim == 1) diff1 = q.y - split.split_val;
                else if (split.split_dim == 2) diff1 = q.z - split.split_val;

                // children are next to each other: leftChild+1 == rightChild
                int leftChild = child1[current];
                int bestChild = leftChild;
                int otherChild = leftChild;

                if ((diff1) < 0) {
                    otherChild++;
                } else {
                    bestChild++;
                }

                if (!backtrack) {
                    /* If this is a leaf node, then do check and return. */
                    if (leftChild == -1) {
                        for (int i = split.left; i < split.right; ++i) {
                            float dist = distance.dist(elements[i], q);
                            result.insert(i, dist);
                        }
                        backtrack = true;
                        lastNode = current;
                        current = parent[current];
                    } else { // go to closer child node
                        lastNode = current;
                        current = bestChild;
                    }
                } else { // continue moving back up the tree or visit far node?
                    // minimum possible distance between query point and a point inside the AABB
                    float mindistsq = 0;
                    float4 aabbMin = aabbLow[otherChild];
                    float4 aabbMax = aabbHigh[otherChild];

                    if (q.x < aabbMin.x) mindistsq += distance.axisDist(q.x, aabbMin.x);
                    else if (q.x > aabbMax.x) mindistsq += distance.axisDist(q.x, aabbMax.x);
                    if (q.y < aabbMin.y) mindistsq += distance.axisDist(q.y, aabbMin.y);
                    else if (q.y > aabbMax.y) mindistsq += distance.axisDist(q.y, aabbMax.y);
                    if (q.z < aabbMin.z) mindistsq += distance.axisDist(q.z, aabbMin.z);
                    else if (q.z > aabbMax.z) mindistsq += distance.axisDist(q.z, aabbMax.z);

                    //  the far node was NOT the last node (== not visited yet) AND there could be a closer point in it
                    if ((lastNode == bestChild) && (mindistsq <= result.worstDist())) {
                        lastNode = current;
                        current = otherChild;
                        backtrack = false;
                    } else {
                        lastNode = current;
                        current = parent[current];
                    }
                }

            }
        }


        template<typename GPUResultSet, typename Distance>
        __global__
        void nearestKernel(const cuda::kd_tree_builder_detail::SplitInfo *splits,
                           const int *child1,
                           const int *parent,
                           const float4 *aabbMin,
                           const float4 *aabbMax, const float4 *elements, const float *query, int stride,
                           int resultStride, int *resultIndex, float *resultDist, int querysize, GPUResultSet result,
                           Distance dist = Distance()) {
            typedef float DistanceType;
            typedef float ElementType;
            //                  typedef DistanceType float;
            size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

            if (tid >= querysize) return;

            float4 q = make_float4(query[tid * stride], query[tid * stride + 1], query[tid * stride + 2], 0);

            result.setResultLocation(resultDist, resultIndex, tid, resultStride);

            searchNeighbors(splits, child1, parent, aabbMin, aabbMax, elements, q, result, dist);

            result.finish();
        }

    }

//! contains some pointers that use cuda data types and that cannot be easily
//! forward-declared.
//! basically it contains all GPU buffers
    template<typename Distance>
    struct KDTreeCuda3dIndex<Distance>::GpuHelper {
        thrust::device_vector<cuda::kd_tree_builder_detail::SplitInfo> *gpu_splits_;
        thrust::device_vector<int> *gpu_parent_;
        thrust::device_vector<int> *gpu_child1_;
        thrust::device_vector<float4> *gpu_aabb_min_;
        thrust::device_vector<float4> *gpu_aabb_max_;
        thrust::device_vector<float4> *gpu_points_;
        thrust::device_vector<int> *gpu_vind_;

        GpuHelper() : gpu_splits_(0), gpu_parent_(0), gpu_child1_(0), gpu_aabb_min_(0), gpu_aabb_max_(0),
                      gpu_points_(0), gpu_vind_(0) {
        }

        ~GpuHelper() {
            delete gpu_splits_;
            gpu_splits_ = 0;
            delete gpu_parent_;
            gpu_parent_ = 0;
            delete gpu_child1_;
            gpu_child1_ = 0;
            delete gpu_aabb_max_;
            gpu_aabb_max_ = 0;
            delete gpu_aabb_min_;
            gpu_aabb_min_ = 0;
            delete gpu_vind_;
            gpu_vind_ = 0;

            delete gpu_points_;
            gpu_points_ = 0;
        }
    };

//! thrust transform functor
//! transforms indices in the internal data set back to the original indices
    struct map_indices {
        const int *v_;

        map_indices(const int *v) : v_(v) {}

        __host__ __device__
        float operator()(const int &i) const {
            if (i >= 0) return v_[i];
            else return i;
        }
    };

//! implementation of L2 distance for the CUDA kernels
    struct CudaL2 {

        static float __host__ __device__ axisDist(float a, float b) {
            return (a - b) * (a - b);
        }

        static float __host__ __device__ dist(float4 a, float4 b) {
            float4 diff = a - b;
            return dot(diff, diff);
        }
    };

//! implementation of L1 distance for the CUDA kernels
//! NOT TESTED!
    struct CudaL1 {

        static float __host__ __device__ axisDist(float a, float b) {
            return fabs(a - b);
        }

        static float __host__ __device__ dist(float4 a, float4 b) {
            return fabs(a.x - b.x) + fabs(a.y - b.y) + (a.z - b.z) + (a.w - b.w);
        }
    };


    template<typename Distance>
    void KDTreeCuda3dIndex<Distance>::knnSearch(const float* queries, size_t n_query, int istride, int* indices,
            float* dists, size_t knn, const SearchParams& params) const {
        int ostride = knn;

        int threadsPerBlock = 128;
        int blocksPerGrid = (n_query + threadsPerBlock - 1) / threadsPerBlock;

        float epsError = 1 + params.eps;
        bool sorted = params.sorted;
        bool use_heap = params.use_heap;

        Distance distance;
        if (knn == 1) {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(
                    &((*gpu_helper_->gpu_splits_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_points_)[0])),
                    queries,
                    istride,
                    ostride,
                    indices,
                    dists,
                    n_query, cuda::SingleResultSet<float>(epsError), distance);
        } else {
            if (use_heap) {
                KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(
                        &((*gpu_helper_->gpu_splits_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_points_)[0])),
                        queries,
                        istride,
                        ostride,
                        indices,
                        dists,
                        n_query, cuda::KnnResultSet<float, true>(knn, sorted, epsError), distance);
            } else {
                KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(
                        &((*gpu_helper_->gpu_splits_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                        thrust::raw_pointer_cast(&((*gpu_helper_->gpu_points_)[0])),
                        queries,
                        istride,
                        ostride,
                        indices,
                        dists,
                        n_query, cuda::KnnResultSet<float, false>(knn, sorted, epsError), distance
                );
            }
        }
        thrust::device_ptr<int> id = thrust::device_pointer_cast(indices);
        thrust::transform(id, id + knn * n_query, id,
                          map_indices(thrust::raw_pointer_cast(&((*gpu_helper_->gpu_vind_))[0])));
    }


//! used in the radius search to count the total number of neighbors
    struct isNotMinusOne {
        __host__ __device__
        bool operator()(int i) {
            return i != -1;
        }
    };

    template<typename Distance>
    int KDTreeCuda3dIndex<Distance>::radiusSearch(const float* queries, size_t n_query, int n_query_stride, int* indices,
            float* dists, float radius, const SearchParams& params) const {
        int max_neighbors = params.max_neighbors;
        assert(max_neighbors > 0);

        bool sorted = params.sorted;
        float epsError = 1 + params.eps;
        bool use_heap = params.use_heap;
        int istride = n_query_stride;
        int ostride = max_neighbors;

        Distance distance;
        int threadsPerBlock = 128;
        int blocksPerGrid = (n_query + threadsPerBlock - 1) / threadsPerBlock;

        if (use_heap) {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(
                    &((*gpu_helper_->gpu_splits_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_points_)[0])),
                    queries,
                    istride,
                    ostride,
                    indices,
                    dists,
                    n_query, cuda::KnnRadiusResultSet<float, true>(max_neighbors, sorted, epsError,
                                                                               radius), distance);
        } else {
            KdTreeCudaPrivate::nearestKernel<<<blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(
                    &((*gpu_helper_->gpu_splits_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_child1_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_parent_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_min_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_aabb_max_)[0])),
                    thrust::raw_pointer_cast(&((*gpu_helper_->gpu_points_)[0])),
                    queries,
                    istride,
                    ostride,
                    indices,
                    dists,
                    n_query, cuda::KnnRadiusResultSet<float, false>(max_neighbors, sorted, epsError,
                                                                                radius), distance);
        }

        thrust::device_ptr<int> id = thrust::device_pointer_cast(indices);
        thrust::transform(id, id + max_neighbors * n_query, id,
                          map_indices(thrust::raw_pointer_cast(&((*gpu_helper_->gpu_vind_))[0])));

        return thrust::count_if(id, id + max_neighbors * n_query, isNotMinusOne());
    }

    template<typename Distance>
    void KDTreeCuda3dIndex<Distance>::uploadTreeToGpu() {
        // just make sure that no weird alignment stuff is going on...
        // shouldn't, but who knows
        // (I would make this a (boost) static assertion, but so far flann seems to avoid boost
        //  assert( sizeof( KdTreeCudaPrivate::GpuNode)==sizeof( Node ) );
        delete gpu_helper_;
        gpu_helper_ = new GpuHelper;
        gpu_helper_->gpu_points_ = new thrust::device_vector<float4>(size_);
        thrust::device_vector<float4> tmp(size_);

        thrust::copy(thrust::device_pointer_cast((float4 *) dataset_),
                     thrust::device_pointer_cast((float4 *) (dataset_)) + size_, tmp.begin());

        CudaKdTreeBuilder builder(tmp, leaf_max_size_);

        builder.buildTree();

        gpu_helper_->gpu_splits_ = builder.splits_;
        gpu_helper_->gpu_aabb_min_ = builder.aabb_min_;
        gpu_helper_->gpu_aabb_max_ = builder.aabb_max_;
        gpu_helper_->gpu_child1_ = builder.child1_;
        gpu_helper_->gpu_parent_ = builder.parent_;
        gpu_helper_->gpu_vind_ = builder.index_x_;
        thrust::gather(builder.index_x_->begin(), builder.index_x_->end(), tmp.begin(), gpu_helper_->gpu_points_->begin());
    }


    template<typename Distance>
    void KDTreeCuda3dIndex<Distance>::clearGpuBuffers() {
        delete gpu_helper_;
        gpu_helper_ = 0;
    }

    // instantiations for L1 and L2.
    template void KDTreeCuda3dIndex<CudaL1>::uploadTreeToGpu();
    template void KDTreeCuda3dIndex<CudaL2>::uploadTreeToGpu();

    template void KDTreeCuda3dIndex<CudaL1>::clearGpuBuffers();
    template void KDTreeCuda3dIndex<CudaL2>::clearGpuBuffers();

    template struct KDTreeCuda3dIndex<CudaL1 >::GpuHelper;
    template struct KDTreeCuda3dIndex<CudaL2 >::GpuHelper;

    template void KDTreeCuda3dIndex<CudaL1>::knnSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, size_t knn, const SearchParams& params) const;
    template void KDTreeCuda3dIndex<CudaL2>::knnSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, size_t knn, const SearchParams& params) const;

    template int KDTreeCuda3dIndex<CudaL1>::radiusSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, float radius, const SearchParams& params) const;
    template int KDTreeCuda3dIndex<CudaL2>::radiusSearch(const float* queries, size_t n_query, int n_query_stride, int* indices, float* dists, float radius, const SearchParams& params) const;

};
