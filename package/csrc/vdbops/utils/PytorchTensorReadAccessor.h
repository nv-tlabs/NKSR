#include <array>

#include <torch/extension.h>

namespace fvdb {

template <typename Scalar, unsigned Dims>
struct PytorchTensorReadAccessor;

template <typename Scalar>
struct PytorchTensorReadAccessor<Scalar, 1> {
    using ScalarT = Scalar;

    PytorchTensorReadAccessor() = delete;
    PytorchTensorReadAccessor(const at::TensorAccessor<Scalar, 1>& acc) :
        mAcc(acc) {}

    inline const ScalarT& get(uint32_t i) const { return mAcc[i]; }

    inline uint32_t rows() const {
        return mAcc.sizes()[0];
    }

    inline uint32_t cols() const {
        return 1;
    }

private:
    at::TensorAccessor<Scalar, 1> mAcc;
};

template <typename Scalar>
struct PytorchTensorReadAccessor<Scalar, 2> {
    using ScalarT = Scalar;

    PytorchTensorReadAccessor() = delete;
    PytorchTensorReadAccessor(const at::TensorAccessor<Scalar, 2>& acc) :
        mAcc(acc) {}

    inline const ScalarT& get(uint32_t i, uint32_t j) const { return mAcc[i][j]; }

    inline uint32_t rows() const {
        return mAcc.sizes()[0];
    }

    inline uint32_t cols() const {
        return mAcc.sizes()[1];
    }

private:
    at::TensorAccessor<Scalar, 2> mAcc;
};

template <typename Scalar>
struct PytorchTensorReadAccessor<Scalar, 3> {
    using ScalarT = Scalar;

    PytorchTensorReadAccessor() = delete;
    PytorchTensorReadAccessor(const at::TensorAccessor<Scalar, 3>& acc) :
        mAcc(acc) {}

    inline const ScalarT& get(uint32_t i, uint32_t j, uint32_t k) const { return mAcc[i][j][k]; }

    inline uint32_t batchSize() const {
        return mAcc.sizes()[0];
    }

    inline uint32_t rows() const {
        return mAcc.sizes()[1];
    }

    inline uint32_t cols() const {
        return mAcc.sizes()[2];
    }

private:
    at::TensorAccessor<Scalar, 3> mAcc;
};

} // namespace fvdb