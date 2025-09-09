#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>


namespace fvdb {

struct VoxelCoordTransform {

    __hostdev__
    VoxelCoordTransform() {};

    __hostdev__
    VoxelCoordTransform(double scale, double translate) : mTransform(scale, translate) {}

    template <typename ScalarT>
    __hostdev__
    ScalarT apply(ScalarT x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return x * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>();
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> apply(const nanovdb::Vec3<ScalarT>& x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>(x[0] * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>(),
                                      x[1] * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>(),
                                      x[2] * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> applyGrad(const nanovdb::Vec3<ScalarT>& x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>(mTransform.scale<ScalarT>(),
                                      mTransform.scale<ScalarT>(),
                                      mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> applyGrad(ScalarT x, ScalarT y, ScalarT z) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>(mTransform.scale<ScalarT>(),
                                      mTransform.scale<ScalarT>(),
                                      mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    ScalarT applyInv(ScalarT x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return (x - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>();
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> applyInv(const nanovdb::Vec3<ScalarT>& x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>((x[0] - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>(),
                                      (x[1] - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>(),
                                      (x[2] - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> applyInvGrad(const nanovdb::Vec3<ScalarT>& x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>((ScalarT)(1.0) / mTransform.scale<ScalarT>(),
                                      (ScalarT)(1.0) / mTransform.scale<ScalarT>(),
                                      (ScalarT)(1.0) / mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> applyInvGrad(ScalarT x, ScalarT y, ScalarT z) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>((ScalarT)(1.0) / mTransform.scale<ScalarT>(),
                                      (ScalarT)(1.0) / mTransform.scale<ScalarT>(),
                                      (ScalarT)(1.0) / mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> apply(ScalarT x, ScalarT y, ScalarT z) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>(x * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>(),
                                      y * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>(),
                                      z * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>());
    }

    template <typename ScalarT, typename InVec3T>
    __hostdev__
    nanovdb::Vec3<ScalarT> apply(const InVec3T& x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>(x[0] * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>(),
                                      x[1] * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>(),
                                      x[2] * mTransform.scale<ScalarT>() + mTransform.translate<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Vec3<ScalarT> applyInv(ScalarT x, ScalarT y, ScalarT z) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>((x - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>(),
                                      (y - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>(),
                                      (z - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>());
    }

    template <typename ScalarT, typename InVec3T>
    __hostdev__
    nanovdb::Vec3<ScalarT> applyInv(const InVec3T& x) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Vec3<ScalarT>((x[0] - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>(),
                                      (x[1] - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>(),
                                      (x[2] - mTransform.translate<ScalarT>()) / mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Ray<ScalarT> applyToRay(nanovdb::Ray<ScalarT> ray) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Ray<ScalarT>(apply<ScalarT>(ray.eye()), ray.dir(),
                                     ray.t0() * mTransform.scale<ScalarT>(),
                                     ray.t1() * mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    nanovdb::Ray<ScalarT> applyToRay(ScalarT rayOx, ScalarT rayOy, ScalarT rayOz,
                            ScalarT rayDx, ScalarT rayDy, ScalarT rayDz,
                            ScalarT t0 = static_cast<ScalarT>(0),
                            ScalarT t1 = std::numeric_limits<ScalarT>::infinity()) const {
        static_assert(std::is_floating_point<ScalarT>::value);
        return nanovdb::Ray<ScalarT>(apply<ScalarT>(rayOx, rayOy, rayOz),
                                     nanovdb::Vec3<ScalarT>(rayDx, rayDy, rayDz),
                                     t0 * mTransform.scale<ScalarT>(), t1 * mTransform.scale<ScalarT>());
    }

    template <typename ScalarT>
    __hostdev__
    ScalarT scale() const {
        return mTransform.scale<ScalarT>();
    }

    template <typename ScalarT>
    __hostdev__
    ScalarT translate() const {
        return mTransform.translate<ScalarT>();
    }


private:

    struct Transform {
        __hostdev__
        Transform() {};

        __hostdev__
        Transform(double scale, double translate) :
            mScalef((float) scale), mTranslatef((float) translate),
            mScaled(scale), mTranslated(translate) {}

        float mScalef = static_cast<float>(1);
        float mTranslatef = static_cast<float>(0);
        double mScaled = static_cast<double>(1);
        double mTranslated = static_cast<double>(0);

        template <typename T>
        __hostdev__ inline T scale() const;

        template <typename T>
        __hostdev__ inline T translate() const;
    } mTransform;
};

template <>
__hostdev__ inline float VoxelCoordTransform::Transform::scale<float>() const {
    return mScalef;
}

template <>
__hostdev__ inline double VoxelCoordTransform::Transform::scale<double>() const {
    return mScaled;
}

template <>
__hostdev__ inline float VoxelCoordTransform::Transform::translate<float>() const {
    return mTranslatef;
}

template <>
__hostdev__ inline double VoxelCoordTransform::Transform::translate<double>() const {
    return mTranslated;
}

} // namespace fvdb