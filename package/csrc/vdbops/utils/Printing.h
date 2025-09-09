#pragma once

#include <iostream> // for std::ostream

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <torch/extension.h>


inline std::ostream& operator<<(std::ostream& os, const nanovdb::Ray<float>& r) {
    os << "eye=" << r.eye() << " dir=" << r.dir() << " 1/dir="<<r.invDir()
       << " t0=" << r.t0()  << " t1="  << r.t1();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const nanovdb::Ray<double>& r) {
    os << "eye=" << r.eye() << " dir=" << r.dir() << " 1/dir="<<r.invDir()
       << " t0=" << r.t0()  << " t1="  << r.t1();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const nanovdb::Vec3<float>& v) {
    os << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const nanovdb::Vec3<double>& v) {
    os << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const nanovdb::Coord& c) {
    os << "[" << c[0] << ", " << c[1] << ", " << c[2] << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, at::IntArrayRef c) {
    os << "[";
    for (size_t i = 0; i < c.size(); i += 1) {
        os << c[i];
        if (i < c.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}