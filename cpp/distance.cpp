// distance.cpp — the hot-path distance kernels, ported to C++ via pybind11.
//
// Why this file exists
// --------------------
// Profiling the pure-Python HNSW build showed the time was dominated not by the
// arithmetic but by *per-call overhead*: every graph hop calls a distance
// function on a single (d,) query against one row, and at d≈9 the numpy call
// machinery (bounds checks, temp allocation, dtype dispatch) costs far more
// than the ~9 multiply-adds it guards. Batching helps the brute-force/IVF fine
// scan, but the graph traversal is inherently one-point-at-a-time.
//
// These kernels remove that overhead: tight loops over contiguous float32 with
// no allocation on the scalar path. They are drop-in replacements for the
// numpy distance functions in src/vectors.py; that module imports this
// extension when it is built and transparently falls back to numpy otherwise.
//
// Build:  python cpp/build.py     (or see cpp/README.md)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstddef>

namespace py = pybind11;

// Squared L2 distance between a single query (d,) and one vector (d,).
// This is the scalar hot path hammered by HNSW graph traversal.
static double l2_sq_point(py::array_t<float, py::array::c_style | py::array::forcecast> query,
                          py::array_t<float, py::array::c_style | py::array::forcecast> vec) {
    const ssize_t d = query.shape(0);
    const float* q = query.data();
    const float* v = vec.data();
    double acc = 0.0;
    for (ssize_t i = 0; i < d; ++i) {
        const double diff = static_cast<double>(q[i]) - static_cast<double>(v[i]);
        acc += diff * diff;
    }
    return acc;
}

// Batched squared L2: query (d,) against every row of matrix (n, d) -> (n,).
// Used by the brute-force oracle and the IVF fine scan.
static py::array_t<float> l2_sq_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        py::array_t<float, py::array::c_style | py::array::forcecast> matrix) {
    const ssize_t n = matrix.shape(0);
    const ssize_t d = matrix.shape(1);
    const float* q = query.data();
    const float* m = matrix.data();

    auto out = py::array_t<float>(n);
    float* o = out.mutable_data();

    // Release the GIL: this is a pure-numeric loop touching no Python objects,
    // so other threads can run while we crunch.
    {
        py::gil_scoped_release release;
        for (ssize_t r = 0; r < n; ++r) {
            const float* row = m + r * d;
            double acc = 0.0;
            for (ssize_t i = 0; i < d; ++i) {
                const double diff = static_cast<double>(q[i]) - static_cast<double>(row[i]);
                acc += diff * diff;
            }
            o[r] = static_cast<float>(acc);
        }
    }
    return out;
}

// Batched Euclidean (sqrt of the above), matching src.vectors.l2_distance.
static py::array_t<float> l2_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        py::array_t<float, py::array::c_style | py::array::forcecast> matrix) {
    auto sq = l2_sq_batch(query, matrix);
    float* o = sq.mutable_data();
    const ssize_t n = sq.shape(0);
    {
        py::gil_scoped_release release;
        for (ssize_t r = 0; r < n; ++r) o[r] = std::sqrt(o[r]);
    }
    return sq;
}

PYBIND11_MODULE(vecsearch_native, m) {
    m.doc() = "Hand-written C++ distance kernels for vecsearch (pybind11).";
    m.def("l2_sq_point", &l2_sq_point,
          "Squared L2 distance between two (d,) float32 vectors.",
          py::arg("query"), py::arg("vec"));
    m.def("l2_sq_batch", &l2_sq_batch,
          "Squared L2 from a (d,) query to every row of an (n,d) matrix.",
          py::arg("query"), py::arg("matrix"));
    m.def("l2_batch", &l2_batch,
          "Euclidean distance from a (d,) query to every row of an (n,d) matrix.",
          py::arg("query"), py::arg("matrix"));
}
