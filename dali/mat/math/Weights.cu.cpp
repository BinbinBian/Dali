#include "dali/mat/math/Weights.h"
#include "mshadow/random.h"
#include "dali/mat/math/__MatMacros__.h"
#include "dali/mat/math/memory/TensorOps.h"
#include "dali/mat/math/memory/SynchronizedMemory.h"

template<typename R>
typename weights<R>::initializer_t weights<R>::empty() {
    return [](sync_t&){};
};

template<typename R>
typename weights<R>::initializer_t weights<R>::zeros() {
    return [](sync_t& matrix){
        DALI_FUNCTION_1_MUT(TensorOps::fill, matrix, (R)0.0);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::eye(R diag) {
    return [diag](sync_t& matrix) {
        DALI_FUNCTION_1_MUT(TensorOps::eye, matrix, diag);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R lower, R upper) {
    return [lower, upper](sync_t& matrix) {
        DALI_FUNCTION_1_MUT(TensorOps::random::uniform, matrix, lower, upper);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R bound) {
    return uniform(-bound/2.0, bound/2.0);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R mean, R std) {
    return [mean, std](sync_t& matrix) {
        DALI_FUNCTION_1_MUT(TensorOps::random::gaussian, matrix, mean, std);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R std) {
    return gaussian(0.0, std);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::svd(initializer_t preinitializer) {
    return [preinitializer](sync_t& matrix) {
        // assert(matrix.dims().size() == 2);
        // preinitializer(matrix);
        // auto svd = GET_MAT(matrix).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        // int n = matrix.dims(0);
        // int d = matrix.dims(1);
        // if (n < d) {
        //     GET_MAT(matrix) = svd.matrixV().block(0, 0, n, d);
        // } else {
        //     GET_MAT(matrix) = svd.matrixU().block(0, 0, n, d);
        // }
    };
}

template struct weights<float>;
template struct weights<double>;
