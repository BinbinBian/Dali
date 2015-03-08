#include "Graph.h"

#define EPS 1e-9


using std::stringstream;
using std::vector;
using utils::Timer;

template<typename T>
Graph<T>::Graph (bool _needs_backprop) : needs_backprop(_needs_backprop) {}
template<typename T>
Graph<T>::Graph () : needs_backprop(true) {}

template<typename T>
void Graph<T>::backward () {
    Timer t("graph_backward");
    for (auto it = this->backprop.rbegin(); it != this->backprop.rend(); ++it)
        (*it)();
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul_broadcast(
    shared_mat matrix1,
    shared_mat matrix2) {
    if (matrix1->n != matrix2->n || matrix2->d != 1) {
        stringstream error_msg;
        error_msg << "Matrices " << *matrix1 << " and "
                                 << *matrix2
                  << " cannot be element multiplied with broadcast,"
                     " they do not have the same dimensions.";
        throw std::invalid_argument(error_msg.str());
    }
    auto out = std::make_shared<mat>(
        matrix1->n,
        matrix1->d,
        true);
    out->w = (matrix1->w.array().colwise() * matrix2->w.col(0).array()).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += ((out->dw).array().colwise() * (matrix2->w).col(0).array()).matrix();
            matrix2->dw.noalias() += ((matrix1->w).array() * (out->dw).array()).matrix().rowwise().sum();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul(
    shared_mat matrix1,
    shared_mat matrix2) {
    if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
        if (matrix1->d == 1) {
            return eltmul_broadcast(matrix2, matrix1);
        }
        return eltmul_broadcast(matrix1, matrix2);
    }
    if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
        throw std::invalid_argument("Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = std::make_shared<mat>(
        matrix1->n,
        matrix1->d,
        true);
    out->w = (matrix1->w.array() * matrix2->w.array()).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += ((matrix2->w).array() * (out->dw).array()).matrix();
            matrix2->dw.noalias() += ((matrix1->w).array() * (out->dw).array()).matrix();
        });
    return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul(
    shared_mat matrix,
    T alpha) {

    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);
    out->w = (matrix->w.array() * alpha).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix, alpha, out]() {
            matrix->dw.noalias() += (alpha * (out->dw).array()).matrix();
        });
    return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul_broadcast_rowwise(
    shared_mat matrix1,
    shared_mat row_vector) {
    if (matrix1->d != row_vector->d || row_vector->n != 1)
        throw std::invalid_argument("Matrices A and B^T cannot be element multiplied with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<mat>(
            matrix1->n,
            matrix1->d,
            true);
    out->w = (matrix1->w.array().rowwise() * row_vector->w.row(0).array()).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, row_vector, out]() {
            matrix1->dw.noalias() += ((out->dw).array().rowwise() * (row_vector->w).row(0).array()).matrix();
            row_vector->dw.noalias() += (((matrix1->w).array() * (out->dw).array()).matrix().colwise().sum()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::eltmul_rowwise(
    shared_mat matrix1,
    shared_mat matrix2) {

    if (matrix1->n != matrix2->d || matrix1->d != matrix2->n)
        throw std::invalid_argument("Matrices A and B^T cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = std::make_shared<mat>(
        matrix1->n,
        matrix1->d,
        true);
    out->w = (matrix1->w.array() * matrix2->w.transpose().array()).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += ((matrix2->w).transpose().array() * (out->dw).array()).matrix();
            matrix2->dw.noalias() += ((matrix1->w).array() * (out->dw).array()).matrix().transpose();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add(
        shared_mat matrix1,
        shared_mat matrix2) {
    if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
        if (matrix1->d == 1) {
            return add_broadcast(matrix2, matrix1);
        }
        return add_broadcast(matrix1, matrix2);
    }
    if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
        throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<T>>(
        matrix1->n,
        matrix1->d,
        true);
    out->w = matrix1->w + matrix2->w;
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw;
            matrix2->dw.noalias() += out->dw;
        });
    return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::sub(
        shared_mat matrix1,
        shared_mat matrix2) {
    if (matrix1->d != matrix2->d && (matrix1->d == 1 || matrix2->d == 1)) {
        if (matrix1->d == 1) {
            return sub_broadcast_reversed(matrix2, matrix1);
        }
        return sub_broadcast(matrix1, matrix2);
    }
    if (matrix1->n != matrix2->n || matrix1->d != matrix2->d)
        throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<T>>(
        matrix1->n,
        matrix1->d,
        true);
    out->w = matrix1->w - matrix2->w;
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw;
            matrix2->dw.noalias() -= out->dw;
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add_broadcast(shared_mat matrix1, shared_mat matrix2) {
    // broadcast matrix 2:
    if (matrix1->n != matrix2->n || matrix2->d != 1)
            throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<T>>(
            matrix1->n,
            matrix1->d,
            true);
    out->w = (matrix1->w.colwise() + matrix2->w.col(0)).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw;
            matrix2->dw.noalias() += out->dw.rowwise().sum();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::sub_broadcast(shared_mat matrix1, shared_mat matrix2) {
    // broadcast matrix 2:
    if (matrix1->n != matrix2->n || matrix2->d != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<T>>(
        matrix1->n,
        matrix1->d,
        true);
    out->w = (matrix1->w.colwise() - matrix2->w.col(0)).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw;
            matrix2->dw.noalias() -= out->dw.rowwise().sum();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::sub_broadcast_reversed(shared_mat matrix1, shared_mat matrix2) {
    // broadcast matrix 2:
    if (matrix1->n != matrix2->n || matrix2->d != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<T>>(
        matrix1->n,
        matrix1->d,
        true);
    out->w = ((-matrix1->w).colwise() + matrix2->w.col(0)).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out] () {
            matrix1->dw.noalias() -= out->dw;
            matrix2->dw.noalias() += out->dw.rowwise().sum();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::add(std::initializer_list<shared_mat> matrices) {
    auto out = std::make_shared<Mat<T>>(
        (*matrices.begin())->n,
        (*matrices.begin())->d,
        false);
    auto matrices_vector = vector<shared_mat>(matrices);
    for (auto& matrix : matrices_vector)
        out->w += matrix->w;
    if (needs_backprop)
        backprop.emplace_back([matrices_vector, out]() {
            for (auto& matrix : matrices_vector) {
                matrix->dw.noalias() += out->dw;
            }
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::square(shared_mat matrix) {
    auto out = std::make_shared<mat>(
            matrix->n,
            matrix->d,
            true);
    out->w = matrix->w.array().square();
    if (needs_backprop)
        backprop.emplace_back([matrix, out]() {
            matrix->dw.noalias() += 2.0 * ((matrix->w).array() * (out->dw).array()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::sigmoid(shared_mat matrix) {
    auto out = std::make_shared<mat>(
            matrix->n,
            matrix->d,
            true);
    out->w = matrix->w.unaryExpr(utils::sigmoid_operator<T>());
    if (needs_backprop)
        backprop.emplace_back([matrix, out](){
            matrix->dw.noalias() += (((out->w).array() - out->w.array().square()) * out->dw.array()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::softmax(shared_mat matrix, T temperature) {
    auto out = std::make_shared<Mat<T>>(
            matrix->n,
            matrix->d,
            false);

    DEBUG_ASSERT_NOT_NAN(matrix->w);

    auto layer_max = matrix->w.colwise().maxCoeff().array().matrix();
    auto exped_distributions = (matrix->w.rowwise() - layer_max.row(0)).array().exp().matrix();

    auto total_distribution = exped_distributions.colwise().sum().array().matrix();
    out->w = (exped_distributions.array().rowwise() / total_distribution.row(0).array());

    DEBUG_ASSERT_POSITIVE(out->w);

    if (needs_backprop)
        backprop.emplace_back([matrix, temperature, out](){
            matrix->dw.noalias() += (((out->w).array() - out->w.array().square())/temperature * out->dw.array()).matrix();
        });
    return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::steep_sigmoid(shared_mat matrix, T aggressiveness) {
    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);
    out->w = matrix->w.unaryExpr(utils::steep_sigmoid_operator<T>(aggressiveness));
    if (needs_backprop)
        backprop.emplace_back([matrix, out, aggressiveness](){
            matrix->dw.noalias() += (aggressiveness * ((out->w).array() - out->w.array().square()) * out->dw.array()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::sum(shared_mat matrix) {
    auto out = std::make_shared<mat>(1,1,true);
    out->w(0) = matrix->w.array().sum();
    if (needs_backprop)
        backprop.emplace_back([matrix, out]() {
            matrix->dw.array() += out->dw(0);
        });
    return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::mean(shared_mat matrix) {
    auto out = std::make_shared<mat>(1,1,true);
    out->w(0) = matrix->w.array().mean();
    if (needs_backprop)
        backprop.emplace_back([matrix, out](){
            matrix->dw.array() += (1.0 / (matrix->n * matrix->d)) * out->dw(0);
        });
    return out;
}



template<typename T>
typename Graph<T>::shared_mat Graph<T>::binary_cross_entropy(shared_mat matrix, T t) {
    assert(0 <= t && t <= 1);
    DEBUG_ASSERT_BOUNDS(matrix->w,0.0,1.0 + EPS);
    auto out =  std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);

    auto x = matrix->w.array();

    out->w = (-(t * x.log() + (1.0-t) * (1.0 - x).log())).matrix();

    DEBUG_ASSERT_NOT_NAN(out->w);

    if (needs_backprop)
        backprop.emplace_back([matrix, t, out](){
            auto x = matrix->w.array();
            matrix->dw.array() += (t-x) / (x*(x- 1.0) )* out->dw.array();
            DEBUG_ASSERT_NOT_NAN(matrix->dw);
        });

    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::cross_entropy(shared_mat matrix, uint answer_idx) {
    DEBUG_ASSERT_BOUNDS(matrix->w,0.0,1.0 + EPS);
    auto out =  std::make_shared<mat>(1, 1, true);

    auto x = matrix->w.array();

    out->w(0,0) = - std::log(x(answer_idx, 0) + EPS);

    DEBUG_ASSERT_NOT_NAN(out->w);

    if (needs_backprop)
        backprop.emplace_back([matrix, answer_idx, out](){
            auto x = matrix->w.array();
            matrix->dw(answer_idx, 0) += -1.0/(x(answer_idx, 0) + EPS) * out->dw(0,0);
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::log(shared_mat matrix) {
    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);
    out->w = matrix->w.array().log();
    if (needs_backprop)
        backprop.emplace_back([matrix, out](){
            matrix->dw.noalias() += ((1.0 / (matrix->w).array()) * (out->dw).array()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::exp(shared_mat matrix) {
    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);
    out->w = matrix->w.array().exp();
    if (needs_backprop)
        backprop.emplace_back([matrix, out](){
            matrix->dw.noalias() += ((out->w).array() * (out->dw).array()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::hstack(shared_mat matrix1, shared_mat matrix2) {
    if (matrix1->n != matrix2->n)
        throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
    auto out = std::make_shared<mat>(
        matrix1->n,
        matrix1->d + matrix2->d,
        true
    );
    out->w.block(0,0, matrix1->n, matrix1->d) = matrix1->w;
    out->w.block(0,matrix1->d, matrix2->n, matrix2->d) = matrix2->w;
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw.block(0,0, matrix1->n, matrix1->d);
            matrix2->dw.noalias() += out->dw.block(0,matrix1->d, matrix2->n, matrix2->d);
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::hstack(std::initializer_list<shared_mat> matrices) {
    vector<shared_mat> matrices_vector(matrices);
    return hstack(matrices_vector);
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::hstack(const std::vector<shared_mat>& matrices) {
    int n = -1;
    int d_total = 0;
    for (auto& mat : matrices) {
        if (n == -1) {
            n = mat->n;
        } else {
            if (mat->n != n) {
                throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
            }
        }
        d_total+= mat->d;
    }
    auto out = std::make_shared<mat>(
        n,
        d_total,
        true
    );
    int offset = 0;
    for (auto& mat : matrices) {
        out->w.block(0, offset, mat->n, mat->d) = mat->w;
        offset += mat->d;
    }
    if (needs_backprop)
        backprop.emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                mat->dw.noalias() += out->dw.block(0, offset, mat->n, mat->d);
                offset += mat->d;
            }
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::vstack(shared_mat matrix1, shared_mat matrix2) {
    if (matrix1->d != matrix2->d)
        throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
    auto out = std::make_shared<mat>(
        matrix1->n + matrix2->n,
        matrix1->d,
        true
    );
    out->w.block(0,0, matrix1->n, matrix1->d) = matrix1->w;
    out->w.block(matrix1->n,0, matrix2->n, matrix2->d) = matrix2->w;
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw.block(0,0, matrix1->n, matrix1->d);
            matrix2->dw.noalias() += out->dw.block(matrix1->n,0, matrix2->n, matrix2->d);
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::vstack(std::initializer_list<shared_mat> matrices) {
    vector<shared_mat> matrices_vector(matrices);
    return vstack(matrices_vector);
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::vstack(const std::vector<shared_mat>& matrices) {
    assert(matrices.size() > 0);
    int d = matrices[0]->d;
    int n_total = 0;
    for (auto& mat : matrices) {
        if (mat->d != d) {
            throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
        }
        n_total += mat->n;
    }
    auto out = std::make_shared<mat>(
        n_total,
        d,
        true
    );
    int offset = 0;
    for (auto& mat : matrices) {
        out->w.block(offset, 0, mat->n, mat->d) = mat->w;
        offset += mat->n;
    }
    if (needs_backprop)
        backprop.emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                mat->dw.noalias() += out->dw.block(offset,0, mat->n, mat->d);
                offset += mat->n;
            }
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::transpose(shared_mat matrix) {
    auto out = std::make_shared<mat>(
        matrix->d,
        matrix->n,
        true);
    out->w = matrix->w.transpose();
    if (needs_backprop)
        backprop.emplace_back([matrix, out](){
            matrix->dw.noalias() += (out->dw).transpose();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::tanh(shared_mat matrix) {
    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);
    out->w = matrix->w.unaryExpr(utils::tanh_operator<T>());
    if (needs_backprop)
        backprop.emplace_back([matrix, out](){
            matrix->dw.noalias() += (out->w.unaryExpr(utils::dtanh_operator<T>()).array() * out->dw.array()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::relu(shared_mat matrix) {
    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);
    out->w = matrix->w.unaryExpr(utils::relu_operator<T>());
    if (needs_backprop)
        this->backprop.emplace_back([matrix, out](){
            matrix->dw.noalias() += (out->w.unaryExpr(utils::sign_operator<T>()).array() * out->dw.array()).matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul(
    shared_mat matrix1,
    shared_mat matrix2) {
    if (matrix1->d != matrix2->n)
        throw std::invalid_argument("matmul dimensions misaligned.");
    auto out = std::make_shared<mat>(
        matrix1->n,
        matrix2->d,
        true);
    out->w = matrix1->w * matrix2->w;
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, out](){
            matrix1->dw.noalias() += (out->dw) * ((matrix2->w).transpose());
            matrix2->dw.noalias() += matrix1->w.transpose() * (out->dw);
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_with_bias(
    shared_mat matrix1,
    shared_mat matrix2,
    shared_mat bias) {
    if (matrix1->d != matrix2->n)
            throw std::invalid_argument("matmul dimensions misaligned.");
    if (matrix1->n != bias->n || bias->d != 1)
            throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<mat>(
            matrix1->n,
            matrix2->d,
            true);
    out->w = ((matrix1->w * matrix2->w).colwise() + bias->w.col(0)).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, matrix2, bias, out]() {
            matrix1->dw.noalias() += (out->dw) * ((matrix2->w).transpose());
            matrix2->dw.noalias() += matrix1->w.transpose() * (out->dw);
            bias->dw.noalias()    += out->dw.rowwise().sum().matrix();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_broadcast_mul_with_bias(
    shared_mat matrix1,
    shared_mat input_to_1,
    shared_mat matrix2,
    shared_mat input_to_2,
    shared_mat bias) {
    if (matrix1->d != input_to_1->n)
        throw std::invalid_argument("matmul 1 dimensions misaligned.");
    if (matrix2->d != input_to_2->n)
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2->n != bias->n || matrix1->n != bias->n || input_to_1->d != 1 || bias->d != 1)
        throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<mat>(
            matrix1->n,
            input_to_2->d,
            true);
    // both input to 1 and bias are columns,
    // so we add both of those before adding the true matrix
    // product in broadcasted form
    out->w = (
          (
              (
                  (matrix2->w * input_to_2->w)
              )
          ).colwise() + (bias->w + (matrix1->w * input_to_1->w)).col(0)
      ).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out] () {
            // first multiply:
            // broadcasting input means taking outer product here:
            matrix1->dw += ((out->dw).rowwise().sum() * ((input_to_1->w).transpose()));
            // broadcasting output means sum after the reverse product here:
            input_to_1->dw.noalias() += (matrix1->w.transpose() * (out->dw)).rowwise().sum();
            // second multiply:
            matrix2->dw.noalias() += (out->dw) * ((input_to_2->w).transpose());

            input_to_2->dw.noalias() += matrix2->w.transpose() * (out->dw);
            // bias vector:
            bias->dw.noalias() += out->dw.rowwise().sum();
        });
    return out;
}


template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_mul_with_bias(std::initializer_list<shared_mat> matrices) {
    vector<shared_mat> matrices_vector(matrices);
    return mul_add_mul_with_bias(matrices_vector);
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_mul_with_bias(const vector<shared_mat>& matrices) {
    auto out = std::make_shared<mat>(
            matrices[0]->n,
            matrices[1]->d,
            false);
    auto matrices_ptr = matrices.begin();
    while (matrices_ptr != (matrices.end() - 1)) {
        out->w += (*matrices_ptr)->w * (*(matrices_ptr + 1))->w;
        DEBUG_ASSERT_MAT_NOT_NAN(out);
        DEBUG_ASSERT_MAT_NOT_NAN((*matrices_ptr));
        DEBUG_ASSERT_MAT_NOT_NAN((*(matrices_ptr + 1)));
        matrices_ptr+=2;
    }

    DEBUG_ASSERT_NOT_NAN((*(matrices.begin() + matrices.size() - 1))->w);
    out->w.colwise() += (*(matrices.begin() + matrices.size() - 1))->w.col(0);
    if (needs_backprop)
        backprop.emplace_back([matrices, out](){
            auto matrices_ptr = matrices.begin();
            while (matrices_ptr != (matrices.end() - 1)) {
                (*matrices_ptr)->dw.noalias()     += (out->dw) * (*(matrices_ptr+1))->w.transpose();
                (*(matrices_ptr+1))->dw.noalias() += (*matrices_ptr)->w.transpose() * (out->dw);
                matrices_ptr+=2;
            }
            auto bias = *(matrices.begin() + matrices.size() - 1);
            bias->dw.noalias() += out->dw.rowwise().sum();
        });

    DEBUG_ASSERT_NOT_NAN(out->w);
    return out;
}

// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
template<typename T>
typename Graph<T>::shared_mat Graph<T>::mul_add_mul_with_bias(
    shared_mat matrix1,
    shared_mat input_to_1,
    shared_mat matrix2,
    shared_mat input_to_2,
    shared_mat bias) {
    if (matrix1->d != input_to_1->n)
        throw std::invalid_argument("matmul 1 dimensions misaligned.");
    if (matrix2->d != input_to_2->n)
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2->n != bias->n || matrix1->n != bias->n || bias->d != 1)
        throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    if (input_to_1->d != input_to_2->d) {
        if (input_to_1->d == 1) {
            return mul_add_broadcast_mul_with_bias(matrix1, input_to_1, matrix2, input_to_2, bias);
        }
        return mul_add_broadcast_mul_with_bias(matrix2, input_to_2, matrix1, input_to_1, bias);
    }
    auto out = std::make_shared<mat>(
            matrix1->n,
            input_to_1->d,
            true);
    out->w = (
              (
                  (
                      (matrix1->w * input_to_1->w) +
                      (matrix2->w * input_to_2->w)
                  )
              ).colwise() + bias->w.col(0)
          ).matrix();
    if (needs_backprop)
        backprop.emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out](){
            // first multiply:
            // broadcasting input means taking outer product here:
            matrix1->dw += (out->dw * (input_to_1->w).transpose());
            // broadcasting output means sum after the reverse product here:
            input_to_1->dw.noalias() += matrix1->w.transpose() * (out->dw);
            // second multiply:
            matrix2->dw.noalias() += (out->dw) * (input_to_2->w).transpose();

            input_to_2->dw.noalias() += matrix2->w.transpose() * (out->dw);
            // bias vector:
            bias->dw.noalias() += out->dw.rowwise().sum();
        });
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::rows_pluck(
        shared_mat matrix,
        Indexing::Index indices
        ) {
    Timer rp_timer("ops_rows_pluck");
    auto out = std::make_shared<mat>(
        matrix->d,
        indices.size(),
        true);

    for (std::size_t offset = 0; offset < indices.size(); ++offset) {
        out->w.col(offset) = matrix->w.row(indices[offset]).transpose();
    }
    rp_timer.stop();
    if (needs_backprop) {
        backprop.emplace_back([matrix, out, indices](){
            auto index_ptr = indices.data();
            for (std::size_t i = 0; i < out->d; ++i) {
                // for each row do the same operation as for row_pluck:
                matrix->dw.row(*index_ptr).noalias() += out->dw.col(i).transpose();
                index_ptr++;
            }
        });
    }
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::dropout(
    shared_mat matrix,
    T drop_prob) {

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);

    auto bool_mat = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(matrix->n, matrix->d);

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix->w.data();
    auto out_ptr  = out->w.data();
    auto bool_ptr = bool_mat->data();

    for (int i = 0; i < matrix->n * matrix->d;++i) {
        (*bool_ptr) = distribution(generator) ? 1.0 : 0.0;
        (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
        out_ptr++;
        data_ptr++;
        bool_ptr++;
    }

    if (needs_backprop) {
        backprop.emplace_back([matrix, out, bool_mat](){
            matrix->dw += (out->dw.array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::dropout_normalized(
    shared_mat matrix,
    T drop_prob) {

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);

    auto bool_mat = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(matrix->n, matrix->d);

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix->w.data();
    auto out_ptr  = out->w.data();
    auto bool_ptr = bool_mat->data();

    T normalized_drop_prob = 1.0 / (1.0 - drop_prob);
    for (int i = 0; i < matrix->n * matrix->d;++i) {
        (*bool_ptr) = distribution(generator) ? normalized_drop_prob : 0.0;
        (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
        out_ptr++;
        data_ptr++;
        bool_ptr++;
    }

    if (needs_backprop) {
        backprop.emplace_back([matrix, out, bool_mat](){
            matrix->dw += (out->dw.array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::fast_dropout(shared_mat matrix) {
    auto out = std::make_shared<mat>(
        matrix->n,
        matrix->d,
        true);

    auto randn_mat = std::make_shared<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(matrix->n, matrix->d);

    std::default_random_engine generator;
    std::normal_distribution<T> distribution(1.0, 1.0);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix->w.data();
    auto out_ptr  = out->w.data();
    auto randn_ptr = randn_mat->data();

    for (int i = 0; i < matrix->n * matrix->d;++i) {
        (*randn_ptr) = distribution(generator);
        (*out_ptr) = (*randn_ptr) * *data_ptr;
        out_ptr++;
        data_ptr++;
        randn_ptr++;
    }

    if (needs_backprop) {
        backprop.emplace_back([matrix, out, randn_mat](){
            matrix->dw += (out->dw.array() * (*randn_mat).array()).matrix();
        });
    }
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::rows_cols_pluck(
        shared_mat matrix,
        Indexing::Index row_indices,
        Indexing::Index col_indices) {
    if (row_indices.size() != col_indices.size())
        throw std::invalid_argument("Cannot pluck column row pairs, not the same amount of row and column indices.");
    Timer rp_timer("ops_rows_pluck");
        auto out = std::make_shared<mat>(
            1,
            row_indices.size(),
            true);
        for (int offset = 0; offset < row_indices.size(); ++offset)
            out->w(offset) = matrix->w(row_indices[offset], col_indices[offset]);
    rp_timer.stop();
    if (needs_backprop) {
        backprop.emplace_back([matrix, out, row_indices, col_indices](){
            auto row_index_ptr = row_indices.data();
            auto col_index_ptr = col_indices.data();
            for (int i = 0; i < out->d; ++i) {
                // for each row do the same operation as for row_pluck:
                matrix->dw(*row_index_ptr, *col_index_ptr) += out->dw(i);
                row_index_ptr++;
                col_index_ptr++;
            }
        });
    }
    return out;
}

template<typename T>
typename Graph<T>::shared_mat Graph<T>::row_pluck(
        shared_mat matrix,
        int row) {
    auto out = std::make_shared<mat>(matrix->d, 1, true);
    out->w = matrix->w.row(row).transpose();
    if (needs_backprop)
        backprop.emplace_back([matrix, out, row]() {
            matrix->dw.row(row).noalias() += out->dw.col(0).transpose();
        });
    return out;
}
template<typename T>
int Graph<T>::backprop_size() const {
    return backprop.size();
}

template class Graph<float>;
template class Graph<double>;

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Graph<T>& a) {
    return strm << "<#Graph needs_backprop=" << a.needs_backprop<< ", backprop_size=" << a.backprop_size() << " >";
}

template std::ostream& operator<< <double>(std::ostream& strm, const Graph<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Graph<float>& a);
