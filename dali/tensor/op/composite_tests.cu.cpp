#include <chrono>
#include <vector>
#include <iomanip>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/op.h"
#include "dali/array/op/initializer.h"

using std::vector;

TEST(TensorCompositeTests, matrix_dot_with_bias) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::dot_with_bias(Xs[0], Xs[1], Xs[2]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(-10, 10, {num_examples, input_size}, DTYPE_DOUBLE);
        auto W = Tensor::uniform(-10, 10, {input_size, hidden_size}, DTYPE_DOUBLE);
        auto bias = Tensor::uniform(-2, 2,  {hidden_size}, DTYPE_DOUBLE)[Broadcast()];
        ASSERT_TRUE(gradient_same(functor, {X, W, bias}, 1e-4));
    }
}

TEST(TensorCompositeTests, matrix_multiple_dot_with_bias) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::multiple_dot_with_bias({Xs[0], Xs[2]}, {Xs[1], Xs[3]}, Xs[4]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(-10, 10, {num_examples, input_size}, DTYPE_DOUBLE);
        auto W = Tensor::uniform(-10, 10, {input_size, hidden_size}, DTYPE_DOUBLE);

        auto X_other = Tensor::uniform(-10, 10, {num_examples, other_input_size}, DTYPE_DOUBLE);
        auto W_other = Tensor::uniform(-10, 10, {other_input_size, hidden_size}, DTYPE_DOUBLE);

        auto bias = Tensor::uniform(-2, 2,  {hidden_size}, DTYPE_DOUBLE)[Broadcast()];
        ASSERT_TRUE(gradient_same(functor, { X, W, X_other, W_other, bias}, 0.0003));
    }
}

TEST(TensorCompositeTests, matrix_multiple_dot_with_bias_fancy_broadcast) {
    auto functor = [](vector<Tensor> Xs)-> Tensor {
        return tensor_ops::multiple_dot_with_bias({Xs[0], Xs[2], Xs[4]}, {Xs[1], Xs[3], Xs[5]}, Xs[6]);
    };
    int num_examples = 20;
    int hidden_size = 10;
    int input_size = 5;
    int other_input_size = 7;
    EXPERIMENT_REPEAT {
        auto X = Tensor::uniform(-10, 10, {num_examples, input_size}, DTYPE_DOUBLE);
        auto W = Tensor::uniform(-10, 10, {input_size, hidden_size}, DTYPE_DOUBLE);

        auto X_fancy   = Tensor::uniform(-10, 10, {input_size}, DTYPE_DOUBLE)[Broadcast()];
        auto W_fancy = Tensor::uniform(-10, 10, {input_size, hidden_size}, DTYPE_DOUBLE);

        auto X_other = Tensor::uniform(-10, 10, {num_examples, other_input_size}, DTYPE_DOUBLE);
        auto W_other = Tensor::uniform(-10, 10, {other_input_size, hidden_size}, DTYPE_DOUBLE);

        auto bias = Tensor::uniform(-2, 2,  {hidden_size}, DTYPE_DOUBLE)[Broadcast()];
        ASSERT_TRUE(gradient_same(functor, {X, W, X_fancy, W_fancy, X_other, W_other, bias}, 0.0003));
    }
}
