#include "typed_array.h"

#include "dali/config.h"
#include "dali/utils/print_utils.h"


namespace internal {
    template<typename MDevT, typename T>
    T* TypedArrayShared<MDevT,T>::ptr_internal(memory::AM access_mode) const {
        return (T*)(array.memory()->data(device, access_mode)) + array.offset();
    }

    template<typename MDevT, typename T>
    T* TypedArrayShared<MDevT,T>::ptr(memory::AM access_mode) const {
        ASSERT2(this->array.contiguous_memory(),
                "This function is only supported for contiguous_memory");
        return ptr_internal(access_mode);
    }

    template<typename MDevT, typename T>
    TypedArrayShared<MDevT, T>::TypedArrayShared(const Array& _array, const memory::Device& _device, const std::vector<int>& _output_shape)
            : array(_array.reshape_broadcasted(_output_shape)), device(_device) {
    }

    template<typename MDevT, typename T>
    std::tuple<bool,mshadow::Tensor<MDevT, 2, T>> TypedArrayShared<MDevT, T>::blas_friendly_tensor() const {
        ASSERT2(array.ndim() == 2,
                utils::MS() << "blas_friendly_tensor is only available to 2D tensors ("
                            << array.ndim() << "D tensor passed.)");
        if (array.strides().size() == 0) {
            return std::make_tuple(false, mtensor<2>());
        }

        const std::vector<int>& a_strides = array.strides();

        if (a_strides[1] == 1) {
            auto ret = mtensor<2>();
            ret.stride_ = a_strides[0];
            return std::make_tuple(false, ret);
        } else if (a_strides[0] == 1) {
            auto ret = mtensor<2>();
            ret.stride_ = a_strides[1];
            return std::make_tuple(true, ret);
        } else {
            ASSERT2(a_strides[0] == 1 || a_strides[1] == 1,
                    utils::MS() << "gemm does not support doubly strided matrices (input strides: " << a_strides << ")");
        }
    }



    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 1, T> TypedArrayShared<MDevT,T>::contiguous_d1(memory::AM access_mode) const { return contiguous_d<1>(access_mode); }
    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 2, T> TypedArrayShared<MDevT,T>::contiguous_d2(memory::AM access_mode) const { return contiguous_d<2>(access_mode); }
    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 3, T> TypedArrayShared<MDevT,T>::contiguous_d3(memory::AM access_mode) const { return contiguous_d<3>(access_mode); }
    template<typename MDevT, typename T>
    mshadow::Tensor<MDevT, 4, T> TypedArrayShared<MDevT,T>::contiguous_d4(memory::AM access_mode) const { return contiguous_d<4>(access_mode); }

    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 1, T> TypedArrayShared<MDevT,T>::d1(memory::AM access_mode) const { return d<1>(access_mode); }
    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 2, T> TypedArrayShared<MDevT,T>::d2(memory::AM access_mode) const { return d<2>(access_mode); }
    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 3, T> TypedArrayShared<MDevT,T>::d3(memory::AM access_mode) const { return d<3>(access_mode); }
    template<typename MDevT, typename T>
    DaliWrapperExp<MDevT, 4, T> TypedArrayShared<MDevT,T>::d4(memory::AM access_mode) const { return d<4>(access_mode); }


    template class TypedArrayShared<mshadow::cpu, int>;
    template class TypedArrayShared<mshadow::cpu, float>;
    template class TypedArrayShared<mshadow::cpu, double>;
    template class TypedArrayShared<mshadow::gpu, int>;
    template class TypedArrayShared<mshadow::gpu, float>;
    template class TypedArrayShared<mshadow::gpu, double>;

} // namespace internal

template class TypedArray<memory::DEVICE_T_CPU, int>;
template class TypedArray<memory::DEVICE_T_CPU, float>;
template class TypedArray<memory::DEVICE_T_CPU, double>;


#ifdef DALI_USE_CUDA
    template<typename T>
    thrust::device_ptr<T> TypedArray<memory::DEVICE_T_GPU, T>::to_thrust(memory::AM access_mode) const {
        return thrust::device_pointer_cast(this->ptr(access_mode));
    }

    template class TypedArray<memory::DEVICE_T_GPU, int>;
    template class TypedArray<memory::DEVICE_T_GPU, float>;
    template class TypedArray<memory::DEVICE_T_GPU, double>;

#endif
