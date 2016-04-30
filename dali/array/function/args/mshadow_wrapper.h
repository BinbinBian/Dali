#ifndef DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H
#define DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H

#include <cassert>

#include "dali/array/function/typed_array.h"
#include "dali/array/memory/device.h"
#include "dali/utils/assert2.h"

////////////////////////////////////////////////////////////////////////////////
//                           MSHADOW_WRAPPER_EXP                              //
//                                   ---                                      //
//  This expression is used to inject Dali striding information to mshadow    //
//  expression processor                                                      //
////////////////////////////////////////////////////////////////////////////////


template<typename SrcExp, typename DType, int srcdim>
struct DaliWrapperExp: public mshadow::expr::MakeTensorExp<
                                    DaliWrapperExp<SrcExp, DType, srcdim>,
                                    SrcExp, srcdim, DType
                                 > {
    const SrcExp src_;
    const Array array;

    DaliWrapperExp(const SrcExp &src, const Array& dali_src) :
            src_(src),
            array(dali_src) {
        ASSERT2(src_.shape_[srcdim - 1] == src_.stride_,
                "DaliWrapperExp should never reach that condition (only tensors should be passed as arguments).");
        this->shape_ = mshadow::expr::ShapeCheck<srcdim, SrcExp>::Check(src_);
    }
};


namespace mshadow {
    namespace expr {
        template<typename SrcExp, typename DType, int etype>
        inline DaliWrapperExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
        MakeDaliWrapperExp(const Exp<SrcExp, DType, etype> &src, const Array& dali_src) {
            return DaliWrapperExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), dali_src);
        }

        template<typename SrcExp, typename DType, int srcdim>
        struct ExpInfo<DaliWrapperExp<SrcExp, DType, srcdim> > {
            static const int kDimSrc = ExpInfo<SrcExp>::kDim;
            static const int kDim = kDimSrc >= 0 ? srcdim : -1;
            static const int kDevMask = ExpInfo<SrcExp>::kDevMask;
        };

        template<typename SrcExp, typename DType, int srcdim>
        struct ShapeCheck<srcdim, DaliWrapperExp<SrcExp, DType, srcdim> > {
            inline static Shape<srcdim>
            Check(const DaliWrapperExp<SrcExp, DType, srcdim> &t) {
                return t.shape_;
            }
        };

        template<typename SrcExp, typename DType, int srcdim>
        struct Plan<DaliWrapperExp<SrcExp, DType, srcdim>, DType> {
          public:
            explicit Plan(const DaliWrapperExp<SrcExp, DType, srcdim> &e) :
                    src_(MakePlan(e.src_)) {
                    // shape(e.array.shape()),
                    // strides(e.array.strides()) {
                // has_strides = !strides.empty();
                has_strides = false;
            }

            MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
                if (!has_strides) {
                    return src_.Eval(i, j);
                } else {
                    assert(false);
                    // const int ndims = shape.size();
                    //
                    // index_t new_i = 0;
                    // index_t residual_shape = strides[ndims-1];
                    //
                    // for (int dim_idx = ndims - 2; dim_idx >= 0; --dim_idx) {
                    //     new_i += ((i % shape[dim_idx]) * strides[dim_idx]) * residual_shape;
                    //     i /=  shape[dim_idx];
                    //     residual_shape *= shape[dim_idx] * strides[dim_idx];
                    // }
                    //
                    // return src_.Eval(new_i, j * strides[ndims - 1]);
                    return 0;
                }
            }

          private:
            Plan<SrcExp, DType> src_;
            // std::vector<int> shape;
            // std::vector<int> strides;
            bool has_strides;
        };
    } //namespace expr
} // namespace mshadow
////////////////////////////////////////////////////////////////////////////////
//                             MSHADOW_WRAPPER                                //
//                                   ---                                      //
//  This class would not be needed at all if we defined to_mshadow_expr       //
//  function on Array. The reason not to do that is to hide all mshadow usage //
//  in cpp files whereever possible.                                          //
////////////////////////////////////////////////////////////////////////////////


template<int devT,typename T, typename ExprT>
struct MshadowWrapper {
    static inline auto wrap(const ExprT& sth, memory::Device device) ->
            decltype(sth.template to_mshadow_expr<devT,T>(device)) {
        return sth.template to_mshadow_expr<devT,T>(device);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,Array> {
    static inline auto wrap(const Array& array, memory::Device device) ->
            decltype(MakeDaliWrapperExp(TypedArray<devT,T>(array, device).d2(), array)) {
        return MakeDaliWrapperExp(TypedArray<devT,T>(array, device).d2(), array);
    }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,float> {
    static inline T wrap(const float& scalar, memory::Device device) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,double> {
    static inline T wrap(const double& scalar, memory::Device device) { return (T)scalar; }
};

template<int devT,typename T>
struct MshadowWrapper<devT,T,int> {
    static inline T wrap(const int& scalar, memory::Device device) { return (T)scalar; }
};

#endif // DALI_ARRAY_FUNCTION_ARGS_MSHADOW_WRAPPER_H
