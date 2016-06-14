////////////////////////////////////////////////////////////////////////////////
//                                   UTILS                                    //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<int dstdim>
    mshadow::Shape<dstdim> canonical_reshape(const std::vector<int>& src_shape,
                                             bool collapse_leading,
                                             int dim_to_collapse) {
        int srcdim = src_shape.size();
        if (dim_to_collapse > 1) {
            auto collapsed_src_shape = src_shape;
            if (collapse_leading) {
                // {2, 3, 4, 5} -> collapse(2) -> {2, 3, 20}
                for (int i = 0; i < dim_to_collapse - 1; i++) {
                    collapsed_src_shape[srcdim - dim_to_collapse] *= collapsed_src_shape[srcdim - i - 1];
                }
                collapsed_src_shape.erase(collapsed_src_shape.end() + 1 - dim_to_collapse, collapsed_src_shape.end());
                return canonical_reshape<dstdim>(collapsed_src_shape, collapse_leading, 0);
            } else {
                // {2, 3, 4, 5} -> collapse(2) -> {6, 4, 5}
                for (int i = 0; i < dim_to_collapse - 1; i++) {
                    collapsed_src_shape[dim_to_collapse - 1] *= collapsed_src_shape[i];
                }
                collapsed_src_shape.erase(collapsed_src_shape.begin(), collapsed_src_shape.begin() + dim_to_collapse - 1);
                return canonical_reshape<dstdim>(collapsed_src_shape, collapse_leading, 0);
            }
        }

        mshadow::Shape<dstdim> res;
        #pragma unroll
        for (int i = 0; i < dstdim; i++) res[i] = 1;

        dim_to_collapse = std::min(dim_to_collapse, srcdim);

        if (collapse_leading) {
            for (int i = 0; i < srcdim; ++i) {
                res[std::max(dstdim - 1 - i, 0)] *= src_shape[srcdim - 1 - i];
            }
        } else {
            for (int i = 0; i < srcdim; ++i) {
                res[std::min(i, dstdim - 1)] *= src_shape[i];
            }
        }
        return res;
    }
}
////////////////////////////////////////////////////////////////////////////////
//                            TYPED ARRAY SHARED                              //
//                                   ---                                      //
//  Common to both CPU and GPU implementations of TypedArray below.           //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<typename MDevT, typename T>
    template<int dim>
    mshadow::Tensor<MDevT, dim, T> TypedArrayShared<MDevT,T>::mtensor(memory::AM access_mode, bool collapse_leading, const int& dim_to_collapse) const {
        return mshadow::Tensor<MDevT, dim, T>(
            ptr_internal(access_mode),
            internal::canonical_reshape<dim>(array.shape(), collapse_leading, dim_to_collapse)
        );
    }

    template<typename MDevT, typename T>
    template<int dim>
    mshadow::Tensor<MDevT, dim, T> TypedArrayShared<MDevT,T>::contiguous_d(memory::AM access_mode, bool collapse_leading, const int& dim_to_collapse) const {
        ASSERT2(this->array.contiguous_memory(),
            "contiguous_d can only be called on a TypedArray that has contiguous memory.");
        return mtensor<dim>(access_mode, collapse_leading, dim_to_collapse);
    }

    template<typename MDevT, typename T>
    template<int dim>
    DaliWrapperExp<MDevT, dim, T> TypedArrayShared<MDevT,T>::d(memory::AM access_mode, bool collapse_leading, const int& dim_to_collapse) const {
        return MakeDaliWrapperExp(mtensor<dim>(access_mode, collapse_leading, dim_to_collapse), array);
    }

////////////////////////////////////////////////////////////////////////////////
//                            TYPED SUBTENSOR SHARED                          //
////////////////////////////////////////////////////////////////////////////////


    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeFromRowsExp<mshadow::Tensor<MDevT, dim, IndexT>,
                                   mshadow::Tensor<MDevT, dim + 1, T>,
                                   T,
                                   IndexT>
    TypedArraySubtensorShared<MDevT,T,IndexT>::contiguous_d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take_from_rows(
            indices.template contiguous_d<dim>(access_mode, collapse_leading),
            source.template contiguous_d<dim + 1>(access_mode, collapse_leading)
        );
    }

    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeFromRowsExp<DaliWrapperExp<MDevT, dim, IndexT>,
                                   DaliWrapperExp<MDevT, dim+1, T>,
                                   T,
                                   IndexT>
    TypedArraySubtensorShared<MDevT,T,IndexT>::d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take_from_rows(
            indices.template d<dim>(access_mode, collapse_leading),
            source.template d<dim + 1>(access_mode, collapse_leading)
        );
    }


////////////////////////////////////////////////////////////////////////////////
//                            TYPED GATHER SHARED                             //
////////////////////////////////////////////////////////////////////////////////


    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeExp<mshadow::Tensor<MDevT, 1, IndexT>,
                           mshadow::Tensor<MDevT, dim, T>,
                           T,
                           IndexT>
    TypedArrayGatherShared<MDevT,T,IndexT>::contiguous_d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take(
            indices.template contiguous_d<1>(access_mode, collapse_leading),
            source.template contiguous_d<dim>(access_mode, collapse_leading)
        );
    }

    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeExp<DaliWrapperExp<MDevT, 1,   IndexT>,
                           DaliWrapperExp<MDevT, dim, T>,
                           T,
                           IndexT>
    TypedArrayGatherShared<MDevT,T,IndexT>::d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take(
            indices.template d<1>(access_mode, collapse_leading),
            source.template d<dim>(access_mode, collapse_leading)
        );
    }

}  // namespace internal
