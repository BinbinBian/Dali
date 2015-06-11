#include "MatInternal.h"

using std::vector;
using std::string;

/* MatInternal */

template<typename R>
std::atomic<int> MatInternal<R>::next_matrix(0);

template<typename R>
MatInternal<R>::MatInternal(dim_t n, dim_t d, bool fill_zeros) :
        w(n, d),
        dims({n, d}),
        id(next_matrix.fetch_add(1)) {
    if (fill_zeros) {
        w.fill(0);
    }
}

template<typename R>
MatInternal<R>::MatInternal(const MatInternal<R>& m) :
        w(m.w),
        dims(m.dims),
        id(m.id) {
}

template<typename R>
MatInternal<R>::operator typename MatInternal<R>::eigen_mat () {
    return w;
}

template<typename R>
R& MatInternal<R>::operator()(int i, int j) {
    return w(i,j);
}

template<typename R>
R MatInternal<R>::operator()(int i, int j) const {
    return w(i,j);
}

template<typename R>
R& MatInternal<R>::operator()(int i) {
    return w(i);
}

template<typename R>
R MatInternal<R>::operator()(int i) const {
    return w(i);
}

template<typename R>
const R* MatInternal<R>::data() const {
    return w.data();
}

template<typename R>
R* MatInternal<R>::data() {
    return w.data();
}

template<typename R>
void MatInternal<R>::print() const {
    for (int i = 0; i < dims[0] ; ++i) {
            std::cout << (i == 0 ? "[" : " ");
            for (int j = 0; j < dims[1]; ++j) {
                    std::cout << std::fixed
                              << std::setw( 7 ) // keep 7 digits
                              << std::setprecision( 3 ) // use 3 decimals
                              << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                              << w(i,j) << " ";
            }
            std::cout << (i == dims[0] - 1 ? "]" : "\n");
    }
    std::cout << std::endl;
}

/* GradInternal */

template<typename R>
GradInternal<R>::GradInternal(dim_t n, dim_t d, bool fill_zeros) :
        dw(n, d) {
    if (fill_zeros) {
        dw.fill(0);
    }
}

template<typename R>
GradInternal<R>::GradInternal(const GradInternal<R>& g) :
        dw(g.dw) {
}

template<typename R>
GradInternal<R>::operator typename GradInternal<R>::eigen_mat () {
    return dw;
}



template<typename R>
R& GradInternal<R>::operator()(int i, int j) {
    return dw(i,j);
}

template<typename R>
R GradInternal<R>::operator()(int i, int j) const {
    return dw(i,j);
}

template<typename R>
R& GradInternal<R>::operator()(int i) {
    return dw(i);
}

template<typename R>
R GradInternal<R>::operator()(int i) const {
    return dw(i);
}

template<typename R>
const R* GradInternal<R>::data() const {
    return dw.data();
}

template<typename R>
R* GradInternal<R>::data() {
    return dw.data();
}

template class MatInternal<float>;
template class MatInternal<double>;
template class GradInternal<float>;
template class GradInternal<double>;

/** END GRADINTERNAL **/
