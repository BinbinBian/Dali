#include "dali/mat/Mat.h"

using namespace Eigen;
using std::vector;
using std::string;
using std::stringstream;
using utils::assert2;
using std::make_shared;

const vector<dim_t> mat_missing_dimensions({0,0});

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


/* weights */

template<typename R>
typename weights<R>::initializer_t weights<R>::uninitialized() {
    return [](Mat<R>&){};
};

template<typename R>
typename weights<R>::initializer_t weights<R>::zeros() {
    return [](Mat<R>& matrix){
        matrix.w().fill(0);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R lower, R upper) {
    return [lower, upper](Mat<R>& matrix){
        // TODO(szymon): scale by row length.
        std::default_random_engine generator;
        std::uniform_real_distribution<R> distribution(lower, upper);
        std::random_device rd;
        generator.seed(rd());
        auto randn = [&] (int) {return distribution(generator);};
        matrix.w() = Mat<R>::eigen_mat::NullaryExpr(matrix.dims(0),
                                            matrix.dims(1),
                                            randn);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::uniform(R bound) {
    return uniform(-bound/2.0, bound/2.0);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R mean, R std) {
    return [mean, std](Mat<R>& matrix){
        std::default_random_engine generator;
        std::normal_distribution<R> distribution(mean, std);
        std::random_device rd;
        generator.seed(rd());
        auto randn = [&distribution, &generator] (int) {return distribution(generator);};
        matrix.w() = Mat<R>::eigen_mat::NullaryExpr(
            matrix.dims(0),
            matrix.dims(1),
            randn);
    };
};

template<typename R>
typename weights<R>::initializer_t weights<R>::gaussian(R std) {
    return gaussian(0.0, std);
}

template<typename R>
typename weights<R>::initializer_t weights<R>::svd(initializer_t preinitializer) {
    return [preinitializer](Mat<R>& matrix) {
        assert(matrix.dims().size() == 2);
        preinitializer(matrix);
        auto svd = matrix.w().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        int n = matrix.dims(0);
        int d = matrix.dims(1);
        if (n < d) {
            matrix.w() = svd.matrixV().block(0, 0, n, d);
        } else {
            matrix.w() = svd.matrixU().block(0, 0, n, d);
        }
    };
}

/* Mat */
// this does not need to initialize anything once we get rid of w and dw.
template<typename R>
Mat<R>::Mat() : Mat(0,0) {
}

template<typename R>
typename Mat<R>::eigen_mat& Mat<R>::w() const {
    return m->w;
}

template<typename R>
typename Mat<R>::eigen_mat& Mat<R>::dw() const {
    return g->dw;
}

template<typename R>
const vector<dim_t>& Mat<R>::dims() const {
    if (m != nullptr)
        return m->dims;
    return mat_missing_dimensions;
}

template<typename R>
const dim_t Mat<R>::dims(int idx) const {
    if (m != nullptr)
        return m->dims[idx];
    return (dim_t) 0;
}

template<typename R>
bool Mat<R>::empty() const {
    return number_of_elements() == 0;
}

template<typename R>
const int Mat<R>::id() const {
    if (m != nullptr)
        return m->id;
    return -1;
}

template<typename R>
Mat<R>::Mat(dim_t n, dim_t d) : Mat(n,d, true) {
}

template<typename R>
void Mat<R>::resize(dim_t n, dim_t d) {
    m->dims[0] = n;
    m->dims[1] = d;
    w().conservativeResize(n, d);
    dw().conservativeResize(n, d);
}

/**
This is the only Matrix constructor, all other
constructors reference this one.
If this one breaks, the whole ship goes to the
bottom of the ocean.
Do not let this one break. Please

-Sincerely, the Tux Family

Note: the copy constructor below is only a sideshow,
**this** is where the action is!

**/
template<typename R>
Mat<R>::Mat(dim_t n, dim_t d, typename weights<R>::initializer_t wi) :
        name(nullptr), constant(false) {
    // Don't fill with zeros - it's initializer's job.
    m = make_shared<MatInternal<R>>(n, d, false);
    // We always reset the grad calculation
    g = make_shared<GradInternal<R>>(n, d, true);

    wi(*this);
}

template<typename R>
Mat<R>::Mat (dim_t n, dim_t d, bool fill_zeros) :
        Mat(n, d, fill_zeros ? weights<R>::zeros() : weights<R>::uninitialized()) {
}

template<typename R>
Mat<R>::Mat(string fname) :
        name(nullptr),
        constant(false) {
    auto arr = cnpy::npy_load(fname);
    vector<uint> npy_dims = {arr.shape[0], arr.shape.size() > 1 ? arr.shape[1] : 1};
    m = make_shared<MatInternal<R>>(npy_dims[0], npy_dims[1], false);
    g = make_shared<GradInternal<R>>(npy_dims[0], npy_dims[1], true);

    if (arr.word_size == sizeof(double)) {
        double* loaded_data_double = reinterpret_cast<double*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, dims(0), dims(1));
            w() = wrapped_mat_double_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, dims(0), dims(1));
            w() = wrapped_mat_double.cast<R>();
        }
    } else if (arr.word_size == sizeof(float)) {
        float* loaded_data_float = reinterpret_cast<float*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, dims(0), dims(1));
            w() = wrapped_mat_float_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, dims(0), dims(1));
            w() = wrapped_mat_float.cast<R>();
        }
    } else {
        stringstream error_msg;
        error_msg << "Could not load numpy matrix : \""
           << fname << "\". File dtype (" << arr.word_size << ") not recognized as float or double.";
        throw std::invalid_argument(error_msg.str());
    }
    arr.destruct();
}

template<typename R>
Mat<R>::Mat(const Mat<R>& other, bool copy_w, bool copy_dw) :
        name(other.name),
        constant(other.constant) {

    if (copy_w && other.m != nullptr) {
        // This copies memory using copy constructor
        // The copy is only executed if matrix was actually initialized
        // hence the && other.m part.
        m = make_shared<MatInternal<R>>(*other.m);
    } else {
        // This does not. (only shared_ptr is copied).
        m = other.m;
    }

    if (copy_dw && other.g != nullptr) {
        // see comment for copy_w.
        g = make_shared<GradInternal<R>>(*other.g);
    } else {
        g = other.g;
    }
}

template<typename R>
Mat<R> Mat<R>::shallow_copy() {
    return Mat(*this, false, true);
}

template<typename R>
void Mat<R>::set_name(string& _name) {
    name = std::make_shared<string>(_name);
}

template<typename R>
void Mat<R>::set_name(char * _name) {
    name = std::make_shared<string>(_name);
}

template<typename R>
void Mat<R>::set_name(const char * _name) {
    name = std::make_shared<string>(_name);
}

template<typename R>
void Mat<R>::print() const {
    for (int i = 0; i < dims(0) ; ++i) {
            std::cout << (i == 0 ? "[" : " ");
            for (int j = 0; j < dims(1); ++j) {
                    std::cout << std::fixed
                              << std::setw( 7 ) // keep 7 digits
                              << std::setprecision( 3 ) // use 3 decimals
                              << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                              << this->w()(i,j) << " ";
            }
            std::cout << (i == dims(0) - 1 ? "]" : "\n");
    }
    std::cout << std::endl;
}

template<typename R>
void Mat<R>::grad() {
    assert2(dims(0) == 1 && dims(1) == 1,
            "Grad only works on a \"scalar\" matrix, a 1x1 matrix. "
            "Call G.sum or G.mean before using grad.");
    if (graph::backprop_enabled)
        g->dw(0) += 1;
}

template<typename R>
void Mat<R>::npy_save (string fname, string mode) {
    cnpy::npy_save(fname, w().data(), dims().data(), dims().size(), mode);
}

template<typename R>
unsigned int Mat<R>::number_of_elements() const {
    unsigned int dim = 1;
    for (auto& n : dims())
        dim *= n;
    return dim;
}

template<typename R>
Mat<R> Mat<R>::eltmul_broadcast(Mat<R> matrix2) const {
    return MatOps<R>::eltmul_broadcast(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::eltmul(Mat<R> matrix2) const {
    return MatOps<R>::eltmul(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::eltmul(R alpha) const {
    return MatOps<R>::eltmul(*this, alpha);
}

template<typename R>
Mat<R> Mat<R>::eltmul_broadcast_rowwise(
        Mat<R> row_vector) const {
    return MatOps<R>::eltmul_broadcast_rowwise(*this, row_vector);
}

template<typename R>
Mat<R> Mat<R>::eltmul_rowwise(
        Mat<R> matrix2) const {
    return MatOps<R>::eltmul_rowwise(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::add(
        Mat<R> matrix2) const {
    return MatOps<R>::add(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::sub(
        Mat<R> matrix2) const {
    return MatOps<R>::sub(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::add_broadcast(Mat<R> matrix2) const {
    return MatOps<R>::add_broadcast(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::sub_broadcast(Mat<R> matrix2) const {
    return MatOps<R>::sub_broadcast(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::sub_broadcast_reversed(Mat<R> matrix2) const {
    return MatOps<R>::sub_broadcast_reversed(*this, matrix2);
}

template<typename R>
Mat<R> Mat<R>::square() const {
    return MatOps<R>::square(*this);
}

template<typename R>
Mat<R> Mat<R>::sqrt() const {
    return MatOps<R>::sqrt(*this);
}

template<typename R>
Mat<R> Mat<R>::pow(R power) const {
    return MatOps<R>::pow(*this, power);
}
template<typename R>
Mat<R> Mat<R>::pow(int power) const {
    return MatOps<R>::pow(*this, (R) power);
}

template<typename R>
Mat<R> Mat<R>::elt_inv() const {
    return MatOps<R>::elt_inv(*this);
}

template<typename R>
Mat<R> Mat<R>::sigmoid() const {
    return MatOps<R>::sigmoid(*this);
}

template<typename R>
Mat<R> Mat<R>::steep_sigmoid(R aggressiveness) const {
    return MatOps<R>::steep_sigmoid(*this, aggressiveness);
}

template<typename R>
Mat<R> Mat<R>::sum() const {
    return MatOps<R>::sum(*this);
}

template<typename R>
Mat<R> Mat<R>::mean() const {
    return MatOps<R>::mean(*this);
}

template<typename R>
Mat<R> Mat<R>::log() const {
    return MatOps<R>::log(*this);
}

template<typename R>
Mat<R> Mat<R>::exp() const {
    return MatOps<R>::exp(*this);
}

template<typename R>
Mat<R> Mat<R>::T() const {
    return MatOps<R>::transpose(*this);
}

template<typename R>
Mat<R> Mat<R>::tanh() const {
    return MatOps<R>::tanh(*this);
}

template<typename R>
Mat<R> Mat<R>::relu() const {
    return MatOps<R>::relu(*this);
}

template<typename R>
Mat<R> Mat<R>::mul(Mat<R> other) const {
    return MatOps<R>::mul(*this, other);
}

template<typename R>
Mat<R> Mat<R>::dot(Mat<R> other) const {
    return MatOps<R>::mul(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator[](
        Indexing::Index indices) const {
    return MatOps<R>::rows_pluck(*this, indices);
}
template<typename R>
Mat<R> Mat<R>::operator()(
        Indexing::Index indices) const {
    return MatOps<R>::rows_pluck(*this, indices);
}

template<typename R>
Mat<R> Mat<R>::operator()(
        Indexing::Index row_indices,
        Indexing::Index col_indices) const {
    return MatOps<R>::rows_cols_pluck(*this, row_indices, col_indices);
}

template<typename R>
Mat<R> Mat<R>::operator[](
        int row) const {
    return MatOps<R>::row_pluck(*this, row);
}
template<typename R>
Mat<R> Mat<R>::operator()(
        int row) const {
    return MatOps<R>::row_pluck(*this, row);
}

template<typename R>
Mat<R> Mat<R>::operator()(
        void* nothing,
        int col) const {
    return MatOps<R>::col_pluck(*this, col);
}

template<typename R>
void Mat<R>::npy_save (FILE * fp) {
    std::vector<char> header = cnpy::create_npy_header(w().data(),dims().data(),dims().size());
    fwrite(&header[0],sizeof(char),header.size(),fp);
    fwrite(w().data(),sizeof(R), number_of_elements(), fp);
}

template<typename R>
void Mat<R>::npy_load(cnpy::NpyArray& arr) {
    m = make_shared<MatInternal<R>>(arr.shape[0], arr.shape.size() > 1 ? arr.shape[1] : 1);

    if (arr.word_size == sizeof(double)) {
        double* loaded_data_double = reinterpret_cast<double*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, dims(0), dims(1));
            w() = wrapped_mat_double_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, dims(0), dims(1));
            w() = wrapped_mat_double.cast<R>();
        }
    } else if (arr.word_size == sizeof(float)) {
        float* loaded_data_float = reinterpret_cast<float*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, dims(0), dims(1));
            w() = wrapped_mat_float_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, dims(0), dims(1));
            w() = wrapped_mat_float.cast<R>();
        }
    } else {
        throw std::invalid_argument("Could not load numpy matrix : not recognized as float or double.");
    }
}

template<typename R>
void Mat<R>::npy_load(FILE * fp) {
    auto arr = cnpy::load_the_npy_file(fp);
    npy_load(arr);
    arr.destruct();
}

template<typename R>
void Mat<R>::npy_load(string fname) {
    auto arr = cnpy::npy_load(fname);
    npy_load(arr);
    arr.destruct();
}

template<typename R>
Mat<R>::~Mat() {}

template<typename R>
Mat<R> Mat<R>::Empty(dim_t n, dim_t d) {
    // use an empty matrix and modify
    // it so as to not incur the filling
    // with zeros cost.
    return Mat(n, d, false);
}



template<typename R>
Mat<R> Mat<R>::operator+(Mat<R> other) const {
    return MatOps<R>::add(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator+(R other) const {
    return MatOps<R>::add(*this, other);
}

template<typename R>
Mat<R>& Mat<R>::operator+=(Mat<R> other) {
    auto sum = MatOps<R>::add(*this, other);
    *this = sum;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator+=(R other) {
    auto sum = MatOps<R>::add(*this, other);
    *this = sum;
    return *this;
}

template<typename R>
Mat<R> Mat<R>::operator-(Mat<R> other) const {
    return MatOps<R>::sub(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator-(R other) const {
    return MatOps<R>::add(*this, -other);
}

template<typename R>
Mat<R>& Mat<R>::operator-=(Mat<R> other) {
    auto diff = MatOps<R>::sub(*this, other);
    *this = diff;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator-=(R other) {
    auto diff = MatOps<R>::add(*this, -other);
    *this = diff;
    return *this;
}

template<typename R>
Mat<R> Mat<R>::operator*(Mat<R> other) const {
    return MatOps<R>::eltmul(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator*(R alpha) const {
    return MatOps<R>::eltmul(*this, alpha);
}

template<typename R>
Mat<R>& Mat<R>::operator*=(Mat<R> other) {
    auto prod = MatOps<R>::eltmul(*this, other);
    *this = prod;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator*=(R other) {
    auto prod = MatOps<R>::eltmul(*this, other);
    *this = prod;
    return *this;
}


template<typename R>
Mat<R> Mat<R>::operator-() const {
    return (*this) * -1;
}

template<typename R>
Mat<R> Mat<R>::operator/(Mat<R> other) const {
    return MatOps<R>::eltdivide(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator/(R alpha) const {
    return MatOps<R>::eltdivide(*this, alpha);
}

template<typename R>
Mat<R>& Mat<R>::operator/=(Mat<R> other) {
    auto divided = MatOps<R>::eltdivide(*this, other);
    *this = divided;
    return *this;
}

template<typename R>
Mat<R>& Mat<R>::operator/=(R other) {
    auto divided = MatOps<R>::eltdivide(*this, other);
    *this = divided;
    return *this;
}

template<typename R>
Mat<R> Mat<R>::operator^(R other) const {
    return MatOps<R>::pow(*this, other);
}

template<typename R>
Mat<R> Mat<R>::operator^(int other) const {
    return MatOps<R>::pow(*this, (R) other);
}

template<typename R>
Mat<R> Mat<R>::zeros_like(Mat<R> other) {
    return Mat<R>(other.dims(0), other.dims(1));
}

template<typename R>
Mat<R> Mat<R>::empty_like(Mat<R> other) {
    return Mat<R>(other.dims(0), other.dims(1), false);
}


/* External operators */

template<typename R>
Mat<R> operator+(R other, Mat<R> mat) {
    return MatOps<R>::add(mat, other);
}

template<typename R>
Mat<R> operator-(R other, Mat<R> mat) {
    return MatOps<R>::sub_broadcast_reversed(mat, other);
}

template<typename R>
Mat<R> operator+(int other, Mat<R> mat) {
    return MatOps<R>::add(mat, (R) other);
}

template<typename R>
Mat<R> operator-(int other, Mat<R> mat) {
    return MatOps<R>::sub_broadcast_reversed(mat, (R) other);
}

template<typename R>
Mat<R> operator*(int other, Mat<R> mat) {
    return MatOps<R>::eltmul(mat, (R)other);
}

template<typename R>
Mat<R> operator*(R other, Mat<R> mat) {
    return MatOps<R>::eltmul(mat, other);
}

template Mat<float> operator+(int, Mat<float>);
template Mat<float> operator+(float, Mat<float>);
template Mat<double> operator+(int, Mat<double>);
template Mat<double> operator+(double, Mat<double>);

template Mat<float> operator-(int, Mat<float>);
template Mat<float> operator-(float, Mat<float>);
template Mat<double> operator-(int, Mat<double>);
template Mat<double> operator-(double, Mat<double>);

template Mat<float> operator*(int, Mat<float>);
template Mat<float> operator*(float, Mat<float>);
template Mat<double> operator*(int, Mat<double>);
template Mat<double> operator*(double, Mat<double>);




template<typename R>
std::ostream& operator<<(std::ostream& strm, const Mat<R>& a) {
    if (a.name != nullptr) {
        return strm << "<#Mat name=\"" << *a.name<< "\" n=" << a.dims(0) << ", d=" << a.dims(1) << ">";
    } else {
        return strm << "<#Mat n=" << a.dims(0) << ", d=" << a.dims(1) << ">";
    }
}

template std::ostream& operator<< <double>(std::ostream& strm, const Mat<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Mat<float>& a);

template <typename R>
std::size_t std::hash<Mat<R>>::operator()(const Mat<R>& k) const {
    return k.id();
}

template std::size_t std::hash<Mat<float>>::operator()(const Mat<float>& k)   const;
template std::size_t std::hash<Mat<double>>::operator()(const Mat<double>& k) const;

template <typename R>
bool operator!=(const Mat<R>& A, const Mat<R>& B) {
    return A.id() != B.id();
}

template bool operator!=(const Mat<float>&, const Mat<float>&);
template bool operator!=(const Mat<double>&, const Mat<double>&);

template <typename R>
bool operator==(const Mat<R>& A, const Mat<R>& B) {
    return A.id() == B.id();
}

template bool operator==<float>(const Mat<float>&, const Mat<float>&);
template bool operator==<double>(const Mat<double>&, const Mat<double>&);

template<typename R>
int Mat<R>::argmax() const {
    int i = 0;
    R current_max = -std::numeric_limits<R>::infinity();
    auto ptr = w().data();
    for (int j = 0; j < number_of_elements(); j++) {
        if (*ptr > current_max) {
            current_max = *ptr;
            i = j;
        }
        ptr++;
    }
    return i;
}

template<typename R>
int Mat<R>::argmax_slice(int lower, int upper) const {
    int i = 0;
    R current_max = -std::numeric_limits<R>::infinity();
    auto ptr = w().data();
    for (int j = lower; j < upper; j++) {
        if (*ptr > current_max) {
            current_max = *ptr;
            i = j;
        }
        ptr++;
    }
    return i;
}



namespace utils {
    template<typename R>
    void save_matrices(vector<Mat<R>> parameters, string dirname) {
        utils::ensure_directory(dirname);
        const char * c_dirname = dirname.c_str();
        utils::makedirs(c_dirname);
        int i = 0;
        for (auto& param : parameters) {
            stringstream param_location;
            param_location << dirname << "/param_" << i << ".npy";
            param.npy_save(param_location.str());
            i++;
        }
    }

    template<typename R>
    void load_matrices(vector<Mat<R>> parameters, string dirname) {
        utils::ensure_directory(dirname);
        int i = 0;
        for (auto& param : parameters) {
            stringstream param_location;
            param_location << dirname << "/param_" << i << ".npy";
            param.npy_load(param_location.str());
            i++;
        }
    }

    template <typename T>
    vector<size_t> argsort_rowwise(Mat<T> &m) {
        // initialize original index locations
        vector<size_t> idx(m.dims(0));
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(), [&m](size_t i1, size_t i2) {
            return m.w()(i1) < m.w()(i2);
        });

        return idx;
    }

    template vector<size_t> argsort_rowwise(Mat<float>&);
    template vector<size_t> argsort_rowwise(Mat<double>&);

    template <>
    vector<size_t> argsort(const vector<Mat<float>> &v) {
        // initialize original index locations
        vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1].w()(0) < v[i2].w()(0);});

        return idx;
    }

    template <>
    vector<size_t> argsort(const vector<Mat<double>> &v) {
        // initialize original index locations
        vector<size_t> idx(v.size());
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1].w()(0) < v[i2].w()(0);});

        return idx;
    }

    template void save_matrices(vector<Mat<float> >, string);
    template void save_matrices(vector<Mat<double> >, string);
    template void load_matrices(vector<Mat<float> >, string);
    template void load_matrices(vector<Mat<double> >, string);


    template<typename R>
    json11::Json json_finite_distribution(
        const Mat<R>& probs,
        const vector<string>& labels) {
        assert2(probs.dims(1) == 1, MS() << "Probabilities must be a column vector");
        vector<R> distribution(probs.w().data(), probs.w().data() + probs.dims(0));
        return json11::Json::object {
            { "type", "finite_distribution"},
            { "probabilities", distribution },
            { "labels", labels },
        };
    }

    template json11::Json json_finite_distribution(const Mat<float>&, const vector<string>&);
    template json11::Json json_finite_distribution(const Mat<double>&, const vector<string>&);
}

template class MatInternal<float>;
template class MatInternal<double>;
template class GradInternal<float>;
template class GradInternal<double>;
template class weights<float>;
template class weights<double>;
template class Mat<float>;
template class Mat<double>;
