#ifndef SOLVER_MAT_H
#define SOLVER_MAT_H

#include "dali/tensor/Mat.h"
#include "dali/utils/core_utils.h"

#define SOLVER_MAT_DEFAULT_STEP_SIZE_H 0.035

DECLARE_string(solver);
DECLARE_double(learning_rate);

namespace Solver {
    template<typename R>
    using cache_key_t = void*;
    template<typename R>
    using cache_t = TensorInternal<R, 1>;

    enum Method {
        METHOD_UNINITIALIZED,
        METHOD_ADAGRAD,
        METHOD_ADADELTA,
        METHOD_SGD,
        METHOD_RMSPROP,
        METHOD_ADAM
    };

    const double SMOOTH_DEFAULT = 1e-4;

    template<typename R> class AbstractSolver {
        public:
            Method method;

            R clipval;
            R smooth_eps;
            R regc;
            AbstractSolver();
            AbstractSolver(R clipval, R smooth_eps, R regc, Method method);
            virtual void step( std::vector<Mat<R>>& ) = 0;
            virtual void reset_caches( std::vector<Mat<R>>&);
    };

    template<typename R> class SGD : public AbstractSolver<R> {
        public:
            // This can be overriden by parameter passed to step function.
            R step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;

            SGD (R clipval = 5.0, R regc = 0.0);
            SGD (std::vector<Mat<R>>&, R clipval = 5.0, R regc = 0.0);
            virtual void step( std::vector<Mat<R>>&);
            virtual void step( std::vector<Mat<R>>&, R step_size);
    };

    template<typename R> class AdaGrad : public AbstractSolver<R> {
        public:
            // This can be overriden by parameter passed to step function.
            R step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;

            std::unordered_map<cache_key_t<R>, cache_t<R>> gsums;
            AdaGrad (R smooth_eps = SMOOTH_DEFAULT, R clipval = 100.0, R regc = 0.0);
            AdaGrad (std::vector<Mat<R>>&, R smooth_eps = SMOOTH_DEFAULT, R clipval = 100.0, R regc = 0.0);
            virtual void step( std::vector<Mat<R>>&);
            virtual void step( std::vector<Mat<R>>&, R step_size);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class RMSProp : public AdaGrad<R> {
        public:
            R step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;
            R decay_rate;

            RMSProp (R _decay_rate= 0.999, R smooth_eps = SMOOTH_DEFAULT, R clipval = 100.0, R regc = 0.0);
            RMSProp (std::vector<Mat<R>>&, R _decay_rate= 0.999, R smooth_eps = SMOOTH_DEFAULT, R clipval = 100.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void step(std::vector<Mat<R>>&, R step_size);
    };

    template<typename R> class AdaDelta : public AbstractSolver<R> {
        public:
            R rho;
            std::unordered_map<cache_key_t<R>, cache_t<R>> xsums;
            std::unordered_map<cache_key_t<R>, cache_t<R>> gsums;
            AdaDelta (R rho= 0.95, R smooth_eps = 1e-4, R clipval = 100.0, R regc = 0.0);
            AdaDelta (std::vector<Mat<R>>&, R rho= 0.95, R smooth_eps = 1e-4, R clipval = 100.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class Adam : public AbstractSolver<R> {
        public:
            R b1;
            R b2;
            // This is a large integer:
            unsigned long long epoch;
            std::unordered_map<cache_key_t<R>, cache_t<R>> xsums;
            std::unordered_map<cache_key_t<R>, cache_t<R>> gsums;
            Adam (R b1 = 0.5, R b2 = 1e-6, R smooth_eps = SMOOTH_DEFAULT, R clipval = 100.0, R regc = 0.0);
            Adam (std::vector<Mat<R>>&, R b1 = 0.5, R b2 = 1e-6, R smooth_eps = SMOOTH_DEFAULT, R clipval = 100.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void step(std::vector<Mat<R>>&, R step_size);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R>
    std::shared_ptr<AbstractSolver<R>> construct(std::string solver_name, std::vector<Mat<R>>& params, R learning_rate = 0.01, R regc = 0.0);

}
#endif
