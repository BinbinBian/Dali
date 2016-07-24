#include "debug.h"

#include "dali/array/array.h"

using namespace std::placeholders;

namespace debug {
    Observation<Array> lazy_evaluation_callback;
    Observation<Array> array_as_contiguous;


    Observation<Scope::name_t> Scope::enter;
    Observation<Scope::name_t> Scope::exit;

    Scope::Scope(name_t name_) : name(name_) {
        enter.notify(name); // TODO make macro
    }

    Scope::~Scope() {
        exit.notify(name);
    }

    ScopeObserver::ScopeObserver(callback_t on_enter_, callback_t on_exit_) :
            on_enter(on_enter_),
            on_exit(on_exit_),
            enter_guard(std::bind(&ScopeObserver::on_enter_wrapper, this, _1), &Scope::enter),
            exit_guard(std::bind(&ScopeObserver::on_exit_wrapper, this, _1),   &Scope::exit) {
    }

    void ScopeObserver::on_enter_wrapper(Scope::name_t name) {
        state.trace.emplace_back(name);
        if ((bool)on_enter) {
            on_enter(state);
        }
    }

    void ScopeObserver::on_exit_wrapper(Scope::name_t name) {
        if ((bool)on_exit) {
            on_exit(state);
        }
        ASSERT2(*(state.trace.back()) == *name,
                "Scope exit called out of order.");
        state.trace.pop_back();
    }
}  // namespace debug
