#ifndef DALI_UTILS_SCOPE_H
#define DALI_UTILS_SCOPE_H

#include "dali/utils/observer.h"

#include <functional>
#include <memory>
#include <vector>


struct Scope {
    typedef std::shared_ptr<std::string> name_t;

    static Observation<name_t> enter;
    static Observation<name_t> exit;

    name_t name;

    Scope(name_t name_);
    ~Scope();
};

// TODO(szymon): make thread safe - might be as simple as making state thread_local.
struct ScopeObserver {
    struct State;
    typedef std::function<void(const ScopeObserver::State&)> callback_t;

    struct State {
        std::vector<Scope::name_t> trace;
    };
    ScopeObserver(callback_t on_enter_, callback_t on_exit_);
  private:

    decltype(Scope::enter)::guard_t enter_guard;
    decltype(Scope::exit)::guard_t  exit_guard;

    const callback_t on_enter;
    const callback_t on_exit;

    State state;

    void on_enter_wrapper(Scope::name_t name);

    void on_exit_wrapper(Scope::name_t name);
};

#endif  // DALI_UTILS_SCOPE_H
