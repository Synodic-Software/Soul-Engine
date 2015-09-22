
//          Copyright Oliver Kowalke 2013.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_FIBERS_CONTEXT_H
#define BOOST_FIBERS_CONTEXT_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/context/all.hpp>

#include <boost/fiber/detail/config.hpp>
#include <boost/fiber/detail/fss.hpp>
#include <boost/fiber/detail/invoke.hpp>
#include <boost/fiber/detail/spinlock.hpp>
#include <boost/fiber/scheduler.hpp>
#include <boost/fiber/exceptions.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace fibers {

class fiber;
class fiber_properties;

class BOOST_FIBERS_DECL context {
private:
    enum class fiber_status {
        ready = 0,
        running,
        waiting,
        terminated
    };

    enum flag_t {
        flag_main_context           = 1 << 1,
        flag_interruption_blocked   = 1 << 2,
        flag_interruption_requested = 1 << 3,
        flag_detached               = 1 << 4
    };

    struct BOOST_FIBERS_DECL fss_data {
        void                                *   vp;
        detail::fss_cleanup_function::ptr_t     cleanup_function;

        fss_data() :
            vp( nullptr),
            cleanup_function() {
        }

        fss_data( void * vp_,
                  detail::fss_cleanup_function::ptr_t const& fn) :
            vp( vp_),
            cleanup_function( fn) {
            BOOST_ASSERT( cleanup_function);
        }

        void do_cleanup() {
            ( * cleanup_function)( vp);
        }
    };

    typedef std::map< uintptr_t, fss_data >   fss_data_t;

    static thread_local context           *   active_;

#if ! defined(BOOST_FIBERS_NO_ATOMICS)
    std::atomic< std::size_t >                  use_count_;
    std::atomic< fiber_status >                 state_;
    std::atomic< int >                          flags_;
#else
    std::size_t                                 use_count_;
    fiber_status                                state_;
    int                                         flags_;
#endif
    detail::spinlock                            splk_;
    scheduler                               *   scheduler_;
    boost::context::execution_context           ctx_;
    fss_data_t                                  fss_data_;
    std::vector< context * >                    waiting_;
    std::exception_ptr                          except_;
    std::chrono::steady_clock::time_point       tp_;
    fiber_properties                        *   properties_;

protected:
    virtual void deallocate() {
    }

public:
    class id {
    private:
        context  *   impl_;

    public:
        id() noexcept :
            impl_( nullptr) {
        }

        explicit id( context * impl) noexcept :
            impl_( impl) {
        }

        bool operator==( id const& other) const noexcept {
            return impl_ == other.impl_;
        }

        bool operator!=( id const& other) const noexcept {
            return impl_ != other.impl_;
        }
        
        bool operator<( id const& other) const noexcept {
            return impl_ < other.impl_;
        }
        
        bool operator>( id const& other) const noexcept {
            return other.impl_ < impl_;
        }
        
        bool operator<=( id const& other) const noexcept {
            return ! ( * this > other);
        }
        
        bool operator>=( id const& other) const noexcept {
            return ! ( * this < other);
        }

        template< typename charT, class traitsT >
        friend std::basic_ostream< charT, traitsT > &
        operator<<( std::basic_ostream< charT, traitsT > & os, id const& other) {
            if ( nullptr != other.impl_) {
                return os << other.impl_;
            } else {
                return os << "{not-valid}";
            }
        }

        explicit operator bool() const noexcept {
            return nullptr != impl_;
        }

        bool operator!() const noexcept {
            return nullptr == impl_;
        }
    };

    static context * active() noexcept;

    static context * active( context * active) noexcept;

    context   *   nxt;

    // main fiber
    context() :
        use_count_( 1), // allocated on stack
        state_( fiber_status::running),
        flags_( flag_main_context),
        splk_(),
        scheduler_( nullptr),
        ctx_( boost::context::execution_context::current() ),
        fss_data_(),
        waiting_(),
        except_(),
        tp_( (std::chrono::steady_clock::time_point::max)() ),
        properties_( nullptr),
        nxt( nullptr) {
    }

    // worker fiber
    template< typename StackAlloc, typename Fn, typename ... Args >
    context( boost::context::preallocated palloc,
                   StackAlloc salloc,
                   Fn && fn,
                   Args && ... args) :
        use_count_( 1), // allocated on stack
        state_( fiber_status::ready),
        flags_( 0),
        splk_(),
        scheduler_( nullptr),
        ctx_( palloc, salloc,
              // lambda, executed in execution context
              // mutable: generated operator() is not const -> enables std::move( fn)
              // std::make_tuple: stores decayed copies of its args, implicitly unwraps std::reference_wrapper
              [=,fn=std::forward< Fn >( fn),tpl=std::make_tuple( std::forward< Args >( args) ...)] () mutable -> void {
                try {
                    BOOST_ASSERT( is_running() );
                    detail::invoke_helper( std::move( fn), std::move( tpl) );
                    BOOST_ASSERT( is_running() );
                } catch( fiber_interrupted const&) {
                    except_ = std::current_exception();
                } catch( ... ) {
                    std::terminate();
                }

                // mark fiber as terminated
                set_terminated();

                // notify waiting (joining) fibers
                release();

                // switch to another fiber
                do_schedule();

                BOOST_ASSERT_MSG( false, "fiber already terminated");
              }),
        fss_data_(),
        waiting_(),
        except_(),
        tp_( (std::chrono::steady_clock::time_point::max)() ),
        properties_( nullptr),
        nxt( nullptr) {
    }

    virtual ~context();

    void set_scheduler( scheduler * mgr) {
        BOOST_ASSERT( nullptr != mgr);
        scheduler_ = mgr;
    }

    scheduler * get_scheduler() const noexcept {
        return scheduler_;
    }

    id get_id() const noexcept {
        return id( const_cast< context * >( this) );
    }

    bool join( context *);

    bool interruption_blocked() const noexcept {
        return 0 != ( flags_ & flag_interruption_blocked);
    }

    void interruption_blocked( bool blck) noexcept;

    bool interruption_requested() const noexcept {
        return 0 != ( flags_ & flag_interruption_requested);
    }

    void request_interruption( bool req) noexcept;

    bool is_main_context() const noexcept {
        return 0 != ( flags_ & flag_main_context);
    }

    bool is_terminated() const noexcept {
        return fiber_status::terminated == state_;
    }

    bool is_ready() const noexcept {
        return fiber_status::ready == state_;
    }

    bool is_running() const noexcept {
        return fiber_status::running == state_;
    }

    bool is_waiting() const noexcept {
        return fiber_status::waiting == state_;
    }

    void set_terminated() noexcept {
#if ! defined(BOOST_FIBERS_NO_ATOMICS)
        fiber_status previous = state_.exchange( fiber_status::terminated);
#else
        fiber_status previous = state_;
        state_ = fiber_status::terminated;
#endif
        BOOST_ASSERT( fiber_status::running == previous);
        (void)previous;
    }

    void set_ready() noexcept {
#if ! defined(BOOST_FIBERS_NO_ATOMICS)
        fiber_status previous = state_.exchange( fiber_status::ready);
#else
        fiber_status previous = state_;
        state_ = fiber_status::ready;
#endif
        BOOST_ASSERT( fiber_status::waiting == previous || fiber_status::running == previous || fiber_status::ready == previous);
        (void)previous;
    }

    void set_running() noexcept {
#if ! defined(BOOST_FIBERS_NO_ATOMICS)
        fiber_status previous = state_.exchange( fiber_status::running);
#else
        fiber_status previous = state_;
        state_ = fiber_status::running;
#endif
        BOOST_ASSERT( fiber_status::ready == previous);
        (void)previous;
    }

    void set_waiting() noexcept {
#if ! defined(BOOST_FIBERS_NO_ATOMICS)
        fiber_status previous = state_.exchange( fiber_status::waiting);
#else
        fiber_status previous = state_;
        state_ = fiber_status::waiting;
#endif
        BOOST_ASSERT( fiber_status::running == previous);
        (void)previous;
    }

    void * get_fss_data( void const * vp) const;

    void set_fss_data(
        void const * vp,
        detail::fss_cleanup_function::ptr_t const& cleanup_fn,
        void * data,
        bool cleanup_existing);

    std::exception_ptr get_exception() const noexcept {
        return except_;
    }

    void set_exception( std::exception_ptr except) noexcept {
        except_ = except;
    }

    void resume() {
        BOOST_ASSERT( is_running() ); // set by the scheduler-algorithm

        ctx_();
    }

    std::chrono::steady_clock::time_point const& time_point() const noexcept {
        return tp_;
    }

    void time_point( std::chrono::steady_clock::time_point const& tp) {
        tp_ = tp;
    }

    void time_point_reset() {
        tp_ = (std::chrono::steady_clock::time_point::max)();
    }

    void set_properties( fiber_properties* props);

    fiber_properties* get_properties() const {
        return properties_;
    }

    void release();

    void do_spawn( fiber const&);

    void do_schedule();

    void do_wait( detail::spinlock_lock &);

    template< typename Clock, typename Duration >
    bool do_wait_until( std::chrono::time_point< Clock, Duration > const& timeout_time,
                        detail::spinlock_lock & lk) {
        return scheduler_->wait_until( timeout_time, lk);
    }

    template< typename Rep, typename Period >
    bool do_wait_for( std::chrono::duration< Rep, Period > const& timeout_duration,
                      detail::spinlock_lock & lk) {
        return scheduler_->wait_for( timeout_duration, lk);
    }

    void do_yield();

    void do_join( context *);

    std::size_t do_ready_fibers() const noexcept;

    void do_set_sched_algo( std::unique_ptr< sched_algorithm >);

    template< typename Rep, typename Period >
    void do_wait_interval( std::chrono::duration< Rep, Period > const& wait_interval) noexcept {
        scheduler_->wait_interval( wait_interval);
    }

    std::chrono::steady_clock::duration do_wait_interval() noexcept;

    friend void intrusive_ptr_add_ref( context * f) {
        BOOST_ASSERT( nullptr != f);
        ++f->use_count_;
    }

    friend void intrusive_ptr_release( context * f) {
        BOOST_ASSERT( nullptr != f);
        if ( 0 == --f->use_count_) {
            BOOST_ASSERT( f->is_terminated() );
            f->~context();
        }
    }
};

}}

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif

#endif // BOOST_FIBERS_CONTEXT_H
