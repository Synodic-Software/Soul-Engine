
//          Copyright Oliver Kowalke 2013.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
//          based on tss.hpp from boost.thread

#ifndef BOOST_FIBERS_FSS_H
#define BOOST_FIBERS_FSS_H

#include <cstddef>

#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/intrusive_ptr.hpp>

#include <boost/fiber/detail/fss.hpp>
#include <boost/fiber/context.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace fibers {

template< typename T >
class fiber_specific_ptr {
private:
    struct default_cleanup_function : public detail::fss_cleanup_function {
        void operator()( void * data) {
            delete static_cast< T * >( data);
        }
    };

    struct custom_cleanup_function : public detail::fss_cleanup_function {
        void (*fn)(T*);

        explicit custom_cleanup_function( void(*fn_)(T*) ):
            fn( fn_) {
        }

        void operator()( void* data) {
            if ( fn) {
                fn( static_cast< T * >( data) );
            }
        }
    };

    detail::fss_cleanup_function::ptr_t cleanup_fn_;

public:
    typedef T   element_type;

    fiber_specific_ptr() :
        cleanup_fn_( new default_cleanup_function() ) {
    }

    explicit fiber_specific_ptr( void(*fn)(T*) ) :
        cleanup_fn_( new custom_cleanup_function( fn) ) {
    }

    ~fiber_specific_ptr() {
        context * f( context::active() );
        if ( nullptr != f) {
            f->set_fss_data(
                this, cleanup_fn_, nullptr, true);
        }
    }

    fiber_specific_ptr( fiber_specific_ptr const&) = delete;
    fiber_specific_ptr & operator=( fiber_specific_ptr const&) = delete;

    T * get() const {
        BOOST_ASSERT( context::active() );

        void * vp( context::active()->get_fss_data( this) );
        return static_cast< T * >( vp);
    }

    T * operator->() const {
        return get();
    }

    T & operator*() const {
        return * get();
    }

    T * release() {
        T * tmp = get();
        context::active()->set_fss_data(
            this, cleanup_fn_, nullptr, false);
        return tmp;
    }

    void reset( T * t) {
        T * c = get();
        if ( c != t) {
            context::active()->set_fss_data(
                this, cleanup_fn_, t, true);
        }
    }
};

}}

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif

#endif //  BOOST_FIBERS_FSS_H
