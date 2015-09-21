//          Copyright Oliver Kowalke 2013.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_FIBERS_ALGORITHM_H
#define BOOST_FIBERS_ALGORITHM_H

#include <cstddef>

#include <boost/config.hpp>
#include <boost/assert.hpp>

#include <boost/fiber/properties.hpp>
#include <boost/fiber/detail/config.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace fibers {

class context;

struct BOOST_FIBERS_DECL sched_algorithm {
    virtual ~sched_algorithm() {}

    virtual void awakened( context *) = 0;

    virtual context * pick_next() = 0;

    virtual std::size_t ready_fibers() const noexcept = 0;
};

class BOOST_FIBERS_DECL sched_algorithm_with_properties_base : public sched_algorithm {
public:
    // called by fiber_properties::notify() -- don't directly call
    virtual void property_change_( context * f, fiber_properties * props) = 0;

protected:
    static fiber_properties* get_properties( context * f) noexcept;
    static void set_properties( context * f, fiber_properties * p) noexcept;
};

template< typename PROPS >
struct sched_algorithm_with_properties : public sched_algorithm_with_properties_base {
    typedef sched_algorithm_with_properties_base super;

    // Mark this override 'final': sched_algorithm_with_properties subclasses
    // must override awakened() with properties parameter instead. Otherwise
    // you'd have to remember to start every subclass awakened() override
    // with: sched_algorithm_with_properties<PROPS>::awakened(fb);
    virtual void awakened( context * f) final {
        fiber_properties * props = super::get_properties( f);
        if ( ! props) {
            // TODO: would be great if PROPS could be allocated on the new
            // fiber's stack somehow
            props = new_properties( f);
            // It is not good for new_properties() to return 0.
            BOOST_ASSERT_MSG(props, "new_properties() must return non-NULL");
            // new_properties() must return instance of (a subclass of) PROPS
            BOOST_ASSERT_MSG(dynamic_cast<PROPS*>(props),
                             "new_properties() must return properties class");
            super::set_properties( f, props);
        }
        // Set sched_algo_ again every time this fiber becomes READY. That
        // handles the case of a fiber migrating to a new thread with a new
        // sched_algorithm subclass instance.
        props->set_sched_algorithm( this);

        // Okay, now forward the call to subclass override.
        awakened( f, properties(f) );
    }

    // subclasses override this method instead of the original awakened()
    virtual void awakened( context *, PROPS& ) = 0;

    // used for all internal calls
    PROPS& properties( context * f) {
        return static_cast< PROPS & >( * super::get_properties( f) );
    }

    // override this to be notified by PROPS::notify()
    virtual void property_change( context * f, PROPS & props) {
    }

    // implementation for sched_algorithm_with_properties_base method
    void property_change_( context * f, fiber_properties * props ) final {
        property_change( f, * static_cast< PROPS * >( props) );
    }

    // Override this to customize instantiation of PROPS, e.g. use a different
    // allocator. Each PROPS instance is associated with a particular
    // context.
    virtual fiber_properties * new_properties( context * f) {
        return new PROPS( f);
    }
};

}}

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif

#endif // BOOST_FIBERS_ALGORITHM_H
