//          Copyright Oliver Kowalke 2013.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_FIBERS_DEFAULT_ROUND_ROBIN_H
#define BOOST_FIBERS_DEFAULT_ROUND_ROBIN_H

#include <cstddef>

#include <boost/config.hpp>

#include <boost/fiber/algorithm.hpp>
#include <boost/fiber/detail/config.hpp>
#include <boost/fiber/detail/fifo.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace fibers {

class context;

class BOOST_FIBERS_DECL round_robin : public sched_algorithm {
private:
    typedef detail::fifo        rqueue_t;

    rqueue_t                    rqueue_;

public:
    virtual void awakened( context *);

    virtual context * pick_next();

    virtual std::size_t ready_fibers() const noexcept;
};

}}

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif

#endif // BOOST_FIBERS_DEFAULT_ROUND_ROBIN_H
