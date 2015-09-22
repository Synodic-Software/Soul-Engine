
//          Copyright Eugene Yakubovich 2014.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <chrono>

#include <boost/asio.hpp>
#include <boost/asio/high_resolution_timer.hpp>
#include <boost/config.hpp>

#include <boost/fiber/all.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace fibers {
namespace asio {

//[timer_handler
inline void timer_handler( boost::asio::high_resolution_timer & timer) {
    boost::this_fiber::yield();
    timer.expires_from_now( boost::fibers::wait_interval() );
    timer.async_wait( std::bind( timer_handler, std::ref( timer) ) );
}
//]

//[run_service
inline void run_service( boost::asio::io_service & io_service) {
    boost::asio::high_resolution_timer timer( io_service, std::chrono::seconds(0) );
    timer.async_wait( std::bind( timer_handler, std::ref( timer) ) );
    io_service.run();
}
//]

}}}

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
