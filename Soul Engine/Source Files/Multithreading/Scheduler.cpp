#include "Scheduler.h"
#include "boost\fiber\all.hpp"
#include <boost/assert.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <queue>
#include <string>

//taken from boost/fiber github

class shared_ready_queue : public boost::fibers::sched_algorithm {
private:
	typedef std::queue<boost::fibers::context*> rqueue_t;
	// The important point about this ready queue is that it's a class static,
	// common to all instances of shared_ready_queue.
	static rqueue_t                    rqueue_;

	// so is this mutex
	static std::mutex                  mutex_;
	typedef std::unique_lock<std::mutex> lock_t;

public:
	virtual void awakened(boost::fibers::context * f) {
		BOOST_ASSERT(nullptr != f);

		lock_t lock(mutex_);
		rqueue_.push(f);
	}

	virtual boost::fibers::context * pick_next() {
		lock_t lock(mutex_);
		boost::fibers::context * victim(nullptr);
		if (!rqueue_.empty()) {
			victim = rqueue_.front();
			rqueue_.pop();
			BOOST_ASSERT(nullptr != victim);
		}
		return victim;
	}

	virtual std::size_t ready_fibers() const noexcept{
		lock_t lock(mutex_);
		return rqueue_.size();
	}
};

shared_ready_queue::rqueue_t shared_ready_queue::rqueue_;
std::mutex shared_ready_queue::mutex_;

/*****************************************************************************
*   example thread function
*****************************************************************************/
// Wait until all running fibers have completed. This works because we happen
// to know that all example fibers use yield(), which leaves them in ready
// state. A fiber blocked on a synchronization object is invisible to
// ready_fibers().
void drain() {
	// THIS fiber is running, so won't be counted among "ready" fibers
	while (boost::fibers::ready_fibers()) {
		boost::this_fiber::yield();
	}
}

void thread() {
	boost::fibers::use_scheduling_algorithm<shared_ready_queue>();
	drain();
}

/*****************************************************************************
*   example fiber function
*****************************************************************************/
void whatevah(char me) {
	std::thread::id my_thread = std::this_thread::get_id();
	{
		std::ostringstream buffer;
		buffer << "fiber " << me << " started on thread " << my_thread << '\n';
		std::cout << buffer.str() << std::flush;
	}
	for (unsigned i = 0; i < 5; ++i) {
		boost::this_fiber::yield();
		std::thread::id new_thread = std::this_thread::get_id();
		if (new_thread != my_thread) {
			my_thread = new_thread;
			std::ostringstream buffer;
			buffer << "fiber " << me << " switched to thread " << my_thread << '\n';
			std::cout << buffer.str() << std::flush;
		}
	}
}

void Scheduler::Initialize(){
	boost::fibers::use_scheduling_algorithm<shared_ready_queue>();

	// launch a number of fibers
	for (char c : "abcdefghijklmno") {
		boost::fibers::fiber([c](){ whatevah(c); }).detach();
	}

	// launch a couple threads to help process them
	std::thread threads[] = {
		std::thread(thread),
		std::thread(thread),
		std::thread(thread)
	};
	// drain running fibers
	drain();

	// wait for threads to terminate
	for (std::thread& t : threads) {
		t.join();
	}
}