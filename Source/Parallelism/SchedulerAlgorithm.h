#pragma once

#include "FiberProperties.h"

#include <boost/fiber/algo/algorithm.hpp>
#include <boost/fiber/scheduler.hpp>

#include <list>

class SchedulerAlgorithm :
	public boost::fibers::algo::algorithm_with_properties< FiberProperties > {

public:

	//typedefs
	typedef std::list< boost::fibers::context * >  rqueue_t;
	typedef boost::fibers::scheduler::ready_queue_type lqueue_t;


	//Construction

	SchedulerAlgorithm();
	SchedulerAlgorithm(bool);
	SchedulerAlgorithm(SchedulerAlgorithm const&) = delete;
	SchedulerAlgorithm(SchedulerAlgorithm &&) = delete;

	SchedulerAlgorithm& operator=(SchedulerAlgorithm const&) = delete;
	SchedulerAlgorithm& operator=(SchedulerAlgorithm &&) = delete;


	//Implementation

	void						InsertContext(rqueue_t&, boost::fibers::context*&, int);

	void						awakened(boost::fibers::context*, FiberProperties&)			noexcept override;
	boost::fibers::context*		pick_next()													noexcept override;
	bool						has_ready_fibers()											const noexcept override;
	void						property_change(boost::fibers::context*, FiberProperties &) noexcept override;
	void						suspend_until(std::chrono::steady_clock::time_point const&) noexcept override;
	void						notify()													noexcept override;


private:

	static rqueue_t     	readyQueue;
	static rqueue_t     	mainOnlyQueue;
	static std::mutex   	queueMutex;

	lqueue_t            	localQueue;

	std::mutex              mtx_;
	std::condition_variable cnd_;

	bool                    flag_;
	bool                    suspend_;

};
