#pragma once

#include "FiberProperties.h"
#include "Core/Utility/Types.h"

#include <boost/fiber/algo/algorithm.hpp>
#include <boost/fiber/scheduler.hpp>
#include <boost/fiber/detail/context_spinlock_queue.hpp>
#include <boost/fiber/detail/context_spmc_queue.hpp>

#include <queue>


class SchedulerAlgorithm :
	public boost::fibers::algo::algorithm_with_properties<FiberProperties> {

public:

	typedef boost::fibers::detail::context_spinlock_queue queueType;

	//Construction

	SchedulerAlgorithm(uint, bool, bool = false);
	SchedulerAlgorithm(SchedulerAlgorithm const&) = delete;
	SchedulerAlgorithm(SchedulerAlgorithm &&) = delete;

	SchedulerAlgorithm& operator=(SchedulerAlgorithm const&) = delete;
	SchedulerAlgorithm& operator=(SchedulerAlgorithm &&) = delete;


	//Implementation

	void								awakened(boost::fibers::context*, FiberProperties&)			noexcept override;
	inline boost::fibers::context*		PickFromQueue(queueType&, uint)								noexcept;
	boost::fibers::context*				pick_next()													noexcept override;
	boost::fibers::context*				Steal(uint)													noexcept;
	bool								has_ready_fibers()											const noexcept override;
	void								property_change(boost::fibers::context*, FiberProperties&)  noexcept override;
	void								suspend_until(std::chrono::steady_clock::time_point const&) noexcept override;
	void								notify()													noexcept override;


private:

	static void InitializeSchedulers(uint, std::vector<boost::intrusive_ptr<SchedulerAlgorithm>>&);

	static std::atomic<uint>										counter_;
	static std::vector<boost::intrusive_ptr<SchedulerAlgorithm>>    schedulers_;

	uint                                           id_;
	uint                                           threadCount_;

	queueType readyQueues[6];

	//std::mutex              mainMutex;
	std::mutex              mtx_;
	std::condition_variable cnd_;

	bool                    flag_;
	bool                    suspend_;
	bool					isMain_;

};
