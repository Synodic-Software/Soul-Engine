#include "SchedulerAlgorithm.h"

#include <boost/context/detail/prefetch.hpp>
#include <boost/fiber/mutex.hpp>

std::atomic<uint> SchedulerAlgorithm::counter_(0);
std::vector<boost::intrusive_ptr<SchedulerAlgorithm>> SchedulerAlgorithm::schedulers_;

SchedulerAlgorithm::SchedulerAlgorithm(uint threadCount, bool isMain, bool suspend) :
	id_(counter_++),
	threadCount_(threadCount),
	flag_(false),
	suspend_(suspend),
	isMain_(isMain)
{

	//only initialize schedulers once
	static std::once_flag flag;
	std::call_once(flag, &SchedulerAlgorithm::InitializeSchedulers, threadCount_, std::ref(schedulers_));

	// register this scheduler
	schedulers_[id_] = this;

}

void SchedulerAlgorithm::InitializeSchedulers(uint thread_count,
	std::vector<boost::intrusive_ptr<SchedulerAlgorithm>>& schedulers) {

	schedulers_.resize(thread_count, nullptr);

}

//called when a newly launched, blocked, or yielded fiber wakes up. Because the readyQueue is work stealing, 
//we cant push to it as it may be stolen before properties are set
void SchedulerAlgorithm::awakened(boost::fibers::context * ctx, FiberProperties& props) noexcept {

	if (!ctx->is_context(boost::fibers::type::pinned_context)) {
		ctx->detach();
	}

	if (props.RunOnMain()) {

		switch (props.GetPriority()) {
		case FiberPriority::LOW:
			readyQueues[5].push(ctx);
			break;
		case FiberPriority::HIGH:
			readyQueues[4].push(ctx);
			break;
		case FiberPriority::UX:
			readyQueues[3].push(ctx);
			break;
		}

	}
	else {

		switch (props.GetPriority()) {
		case FiberPriority::LOW:
			readyQueues[2].push(ctx);
			break;
		case FiberPriority::HIGH:
			readyQueues[1].push(ctx);
			break;
		case FiberPriority::UX:
			readyQueues[0].push(ctx);
			break;
		}

	}


}

boost::fibers::context* SchedulerAlgorithm::PickFromQueue(queueType& queue, uint index) noexcept {

	boost::fibers::context* victim = queue.pop();

	if (victim != nullptr) {
		boost::context::detail::prefetch_range(victim, sizeof(boost::fibers::context));
		if (!victim->is_context(boost::fibers::type::pinned_context)) {
			boost::fibers::context::active()->attach(victim);
		}
	}
	else {

		uint id;
		uint count = 0;
		const uint size = schedulers_.size();

		static thread_local std::minstd_rand generator{
			std::random_device{}()
		};

		const std::uniform_int_distribution<uint> distribution{
			0, static_cast<uint>(threadCount_ - 1)
		};

		do {

			do {
				++count;

				// random selection of one logical cpu
				id = distribution(generator);

				// prevent stealing from own scheduler
			} while (id == id_);

			//steal context from other scheduler
			if (schedulers_[id]) {
				victim = schedulers_[id]->Steal(index);
			}

		} while (nullptr == victim && count < size);

		if (nullptr != victim) {
			boost::context::detail::prefetch_range(victim, sizeof(boost::fibers::context));
			BOOST_ASSERT(!victim->is_context(boost::fibers::type::pinned_context));
			boost::fibers::context::active()->attach(victim);
		}

	}

	return victim;

}

//picks the fiber to run next
boost::fibers::context* SchedulerAlgorithm::pick_next() noexcept {

	boost::fibers::context* victim = nullptr;

	if (isMain_) {
		for (auto i = 3; i < 6 && !victim; ++i) {
			victim = PickFromQueue(readyQueues[i], i);
		}
	}

	for (auto i = 0; i < 3 && !victim; ++i) {
		victim = PickFromQueue(readyQueues[i], i);
	}

	return victim;

}

boost::fibers::context* SchedulerAlgorithm::Steal(uint index) noexcept {
	return readyQueues[index].steal();
}

bool SchedulerAlgorithm::has_ready_fibers() const noexcept {

	if (isMain_) {
		return !readyQueues[0].empty() && !readyQueues[1].empty() && !readyQueues[2].empty() &&
			!readyQueues[3].empty() && !readyQueues[4].empty() && !readyQueues[5].empty();
	}

	return !readyQueues[0].empty() && !readyQueues[1].empty() && !readyQueues[2].empty();
}

//on a property change that requires the fiber to migrate queues or reorder itself...
void SchedulerAlgorithm::property_change(boost::fibers::context* ctx, FiberProperties& props) noexcept {
	if (!ctx->ready_is_linked()) {
		return;
	}
	ctx->ready_unlink();
	awakened(ctx, props);
}

void SchedulerAlgorithm::suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept {
	if (suspend_) {
		if ((std::chrono::steady_clock::time_point::max)() == time_point) {
			std::unique_lock< std::mutex > lk(mtx_);
			cnd_.wait(lk, [this]() { return flag_; });
			flag_ = false;
		}
		else {
			std::unique_lock< std::mutex > lk(mtx_);
			cnd_.wait_until(lk, time_point, [this]() { return flag_; });
			flag_ = false;
		}
	}
}
void SchedulerAlgorithm::notify() noexcept {
	if (suspend_) {
		std::unique_lock< std::mutex > lk(mtx_);
		flag_ = true;
		lk.unlock();
		cnd_.notify_all();
	}
}