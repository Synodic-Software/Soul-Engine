#include "SchedulerAlgorithm.h"

//SchedulerAlgorithm::rqueue_t	SchedulerAlgorithm::readyQueue;
//SchedulerAlgorithm::rqueue_t	SchedulerAlgorithm::mainOnlyQueue;
//std::mutex    	  				SchedulerAlgorithm::queueMutex;
//
//SchedulerAlgorithm::SchedulerAlgorithm() :
//	flag_(false),
//	suspend_(false)
//{
//}
//
//SchedulerAlgorithm::SchedulerAlgorithm(bool suspend) :
//	flag_(false),
//	suspend_(suspend) 
//{
//}
//
//void SchedulerAlgorithm::InsertContext(rqueue_t& queue, boost::fibers::context*& ctx, int ctxPriority) {
//	rqueue_t::iterator i(std::find_if(queue.begin(), queue.end(),
//		[ctxPriority, this](boost::fibers::context* c)
//	{ return properties(c).GetPriority() < ctxPriority; }));
//
//	queue.insert(i, ctx);
//}
//
//void SchedulerAlgorithm::awakened(boost::fibers::context * ctx, FiberProperties& props) noexcept {
//
//	//dont push fiber when helper or the main fiber is passed in
//	if (ctx->is_context(boost::fibers::type::pinned_context)) {
//		localQueue.push_back(*ctx);
//	}
//	else {
//		ctx->detach();
//
//		std::unique_lock< std::mutex > lk(queueMutex);
//
//		const int ctxPriority = props.GetPriority();
//
//		//if it needs to run on the main thread
//		if (props.RunOnMain()) {
//			InsertContext(mainOnlyQueue, ctx, ctxPriority);
//		}
//		else {
//			InsertContext(readyQueue, ctx, ctxPriority);
//		}
//	}
//}
//
//boost::fibers::context* SchedulerAlgorithm::pick_next() noexcept {
//
//	boost::fibers::context * ctx(nullptr);
//	const std::thread::id thisID = std::this_thread::get_id();
//
//
//	std::unique_lock< std::mutex > lk(queueMutex);
//
//	//TODO access other queues
//
//	if (!readyQueue.empty()) {
//		ctx = readyQueue.front();
//		readyQueue.pop_front();
//		lk.unlock();
//		BOOST_ASSERT(nullptr != ctx);
//		boost::fibers::context::active()->attach(ctx);
//	}
//	else {
//		lk.unlock();
//
//		if (!localQueue.empty()) {
//			ctx = &localQueue.front();
//			localQueue.pop_front();
//		}
//	}
//	return ctx;
//}
//
//bool SchedulerAlgorithm::has_ready_fibers() const noexcept {
//	std::unique_lock<std::mutex> lock(queueMutex);
//	return !mainOnlyQueue.empty() || !readyQueue.empty() || !localQueue.empty();
//}
//
//void SchedulerAlgorithm::property_change(boost::fibers::context * ctx, FiberProperties & props) noexcept {
//	if (!ctx->ready_is_linked()) {
//		return;
//	}
//	ctx->ready_unlink();
//	awakened(ctx, props);
//}
//
//void SchedulerAlgorithm::suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept {
//	if (suspend_) {
//		if ((std::chrono::steady_clock::time_point::max)() == time_point) {
//			std::unique_lock< std::mutex > lk(mtx_);
//			cnd_.wait(lk, [this]() { return flag_; });
//			flag_ = false;
//		}
//		else {
//			std::unique_lock< std::mutex > lk(mtx_);
//			cnd_.wait_until(lk, time_point, [this]() { return flag_; });
//			flag_ = false;
//		}
//	}
//}
//void SchedulerAlgorithm::notify() noexcept {
//	if (suspend_) {
//		std::unique_lock< std::mutex > lk(mtx_);
//		flag_ = true;
//		lk.unlock();
//		cnd_.notify_all();
//	}
//}