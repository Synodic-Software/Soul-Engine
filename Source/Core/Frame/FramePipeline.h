#pragma once

#include "Frame.h"
#include "Core/Structures/RingBuffer.h"
#include "Parallelism/Graph/Graph.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Core/Utility/Function/function_ref.h"

#include <chrono>

class Soul;

template <std::size_t N>
class FramePipeline {

public:

	FramePipeline(Scheduler&, std::array<std::function<void(Frame&, Frame&)>, N>&&);
	~FramePipeline() = default;

	FramePipeline(const FramePipeline&) = delete;
	FramePipeline(FramePipeline&&) noexcept = delete;

	FramePipeline& operator=(const FramePipeline&) = delete;
	FramePipeline& operator=(FramePipeline&&) noexcept = delete;

	void Execute(std::chrono::nanoseconds);

private:

	Scheduler& scheduler_;
	Graph& graph_;
	RingBuffer<Frame, N> frames_;

};

template <std::size_t N>
FramePipeline<N>::FramePipeline(Scheduler& scheduler,std::array<std::function<void(Frame&, Frame&)>, N>&& tasks) :
	scheduler_(scheduler),
	graph_(scheduler.CreateGraph())
{

	Task* oldTask = nullptr;

	auto index = 0;

	for (auto& task : tasks) {

		auto voidWrapper = [this, task, index]() mutable
		{
			//0 is newest
			task(frames_[index+1], frames_[index]);
		};

		Task& newTask = graph_.AddTask(voidWrapper);

		if (oldTask) {
			newTask.DependsOn(*oldTask);
		}

		oldTask = &newTask;
		++index;

	}

}

template <std::size_t N>
void FramePipeline<N>::Execute(std::chrono::nanoseconds frameTime) {

	//iterate the next frame. Index corresponds to pipeline pos 0 == first stage
	frames_.Push(Frame());

	graph_.Execute(frameTime);

	scheduler_.Block();

}
