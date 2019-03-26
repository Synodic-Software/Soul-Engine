#pragma once

#include "Parallelism/Scheduler.h"


class MockScheduler final : public Scheduler {

public:

	~MockScheduler() override = default;

	MockScheduler(const MockScheduler&) = delete;
	MockScheduler(MockScheduler&&) noexcept = default;

	MockScheduler& operator=(const MockScheduler&) = delete;
	MockScheduler& operator=(MockScheduler&&) noexcept = default;


};
