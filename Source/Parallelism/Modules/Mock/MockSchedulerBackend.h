#pragma once

#include "Parallelism/SchedulerModule.h"


class MockScheduler final : public SchedulerModule {

public:

	~MockScheduler() override = default;

	MockScheduler(const MockScheduler&) = delete;
	MockScheduler(MockScheduler&&) noexcept = default;

	MockScheduler& operator=(const MockScheduler&) = delete;
	MockScheduler& operator=(MockScheduler&&) noexcept = default;


};
