#pragma once

#include "Parallelism/Compute/ComputeBackend.h"


class MockBackend : public ComputeBackend<MockBackend> {

public:

	MockBackend() = default;
	virtual ~MockBackend() = default;

	MockBackend(const MockBackend&) = delete;
	MockBackend(MockBackend&&) noexcept = default;

	MockBackend& operator=(const MockBackend&) = delete;
	MockBackend& operator=(MockBackend&&) noexcept = default;


};
