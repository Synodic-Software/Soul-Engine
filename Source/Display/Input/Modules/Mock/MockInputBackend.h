#pragma once

#include "Display/Input/InputModule.h"


class MockInputBackend : public InputModule {

public:

	MockInputBackend() = default;
	~MockInputBackend() override = default;

	MockInputBackend(const MockInputBackend&) = delete;
	MockInputBackend(MockInputBackend&&) = default;

	MockInputBackend& operator=(const MockInputBackend&) = delete;
	MockInputBackend& operator=(MockInputBackend&&) noexcept = default;


	bool Poll() override;


};
