#pragma once

#include "Soul.h"


class InputModule {

public:

	InputModule() = default;
	virtual ~InputModule() = default;

	InputModule(const InputModule &) = delete;
	InputModule(InputModule&&) noexcept = default;

	InputModule& operator=(const InputModule&) = delete;
	InputModule& operator=(InputModule&&) noexcept = default;


	virtual bool Poll() = 0;


	// Factory
	static std::unique_ptr<InputModule> CreateModule();


};
