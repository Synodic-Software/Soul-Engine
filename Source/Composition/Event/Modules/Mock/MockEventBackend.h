#pragma once

#include "Composition/Event/EventModule.h"


//signal and slots system
class MockEventBackend final : public EventModule {

public:

	//construction and assignment
	MockEventBackend() = default;
	~MockEventBackend() = default;

	MockEventBackend(const MockEventBackend&) = delete;
	MockEventBackend(MockEventBackend&&) = default;

	MockEventBackend & operator=(const MockEventBackend&) = delete;
	MockEventBackend & operator=(MockEventBackend&&) = default;


};