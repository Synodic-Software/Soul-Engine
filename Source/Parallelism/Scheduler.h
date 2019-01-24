#pragma once

#include "Core/Interface/Module/Module.h"

#include "Core/Utility/Types.h"
#include "Core/Utility/Property/Property.h"

#include <memory>

class FiberScheduler;

class Scheduler : public Module {

public:

	Scheduler() = default;
	virtual ~Scheduler() = default;

	Scheduler(const Scheduler &) = delete;
	Scheduler(Scheduler &&) noexcept = default;

	Scheduler& operator=(const Scheduler &) = delete;
	Scheduler& operator=(Scheduler &&) noexcept = default;


	//Factory
	static std::shared_ptr<FiberScheduler> CreateModule(Property<uint>&);


};
