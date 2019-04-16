#pragma once

#include "Core/Interface/Module/Module.h"

#include "Types.h"
#include "Property.h"

#include <memory>


class SchedulerModule : public Module<SchedulerModule> {

public:

	SchedulerModule() = default;
	virtual ~SchedulerModule() = default;

	SchedulerModule(const SchedulerModule &) = delete;
	SchedulerModule(SchedulerModule &&) noexcept = default;

	SchedulerModule& operator=(const SchedulerModule &) = delete;
	SchedulerModule& operator=(SchedulerModule &&) noexcept = default;


	//Factory
	static std::shared_ptr<SchedulerModule> CreateModule(Property<uint>&);


};
