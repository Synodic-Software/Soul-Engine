#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class DisplayWindowAPI : public Module {

public:

	virtual ~DisplayWindowAPI() = default;

	static std::shared_ptr<DisplayWindowAPI> CreateModule();


};
