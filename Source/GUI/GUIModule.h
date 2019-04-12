#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>


class GUIModule : public Module<GUIModule> {

public:

	GUIModule() = default;
	virtual ~GUIModule() = default;

	GUIModule(const GUIModule&) = delete;
	GUIModule(GUIModule&&) noexcept = default;

	GUIModule& operator=(const GUIModule&) = delete;
	GUIModule& operator=(GUIModule&&) noexcept = default;

	// Factory
	static std::unique_ptr<GUIModule> CreateModule();


};

