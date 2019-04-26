#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class InputModule;
class WindowModule;


class GUIModule : public Module<GUIModule> {

public:

	GUIModule() = default;
	virtual ~GUIModule() = default;

	GUIModule(const GUIModule&) = delete;
	GUIModule(GUIModule&&) noexcept = default;

	GUIModule& operator=(const GUIModule&) = delete;
	GUIModule& operator=(GUIModule&&) noexcept = default;


	virtual void Update() = 0;
	virtual void Draw() = 0;

	// Factory
	static std::shared_ptr<GUIModule> CreateModule(std::shared_ptr<InputModule>&,
		std::shared_ptr<WindowModule>&);


};

