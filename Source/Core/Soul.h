#pragma once

#include "SoulParameters.h"
#include <memory>
#include "Display/Window/Window.h"

class Soul {

public:

	Soul(SoulParameters&);
	~Soul();

	Soul(Soul&&) noexcept = delete;
	Soul& operator=(Soul&&) noexcept = delete;

	void Run();
	Window& CreateWindow(WindowParameters&);

private:

	void Raster();
	void Warmup();

	void EarlyFrameUpdate();
	void LateFrameUpdate();
	void EarlyUpdate();
	void LateUpdate();

	SoulParameters& parameters;

	//hidden Soul services and modules
	class Implementation;
	std::unique_ptr<Implementation> detail;

};
