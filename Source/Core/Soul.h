#pragma once

#include "SoulParameters.h"
#include <memory>
#include "Display/Window/SoulWindow.h"

class Soul {

public:

	Soul(SoulParameters&);
	~Soul();

	Soul(Soul&&) noexcept;
	Soul& operator=(Soul&&) noexcept;

	void Run();
	SoulWindow* CreateWindow(WindowParameters&);

private:

	void Raster();

	SoulParameters& parameters;

	//hidden Soul services and modules
	class Implementation;
	std::unique_ptr<Implementation> detail;

};
