#pragma once

#include "SoulParameters.h"
#include <memory>
#include <chrono>

struct WindowParameters;
class Window;
class ConsoleManager;

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

	bool Poll();

	SoulParameters& parameters;

	using tickType = std::chrono::nanoseconds;
	using clockType = std::chrono::high_resolution_clock;
	tickType frameTime;

	//hidden Soul services and modules
	class Implementation;
	std::unique_ptr<Implementation> detail;

	std::unique_ptr<ConsoleManager> console;
};
