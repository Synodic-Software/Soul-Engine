#pragma once

#include "SoulParameters.h"
#include <memory>
#include <chrono>

struct WindowParameters;
class Window;
class Frame;
class CLIConsoleManager;

class Soul {

public:
	friend class CLIConsoleManager;

	Soul(SoulParameters&);
	~Soul();

	Soul(Soul&&) noexcept = delete;
	Soul& operator=(Soul&&) noexcept = delete;

	void Init();
	Window& CreateWindow(WindowParameters&);

private:

	void Run();

	//Pipeline Functions

	void Process(Frame&, Frame&);
	void Update(Frame&, Frame&);
	void Render(Frame&, Frame&);


	//State Functions

	void Warmup();

	void EarlyFrameUpdate();
	void LateFrameUpdate();
	void EarlyUpdate();
	void LateUpdate();

	void Raster();
	bool Poll();

	SoulParameters& parameters;


	std::chrono::nanoseconds frameTime;

	//hidden Soul services and modules
	class Implementation;
	std::unique_ptr<Implementation> detail;

};
