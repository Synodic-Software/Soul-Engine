#pragma once

#include "SoulParameters.h"
#include <memory>
#include <chrono>

class WindowParameters;
class Window;
class Frame;
class CLIConsoleManager;
class Entity;

class FiberScheduler;
class ComputeBackend;
class Display;
class RasterBackend;

class Soul final{

public:
	friend class CLIConsoleManager;

	Soul(SoulParameters&);
	~Soul();

	Soul(Soul&&) noexcept = delete;
	Soul& operator=(Soul&&) noexcept = delete;

	void Init();
	void CreateWindow(WindowParameters&);

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

	SoulParameters& parameters_;

	std::chrono::nanoseconds frameTime_;
	bool active_;

	//services and modules	
	std::shared_ptr<FiberScheduler> schedulerModule_;
	std::shared_ptr<ComputeBackend> computeModule_;
	std::shared_ptr<Display> displayModule_;
	std::shared_ptr<RasterBackend> rasterModule_;

	//hidden Soul services and modules
	class Implementation;
	std::unique_ptr<Implementation> detail;

};
