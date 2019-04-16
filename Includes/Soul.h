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
class ComputeModule;
class DisplayModule;
class RasterBackend;
class GUIModule;

class Soul final{

public:
	friend class CLIConsoleManager;

	Soul(SoulParameters&);
	~Soul();

	Soul(const Soul&) = delete;
	Soul(Soul&&) noexcept = default;

	Soul& operator=(const Soul&) = delete;
	Soul& operator=(Soul&&) noexcept = default;

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
	std::shared_ptr<ComputeModule> computeModule_;
	std::shared_ptr<DisplayModule> displayModule_;
	std::shared_ptr<RasterBackend> rasterModule_;
	std::shared_ptr<GUIModule> guiModule_;

	//hidden Soul services and modules
	class Implementation;
	std::unique_ptr<Implementation> detail;

};
