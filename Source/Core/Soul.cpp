#include "Soul.h"

#include "SoulImplementation.h"
#include "System/Platform.h"

#include "Parallelism/Modules/Fiber/FiberScheduler.h"
#include "Display/Display.h"
#include "Rasterer/RasterBackend.h"

Soul::Soul(SoulParameters& params) :
	parameters_(params),
	frameTime_(),
	active_(true),
	schedulerModule_(Scheduler::CreateModule(parameters_.threadCount)),
	displayModule_(Display::CreateModule()),
	rasterModule_(RasterBackend::CreateModule(schedulerModule_, displayModule_)),
	detail(std::make_unique<Implementation>(*this))
{
	parameters_.engineRefreshRate.AddCallback([this](const int value)
	{
		frameTime_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)) / value;
	});

	//flush parameters_ with new callbacks
	parameters_.engineRefreshRate.Update();
}

//definition to complete PIMPL idiom
Soul::~Soul() = default;


/////////////////////////Core/////////////////////////////////

void Soul::Process(Frame& oldFrame, Frame& newFrame) {

	EarlyFrameUpdate();

	newFrame.Dirty(Poll());

	active_ = displayModule_->Active();

}

void Soul::Update(Frame& oldFrame, Frame& newFrame) {

	EarlyUpdate();

	if (newFrame.Dirty()) {

		//	for (auto& scene : scenes) {
		//		scene->Build(engineRefreshRate);
		//	}
		//	
		//	for (auto const& scene : scenes){
		//		PhysicsEngine::Process(scene);
		//	}

	}

	LateUpdate();

}

void Soul::Render(Frame& oldFrame, Frame& newFrame) {

	LateFrameUpdate();

	if (newFrame.Dirty()) {

		//	//RayEngine::Instance().Process(*scenes[0], engineRefreshRate);

		Raster();

	}

}

void Soul::Warmup() {

	//TODO abstract
	detail->inputManager_->Poll();

	//for (auto& scene : scenes) {
	//	scene->Build(engineRefreshRate);
	//}

}

void Soul::EarlyFrameUpdate() {

	detail->eventManager_.Emit("Update"_hashed, "EarlyFrame"_hashed);

}

void Soul::LateFrameUpdate() {

	detail->eventManager_.Emit("Update"_hashed, "LateFrame"_hashed);

}

void Soul::EarlyUpdate() {

	detail->eventManager_.Emit("Update"_hashed, "Early"_hashed);

	//Update the engine cameras
	//RayEngine::Instance().Update();

	//pull cameras into jobs
	detail->eventManager_.Emit("Update"_hashed, "Job Cameras"_hashed);

}

void Soul::LateUpdate() {

	detail->eventManager_.Emit("Update"_hashed, "Late"_hashed);

}


void Soul::Raster() {

	rasterModule_->Draw();

}

//returns a bool that is true if the engine is dirty
bool Soul::Poll() {

	return detail->inputManager_->Poll();

}

void Soul::Init()
{
	
	Warmup();

	if constexpr (Platform::WithCLI()) {
		FiberParameters fParams(false);
		schedulerModule_->AddTask(fParams, [this]()
		{
			detail->consoleManager_->Poll();
		});
	} else {
		Run();
	}
}

void Soul::CreateWindow(WindowParameters& params) {

	displayModule_->CreateWindow(params, rasterModule_.get());

}


void Soul::Run()
{

	Warmup();

	auto currentTime = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now());
	auto nextTime = currentTime + frameTime_;

	while (active_) {

		currentTime = nextTime;
		nextTime = currentTime + frameTime_;

		detail->framePipeline_.Execute(frameTime_);

		if constexpr (Platform::WithCLI()) std::this_thread::sleep_for(frameTime_);
		else schedulerModule_->YieldUntil(nextTime);

	}

}
