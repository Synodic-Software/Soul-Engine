#include "Soul.h"

#include "System/Platform.h"
#include "Frame/Frame.h"

#include "Parallelism/SchedulerModule.h"
#include "Compute/ComputeModule.h"
#include "Display/DisplayModule.h"
#include "Rasterer/RasterModule.h"
#include "GUI/GUIModule.h"
#include "Transput/Input/InputModule.h"
#include "Composition/Entity/EntityModule.h"
#include "Composition/Event/EventModule.h"



Soul::Soul(SoulParameters& params) :
	parameters_(params),
	frameTime_(),
	active_(true),
	schedulerModule_(SchedulerModule::CreateModule(parameters_.threadCount)),
	computeModule_(ComputeModule::CreateModule()),
	displayModule_(DisplayModule::CreateModule()),
	rasterModule_(RasterModule::CreateModule(schedulerModule_, displayModule_)),
	guiModule_(GUIModule::CreateModule()), 
	inputModule_(InputModule::CreateModule()), 
	entityModule_(EntityModule::CreateModule()), 
	eventModule_(EventModule::CreateModule())
{
	parameters_.engineRefreshRate.AddCallback([this](const int value)
	{
		frameTime_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)) / value;
	});

	//flush parameters_ with new callbacks
	parameters_.engineRefreshRate.Update();
}


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

	inputModule_->Poll();

	//for (auto& scene : scenes) {
	//	scene->Build(engineRefreshRate);
	//}

}

void Soul::EarlyFrameUpdate() {

	eventModule_->Emit("Update"_hashed, "EarlyFrame"_hashed);

}

void Soul::LateFrameUpdate() {

	eventModule_->Emit("Update"_hashed, "LateFrame"_hashed);

}

void Soul::EarlyUpdate() {

	eventModule_->Emit("Update"_hashed, "Early"_hashed);

	//Update the engine cameras
	//RayEngine::Instance().Update();

	//pull cameras into jobs
	eventModule_->Emit("Update"_hashed, "Job Cameras"_hashed);

}

void Soul::LateUpdate() {

	eventModule_->Emit("Update"_hashed, "Late"_hashed);

}


void Soul::Raster() {

	displayModule_->Draw();

}

//returns a bool that is true if the engine is dirty
bool Soul::Poll() {

	return inputModule_->Poll();

}

void Soul::Init()
{
	
	Warmup();

	//TODO: Remove as it is temporary
	Run();

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

		//Execute(frameTime_);

		schedulerModule_->YieldUntil(nextTime);

	}

}
