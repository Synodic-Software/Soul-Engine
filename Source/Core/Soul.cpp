#include "Soul.h"

#include "SoulImplementation.h"
#include "System/Platform.h"

Soul::Soul(SoulParameters& params) :
	parameters(params),
	frameTime(),
	detail(std::make_unique<Implementation>(*this))
{
	parameters.engineRefreshRate.AddCallback([this](int value)
	{
		frameTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)) / value;
	});

	//flush parameters with new callbacks
	parameters.engineRefreshRate.Update();
}

//definition to complete PIMPL idiom
Soul::~Soul() = default;

//////////////////////Synchronization///////////////////////////


void SynchCPU() {
	//Scheduler::Block(); //TODO Implement MT calls
}

void SynchGPU() {
	//CudaCheck(cudaDeviceSynchronize());
}

void SynchSystem() {
	SynchGPU();
	SynchCPU();
}


/////////////////////////Core/////////////////////////////////

void Soul::Process(Frame& oldFrame, Frame& newFrame) {

	EarlyFrameUpdate();

	newFrame.Dirty(Poll());

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

	//Backends should handle multithreading
	detail->windowManager_->Draw();

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
		detail->scheduler_.AddTask(fParams, [this]()
		{
			detail->consoleManager_->Poll();
		});
	} else {
		Run();
	}
}

Window& Soul::CreateWindow(WindowParameters& params) {

	Window& window = detail->windowManager_->CreateWindow(params);
	return window;

}


void Soul::Run()
{

	Warmup();

	auto currentTime = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now());
	auto nextTime = currentTime + frameTime;

	while (!detail->windowManager_->ShouldClose()) {

		currentTime = nextTime;
		nextTime = currentTime + frameTime;

		detail->framePipeline_.Execute(frameTime);

		if constexpr (Platform::WithCLI()) std::this_thread::sleep_for(frameTime);
		else detail->scheduler_.YieldUntil(nextTime);

	}

	//done running, synchronize rastering
	detail->rasterManager_.Synchronize();

}
