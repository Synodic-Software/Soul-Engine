#include "Soul.h"

#include "Platform/Platform.h"

#include "Display/Window/WindowManager.h"
#include "Display/Window/Desktop/DesktopWindowManager.h"

#include "Transput/Input/Desktop/DesktopInputManager.h"

#include "Composition/Event/EventManager.h"
#include "Transput/Input/InputManager.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Core/Utility/HashString/HashString.h"
#include "Composition/Entity/EntityManager.h"
#include "Rasterer/RasterManager.h"
#include "Frame/FramePipeline.h"

#include <variant>

class Soul::Implementation
{

	friend class Soul;

public:

	//monostate allows for empty construction
	using inputManagerVariantType = std::variant<std::monostate, DesktopInputManager>;
	using windowManagerVariantType = std::variant<std::monostate, DesktopWindowManager>;

	Implementation(Soul&);
	~Implementation();

	//services and modules	
	EntityManager entityManager_;
	Scheduler scheduler_;
	EventManager eventManager_;
	inputManagerVariantType inputManagerVariant_;
	InputManager* inputManager_;
	windowManagerVariantType windowManagerVariant_;
	WindowManager* windowManager_;
	RasterManager rasterManager_;
	FramePipeline<3> framePipeline_;

private:

	inputManagerVariantType ConstructInputManager();
	InputManager* ConstructInputPtr();

	windowManagerVariantType ConstructWindowManager();
	WindowManager* ConstructWindowPtr();
};

Soul::Implementation::Implementation(Soul& soul) :
	entityManager_(),
	scheduler_(soul.parameters.threadCount),
	eventManager_(),
	inputManagerVariant_(ConstructInputManager()),
	inputManager_(ConstructInputPtr()),
	windowManagerVariant_(ConstructWindowManager()),
	windowManager_(ConstructWindowPtr()),
	rasterManager_(scheduler_, entityManager_),
	framePipeline_(scheduler_, {
	[&soul](Frame& oldFrame, Frame& newFrame)
	{
		soul.Process(oldFrame, newFrame);
	},
	[&soul](Frame& oldFrame, Frame& newFrame)
	{
		soul.Update(oldFrame, newFrame);
	},
	[&soul](Frame& oldFrame, Frame& newFrame)
	{
		soul.Render(oldFrame, newFrame);
	}})

{


}

Soul::Implementation::~Implementation() {
	windowManager_->Terminate();
}

Soul::Implementation::inputManagerVariantType Soul::Implementation::ConstructInputManager() {

	inputManagerVariantType tmp;

	if constexpr (Platform::IsDesktop()) {
		tmp.emplace<DesktopInputManager>(eventManager_);
		return tmp;
	}

}

InputManager* Soul::Implementation::ConstructInputPtr() {

	if constexpr (Platform::IsDesktop()) {
		return &std::get<DesktopInputManager>(inputManagerVariant_);
	}

}

Soul::Implementation::windowManagerVariantType Soul::Implementation::ConstructWindowManager() {

	windowManagerVariantType tmp;

	if constexpr (Platform::IsDesktop()) {
		tmp.emplace<DesktopWindowManager>(entityManager_, std::get<DesktopInputManager>(inputManagerVariant_), rasterManager_);
		return tmp;
	}

}

WindowManager* Soul::Implementation::ConstructWindowPtr() {

	if constexpr (Platform::IsDesktop()) {
		return &std::get<DesktopWindowManager>(windowManagerVariant_);
	}
}


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

		detail->scheduler_.YieldUntil(nextTime);

	}

	//done running, synchronize rastering
	detail->rasterManager_.Synchronize();

}
