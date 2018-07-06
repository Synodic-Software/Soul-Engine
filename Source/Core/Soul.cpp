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

class Soul::Implementation
{

public:

	//monostate allows for empty construction
	using inputManagerVariantType = std::variant<std::monostate, DesktopInputManager>;
	using windowManagerVariantType = std::variant<std::monostate, DesktopWindowManager>;

	Implementation(const Soul&);

	//services and modules
	EntityManager registry_;
	Scheduler scheduler_;
	EventManager eventManager_;
	inputManagerVariantType inputManagerVariant_;
	InputManager* inputManager_;
	windowManagerVariantType windowManagerVariant_;
	WindowManager* windowManager_;

};

Soul::Implementation::Implementation(const Soul& soul) :
	registry_(),
	scheduler_(soul.parameters.threadCount),
	eventManager_(),
	inputManagerVariant_(),
	inputManager_(nullptr),
	windowManagerVariant_(),
	windowManager_(nullptr)

{
	if constexpr (Platform::IsDesktop()) {
		inputManagerVariant_.emplace<DesktopInputManager>(eventManager_);
		inputManager_ = &std::get<DesktopInputManager>(inputManagerVariant_);

		windowManagerVariant_.emplace<DesktopWindowManager>(dynamic_cast<DesktopInputManager&>(*inputManager_));
		windowManager_ = &std::get<DesktopWindowManager>(windowManagerVariant_);
	}
}

Soul::Soul(SoulParameters& params) :
	parameters(params),
	detail(std::make_unique<Implementation>(*this))
{
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


/* Ray pre process */
void RayPreProcess() {

	//RayEngine::Instance().PreProcess();

}

/* Ray post process */
void RayPostProcess() {

	//RayEngine::Instance().PostProcess();

}

/* Rasters this object. */
void Soul::Raster() {

	//Backends should handle multithreading
	detail->windowManager_->Draw();

}

/* Warmups this object. */
void Soul::Warmup() {

	//TODO abstract
	detail->inputManager_->Poll();

	//for (auto& scene : scenes) {
	//	scene->Build(engineRefreshRate);
	//}

}

/* Early frame update. */
void Soul::EarlyFrameUpdate() {

	detail->eventManager_.Emit("Update"_hashed, "EarlyFrame"_hashed);

}
/* Late frame update. */
void Soul::LateFrameUpdate() {

	detail->eventManager_.Emit("Update"_hashed, "LateFrame"_hashed);

}

/* Early update. */
void Soul::EarlyUpdate() {

	//poll events before this update, making the state as close as possible to real-time input
	detail->inputManager_->Poll();

	detail->eventManager_.Emit("Update"_hashed, "Early"_hashed);

	//Update the engine cameras
	//RayEngine::Instance().Update();

	//pull cameras into jobs
	detail->eventManager_.Emit("Update"_hashed, "Job Cameras"_hashed);

}

/* Late update. */
void Soul::LateUpdate() {

	detail->eventManager_.Emit("Update"_hashed, "Late"_hashed);

}

Window* Soul::CreateWindow(WindowParameters& params) {
	return detail->windowManager_->CreateWindow(params);
}


void Soul::Run()
{

	Warmup();

	//setup timer info
	double t = 0.0f;
	double currentTime = glfwGetTime();
	double accumulator = 0.0f;

	const double refreshDT = 1.0 / parameters.engineRefreshRate;

	while (!detail->windowManager_->ShouldClose()) {

		//start frame timers
		double newTime = glfwGetTime();
		double frameTime = newTime - currentTime;

		if (frameTime > 0.25) {
			frameTime = 0.25;
		}

		currentTime = newTime;
		accumulator += frameTime;

		EarlyFrameUpdate();

		//consumes time created by the renderer
		while (accumulator >= refreshDT) {

			EarlyUpdate();

			/*for (auto& scene : scenes) {
				scene->Build(engineRefreshRate);
			}*/
			/*
			for (auto const& scene : scenes){
				PhysicsEngine::Process(scene);
			}*/

			LateUpdate();

			t += refreshDT;
			accumulator -= refreshDT;
		}


		LateFrameUpdate();

		RayPreProcess();

		//RayEngine::Instance().Process(*scenes[0], engineRefreshRate);

		RayPostProcess();

		Raster();

	}
}
