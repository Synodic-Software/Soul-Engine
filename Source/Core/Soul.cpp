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

#include <variant>

class Soul::Implementation
{

public:

	//monostate allows for empty construction
	using inputManagerVariantType = std::variant<std::monostate, DesktopInputManager>;
	using windowManagerVariantType = std::variant<std::monostate, DesktopWindowManager>;

	Implementation(const Soul&);
	~Implementation();

	//services and modules	
	Scheduler scheduler_;
	EventManager eventManager_;
	inputManagerVariantType inputManagerVariant_;
	InputManager* inputManager_;
	EntityManager entityManager_;
	windowManagerVariantType windowManagerVariant_;
	WindowManager* windowManager_;
	RasterManager rasterManager_;
	

private:

	inputManagerVariantType ConstructInputManager();
	InputManager* ConstructInputPtr();

	windowManagerVariantType ConstructWindowManager();
	WindowManager* ConstructWindowPtr();
};

Soul::Implementation::Implementation(const Soul& soul) :	
	scheduler_(soul.parameters.threadCount),
	eventManager_(),
	inputManagerVariant_(ConstructInputManager()),
	inputManager_(ConstructInputPtr()),
	entityManager_(),
	windowManagerVariant_(ConstructWindowManager()),
	windowManager_(ConstructWindowPtr()),
	rasterManager_(scheduler_, entityManager_)
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
		tmp.emplace<DesktopWindowManager>(entityManager_,std::get<DesktopInputManager>(inputManagerVariant_), rasterManager_);
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

Window& Soul::CreateWindow(WindowParameters& params) {

	Window& window = detail->windowManager_->CreateWindow(params);
	return window;

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

	//done running, synchronize rastering
	detail->rasterManager_.Synchronize();

}
