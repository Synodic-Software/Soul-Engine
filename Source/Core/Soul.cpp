#include "Soul.h"

#include "Display/Window/DisplayManager.h"
#include "Composition/Event/EventManager.h"
#include "Transput/Input/InputManager.h"
#include "Parallelism/Fiber/Scheduler.h"

class Soul::Implementation
{

public:

	Implementation(const Soul&);

	//services and modules
	Scheduler scheduler;
	DisplayManager displayManager;

};

Soul::Implementation::Implementation(const Soul& soul) :
	scheduler(soul.parameters.threadCount)
{
}

Soul::Soul(SoulParameters& params) :
	parameters(params),
	detail(std::make_unique<Implementation>(*this))
{
}

//definitions to complete PIMPL idiom
Soul::~Soul() = default;

Soul::Soul(Soul&&) noexcept = default;
Soul& Soul::operator=(Soul&&) noexcept = default;


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
	detail->displayManager.Draw();

}

/* Warmups this object. */
void Warmup() {

	glfwPollEvents();

	//for (auto& scene : scenes) {
	//	scene->Build(engineRefreshRate);
	//}

}

/* Early frame update. */
void EarlyFrameUpdate() {

	EventManager::Emit("Update", "EarlyFrame");

}
/* Late frame update. */
void LateFrameUpdate() {

	EventManager::Emit("Update", "LateFrame");

}

/* Early update. */
void EarlyUpdate() {

	//poll events before this update, making the state as close as possible to real-time input
	glfwPollEvents();

	//poll input after glfw processes all its callbacks (updating some input states)
	InputManager::Poll();

	EventManager::Emit("Update", "Early");

	//Update the engine cameras
	//RayEngine::Instance().Update();

	//pull cameras into jobs
	EventManager::Emit("Update", "Job Cameras");

}

/* Late update. */
void LateUpdate() {

	EventManager::Emit("Update", "Late");

}

SoulWindow* Soul::CreateWindow(WindowParameters& params) {
	return detail->displayManager.CreateWindow(params);
}


void Soul::Run()
{

	Warmup();

	//setup timer info
	double t = 0.0f;
	double currentTime = glfwGetTime();
	double accumulator = 0.0f;

	const double refreshDT = 1.0 / parameters.engineRefreshRate;

	while (!detail->displayManager.ShouldClose()) {

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
