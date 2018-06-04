#include "Soul.h"

#include "Rasterer/RasterManager.h"
#include "Transput/Configuration/Settings.h"
#include "Core/Utility/Log/Logger.h"
#include "Physics/PhysicsEngine.h"
#include "Parallelism/Compute/ComputeManager.h"
#include "Display/Window/ManagerInterface.h"
#include "Composition/Event/EventManager.h"
#include "Transput/Input/InputManager.h"
#include "Tracer/RayEngine.h"
#include "Parallelism/Fiber/Scheduler.h"

class Soul::Implementation
{

public:

	Implementation(const Soul&);

	//services and modules
	Scheduler scheduler;

};

Soul::Implementation::Implementation(const Soul& soul):
	scheduler(soul.parameters.threadCount)
{
}

Soul::Soul(SoulParameters& params):
	parameters(params),
	detail(std::make_unique<Implementation>(*this))
{
}

//definitions to complete PIMPL idiom
Soul::~Soul() = default;

Soul::Soul(Soul&&) noexcept = default;
Soul& Soul::operator=(Soul&&) noexcept = default;


/* //////////////////////Synchronization///////////////////////////. */

void SynchCPU() {
	//Scheduler::Block(); //TODO Implement MT calls
}

void SynchGPU() {
	//CudaCheck(cudaDeviceSynchronize());
}

void SynchSystem() {
	SynchCPU();
	SynchGPU();
}


/////////////////////////Hints and Toggles///////////////////////////



/////////////////////////Core/////////////////////////////////


/* Initializes the engine. */
void Soul::Initialize() {

	////create the listener for threads initializeing
	//EventManager::Listen("Thread", "Initialize", []()
	//{
	//	ComputeManager::Instance().InitThread();
	//});

	////open the config file for the duration of the runtime
	//Settings::Read("config.ini", TEXT);

	////extract all available GPU devices
	//ComputeManager::Instance().ExtractDevices();

	////set the error callback
	//glfwSetErrorCallback([](int error, const char* description) {
	//	S_LOG_FATAL("GLFW Error occured, Error ID:", error, " Description:", description);
	//});

	////Initialize glfw context for Window handling
	//const int	didInit = glfwInit();

	//if (!didInit) {
	//	S_LOG_FATAL("GLFW did not initialize");
	//}

	//RasterManager::Instance();

}

/* Call to deconstuct both the engine and its dependencies. */
void Soul::Terminate() {
	SynchSystem();

	//Write the settings into a file
	Settings::Write("config.ini", TEXT);

	//destroy glfw, needs to wait on the window manager
	glfwTerminate();

	//extract all available GPU devices
	ComputeManager::Instance().DestroyDevices();

}


/* Ray pre process */
void RayPreProcess() {

	//RayEngine::Instance().PreProcess();

}

/* Ray post process */
void RayPostProcess() {

	//RayEngine::Instance().PostProcess();

}

/* Rasters this object. */
void Raster() {

	//Backends should handle multithreading
	ManagerInterface::Instance().Draw();

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

void Soul::Run()
{

	Warmup();

	//setup timer info
	double t = 0.0f;
	double currentTime = glfwGetTime();
	double accumulator = 0.0f;

	const double refreshDT = 1.0 / parameters.engineRefreshRate;

	while (!ManagerInterface::Instance().ShouldClose()) {

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
