
/////////////////////////Includes/////////////////////////////////

#include "SoulCore.h"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Raster Engine/RasterBackend.h"
#include "Engine Core/BasicDependencies.h"

#include "Utility/Settings.h"
#include "Utility/Logger.h"

#include "Engine Core/Frame/Frame.h"
#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/InputState.h"
#include "Ray Engine/RayEngine.h"
#include "Physics Engine\PhysicsEngine.h"
#include "Bounding Volume Heirarchy/BVH.h"
#include "GPGPU\GPUManager.h"
#include "Display\Window\WindowManager.h"
#include "Display\Layout\SingleLayout.h"
#include "Display\Widget\RenderWidget.h"
#include "Multithreading\Scheduler.h"

namespace Soul {

	/////////////////////////Variables and Declarations//////////////////

	std::list<Scene*> scenes;

	double engineRefreshRate;
	bool running = true;
	/////////////////////////Synchronization///////////////////////////

	void SynchCPU() {
		Scheduler::Block();
	}

	void SynchGPU() {
		//CudaCheck(cudaDeviceSynchronize());
	}

	void SynchSystem() {
		SynchCPU();
		SynchGPU();
	}

	/////////////////////////Hints and Toggles///////////////////////////



	/////////////////////////Engine Core/////////////////////////////////


	//Call to deconstuct both the engine and its dependencies
	void Terminate() {
		Soul::SynchSystem();

		//Write the settings into a file
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			Settings::Write();
		});

		//Clean the RayEngine from stray data
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			//	RayEngine::Clean();
		});

		//destroy all windows
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			WindowManager::Terminate();
		});
		Scheduler::Block();

		//destroy glfw, needs to wait on the window manager
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwTerminate();
		});

		Scheduler::Block();

		Scheduler::Terminate();
	}

	//Initializes the engine
	void Init() {

		//setup the multithreader
		Scheduler::Init();

#ifdef _DEBUG 
		//log errors to the console for now
		Scheduler::AddTask(LAUNCH_CONTINUE, FIBER_LOW, false, []() {
			while (Scheduler::Running()) {
				std::cout << Logger::Get();
				Scheduler::Defer();
			}
		});
#endif

		//open the config file for the duration of the runtime
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			Settings::Read("config.ini");
		});

		//extract all available GPU devices
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			GPUManager::ExtractDevices();
		});

		//set the error callback
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwSetErrorCallback([](int error, const char* description) {
				S_LOG_FATAL("GLFW Error occured, Error ID:", error, " Description:", description);
			});
		});

		//Init glfw context for Window handling
		int didInit;
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&didInit]() {
			didInit = glfwInit();
		});

		Scheduler::Block();

		//init main Window
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			WindowManager::Init(&running);
		});

		if (!didInit) {
			S_LOG_FATAL("GLFW did not initialize");
		}

		engineRefreshRate = Settings::Get("Engine.Engine_Refresh_Rate", 60.0);

		Scheduler::Block();
	}

	void Raster() {

		//Backends should handle multithreading
		WindowManager::Draw();
	}

	void Warmup() {

		//double deltaTime = 1.0 / engineRefreshRate;
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwPollEvents();
		});
		/*for (auto const& scene : scenes) {
			scene->Build(deltaTime);
		}*/
		Scheduler::Block();

	}

	void EarlyFrameUpdate() {

	}
	void LateFrameUpdate() {

	}

	void EarlyUpdate() {

		//poll events before this update, making the state as close as possible to real-time input
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwPollEvents();
		});
		InputState::GetInstance().ResetOffsets(); //temp, replace input engine at some point

		Scheduler::Block();
	}

	void LateUpdate() {

	}

	void Run()
	{

		Warmup();

		double deltaTime = 1.0 / engineRefreshRate;


		//setup timer info
		double t = 0.0f;
		double currentTime = glfwGetTime();
		double accumulator = 0.0f;

		while (running && !WindowManager::ShouldClose()) {

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
			while (accumulator >= deltaTime) {

				deltaTime = 1.0 / engineRefreshRate;

				EarlyUpdate();

				/*for (auto const& scene : scenes) {
					scene->Build(deltaTime);
				}*/
				/*
				for (auto const& scene : scenes){
					PhysicsEngine::Process(scene);
				}*/

				LateUpdate();

				t += deltaTime;
				accumulator -= deltaTime;
			}


			LateFrameUpdate();

			/*	for (auto const& rend : renderObjects) {
					int width, height;
					glfwGetWindowSize(masterWindow, &width, &height);
					rend.rendererHandle->RenderSetup({ width, height }, deltaTime);
				}*/

				/*	for (auto const& scene : scenes){
						RayEngine::Clear();
						RayEngine::Process(scene);
					}*/

			Raster();

		}
	}
}

/////////////////////////User Interface///////////////////////////

void SoulSignalClose() {
	Soul::running = false;
	WindowManager::SignelClose();
}

void SoulRun() {
	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
		Soul::Run();
	});
}

void SetKey(int key, void(*func)(void)) {
	InputState::GetInstance().SetKey(key, std::bind(func));
}

void AddObject(Scene* scene, glm::vec3& globalPos, const char* file, Material* mat) {
	Object* obj = new Object(globalPos, file, mat);
	scene->AddObject(obj);
}
void RemoveObject(void* object) {

}

//Initializes Soul. This should be the first command in a program.
void SoulInit() {
	Soul::Init();
}

void SoulTerminate() {
	Soul::Terminate();
}

void SubmitScene(Scene* scene) {
	Soul::scenes.push_back(scene);
}

void RemoveScene(Scene* scene) {
	Soul::scenes.remove(scene);
}

int main()
{

	try
	{
		SoulInit();

		//InputState::GetInstance().ResetMouse = true;

		SetKey(GLFW_KEY_ESCAPE, SoulSignalClose);

		uint xSize = Settings::Get("MainWindow.Width", 1024);
		uint ySize = Settings::Get("MainWindow.Height", 720);
		uint xPos = Settings::Get("MainWindow.X_Position", 0);
		uint yPos = Settings::Get("MainWindow.Y_Position", 0);
		int monitor = Settings::Get("MainWindow.Monitor", 0);
		WindowType type = static_cast<WindowType>(Settings::Get("MainWindow.Type", static_cast<int>(WINDOWED)));

		WindowManager::CreateWindow(type, "main", monitor, xPos, yPos, xSize, ySize, [](GLFWwindow* win) {
			return new SingleLayout(win, new RenderWidget());
		});

		//WindowManager::CreateWindow(WINDOWED, "test", 0, 0, 0, 300, 300);

		/*Material* whiteGray = new Material();
		whiteGray->diffuse = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
		whiteGray->emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

		Scene* scene = new Scene();
		AddObject(scene, glm::vec3(0, 0, 0), "Rebellion.obj", whiteGray);

		SubmitScene(scene);*/

		SoulRun();
		SoulTerminate();
		return EXIT_SUCCESS;
	}
	catch (std::exception const& e)
	{
		std::cerr << "exception: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "unhandled exception" << std::endl;
	}
	return EXIT_FAILURE;
}