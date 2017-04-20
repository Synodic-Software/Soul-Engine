
/////////////////////////Includes/////////////////////////////////

#include "SoulCore.h"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Raster Engine/RasterBackend.h"
#include "Engine Core/BasicDependencies.h"

#include "Utility/Settings.h"
#include "Utility/Logger.h"

#include "Engine Core/Frame/Frame.h"
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
			Settings::Write("config.ini", TEXT);
		});

		//Clean the RayEngine from stray data
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
				RayEngine::Terminate();
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

		//extract all available GPU devices
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			GPUManager::DestroyDevices();
		});

		Scheduler::Block();

		Scheduler::Terminate();
	}

	//Initializes the engine
	void Initialize() {

		//setup the multithreader
		Scheduler::Initialize();

#if defined(_DEBUG) && !defined(SOUL_SINGLE_STACK) 
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
			Settings::Read("config.ini", TEXT);
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

		//Initialize glfw context for Window handling
		int didInit;
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&didInit]() {
			didInit = glfwInit();
		});

		Scheduler::Block();

		//init main Window
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			WindowManager::Initialize(&running);
		});

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			RayEngine::Initialize();
		});

		if (!didInit) {
			S_LOG_FATAL("GLFW did not initialize");
		}

		Settings::Get("Engine.Engine_Refresh_Rate", 60.0, &engineRefreshRate);

		Scheduler::Block();
	}

	void Raster() {

		//Backends should handle multithreading
		WindowManager::Draw();
	}

	void Warmup() {

		double deltaTime = 1.0 / engineRefreshRate;

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwPollEvents();
		});

		InputState::GetInstance().ResetOffsets(); //temp, replace input engine at some point


		for (auto const& scene : scenes) {
			scene->Build(deltaTime);
		}

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

				for (auto const& scene : scenes) {
					scene->Build(deltaTime);
				}
				/*
				for (auto const& scene : scenes){
					PhysicsEngine::Process(scene);
				}*/

				LateUpdate();

				t += deltaTime;
				accumulator -= deltaTime;
			}


			LateFrameUpdate();

			for (auto const& scene : scenes) {
				RayEngine::Process(scene);
			}

			Raster();

		}
	}
}

/////////////////////////User Interface///////////////////////////

void SoulSignalClose(int pressType) {
	Soul::running = false;
	WindowManager::SignelClose();
}

void SoulRun() {
	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
		Soul::Run();
	});

	Scheduler::Block();
}

double GetDeltaTime() {
	return Soul::engineRefreshRate / 60.0;
}

void SetKey(int key, std::function<void(int)> func) {
	InputState::GetInstance().SetKey(key, func);
}
void MouseEvent(std::function<void(double,double)> func) {
	InputState::GetInstance().AddMouseCallback(func);
}

//Initializes Soul. This should be the first command in a program.
void SoulInit() {
	Soul::Initialize();
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

		InputState::GetInstance().ResetMouse = true;

		SetKey(GLFW_KEY_ESCAPE, SoulSignalClose);

		uint xSize;
		Settings::Get("MainWindow.Width", uint(1024), &xSize);
		uint ySize;
		Settings::Get("MainWindow.Height", uint(720), &ySize);
		uint xPos;
		Settings::Get("MainWindow.X_Position", uint(0), &xPos);
		uint yPos;
		Settings::Get("MainWindow.Y_Position", uint(0), &yPos);
		int monitor;
		Settings::Get("MainWindow.Monitor", 0, &monitor);

		WindowType type;
		int typeCast;
		Settings::Get("MainWindow.Type", static_cast<int>(WINDOWED), static_cast<int*>(&typeCast));
		type = static_cast<WindowType>(typeCast);

		Camera* camera = new Camera();

		camera->SetPosition(glm::vec3((DECAMETER) * 5, DECAMETER*5, (DECAMETER) * 5));
		camera->OffsetOrientation(225, 45);

		Window* mainWindow = WindowManager::CreateWindow(FULLSCREEN, "main", monitor, xPos, yPos, 1920, 1080);

		WindowManager::SetWindowLayout(mainWindow, new SingleLayout(new RenderWidget(camera)));

		double deltaTime = GetDeltaTime();
		float moveSpeed = 1 * METER * deltaTime;

		SetKey(GLFW_KEY_S, [&camera, &moveSpeed](int action) {
			camera->OffsetPosition(float(moveSpeed) * -camera->Forward());
		});

		SetKey(GLFW_KEY_W, [&camera, &moveSpeed](int action) {
			camera->OffsetPosition(float(moveSpeed) * camera->Forward());
		});

		SetKey(GLFW_KEY_A, [&camera, &moveSpeed](int action) {
			camera->OffsetPosition(float(moveSpeed) * -camera->Right());
		});

		SetKey(GLFW_KEY_D, [&camera, &moveSpeed](int action) {
			camera->OffsetPosition(float(moveSpeed) * camera->Right());
		});

		SetKey(GLFW_KEY_Z, [&camera, &moveSpeed](int action) {
			camera->OffsetPosition(float(moveSpeed) * -glm::vec3(0, 1, 0));
		});

		SetKey(GLFW_KEY_X, [&camera, &moveSpeed](int action) {
			camera->OffsetPosition(float(moveSpeed) * glm::vec3(0, 1, 0));
		});


		SetKey(GLFW_KEY_LEFT_SHIFT, [deltaTime, &moveSpeed](int action) {
			if (action == GLFW_PRESS) {
				moveSpeed = 9 * METER * deltaTime;
			}
			else if (action == GLFW_RELEASE) {
				moveSpeed = 1 * METER * deltaTime;
			}
		});

		SetKey(GLFW_KEY_LEFT_ALT, [deltaTime, &moveSpeed](int action) {
			if (action == GLFW_PRESS) {
				moveSpeed = 1 * METER * deltaTime;
			}
			else if (action == GLFW_RELEASE) {
				moveSpeed = 1 * METER * deltaTime;
			}
		});


		SetKey(GLFW_KEY_LEFT_ALT, [deltaTime, &moveSpeed](int action) {
			if (action == GLFW_PRESS) {
				moveSpeed = 1 * METER * deltaTime;
			}
			else if (action == GLFW_RELEASE) {
				moveSpeed = 1 * METER * deltaTime;
			}
		});

		MouseEvent([&camera](double xPos, double yPos) {
			glm::dvec2 mouseChangeDegrees;
			mouseChangeDegrees.x = (float)(xPos / camera->resolution.x *camera->FieldOfView().x);
			mouseChangeDegrees.y = (float)(yPos / camera->resolution.y *camera->FieldOfView().y);

			camera->OffsetOrientation(mouseChangeDegrees.x, mouseChangeDegrees.y);
			camera->UpdateVariables();
		});
	

		Scene* scene = new Scene();

		Material* Tree = new Material("Resources\\Textures\\Tree_Color.png");
		Tree->diffuse = glm::vec4(0.3f, 0.8f, 0.3f, 1.0f);
		Tree->emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

		Material* whiteGray = new Material();
		whiteGray->diffuse = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
		whiteGray->emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

		Material* light = new Material("Resources\\Textures\\White.png");
		light->diffuse = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		light->emit = glm::vec4(20.0f, 20.0f, 20.0f, 1.0f);

		Object* tree = new Object("Resources\\Objects\\Tree.obj", Tree);
		scene->AddObject(glm::mat4(),tree);

		Object* plane = new Object("Resources\\Objects\\Plane.obj", whiteGray);
		scene->AddObject(glm::mat4(),plane);

		Object* sphere = new Object("Resources\\Objects\\Sphere.obj", light);
		scene->AddObject(glm::translate(glm::mat4(), glm::vec3(-(DECAMETER) * 10, DECAMETER * 20, (DECAMETER) * 10)),sphere);

		SubmitScene(scene);

		SoulRun();

		delete whiteGray;
		delete scene;

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