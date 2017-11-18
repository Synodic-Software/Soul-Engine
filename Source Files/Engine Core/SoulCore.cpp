
/////////////////////////Includes/////////////////////////////////

#include "SoulCore.h"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Engine Core/BasicDependencies.h"

#include "Transput/Settings.h"
#include "Utility/Logger.h"

#include "Engine Core/Frame/Frame.h"
#include "Ray Engine/RayEngine.h"
#include "Physics Engine\PhysicsEngine.h"
#include "GPGPU\GPUManager.h"
#include "Display\Window\WindowManager.h"
#include "Display\Layout\SingleLayout.h"
#include "Display\Widget\RenderWidget.h"
#include "Multithreading\Scheduler.h"
#include "Events\EventManager.h"
#include "Input/InputManager.h"

#undef GetJob

namespace Soul {

	/* //////////////////////Variables and Declarations//////////////////. */

	std::list<Scene*> scenes;

	/* The engine refresh rate */
	double engineRefreshRate;
	/* The alloted render time */
	double allotedRenderTime;
	/* True to running */
	bool running = true;
	/* //////////////////////Synchronization///////////////////////////. */

	void SynchCPU() {
		Scheduler::Block();
	}

	/* Synchronises the GPU. */
	void SynchGPU() {
		//CudaCheck(cudaDeviceSynchronize());
	}

	/* Synchronises the system. */
	void SynchSystem() {
		SynchCPU();
		SynchGPU();
	}

	/////////////////////////Hints and Toggles///////////////////////////



	/////////////////////////Engine Core/////////////////////////////////


	/* Call to deconstuct both the engine and its dependencies. */
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

	/* Initializes the engine. */
	void Initialize() {

		//create the listener for threads initializeing
		EventManager::Listen("Thread","Initialize",[]()
		{
			GPUManager::InitThread();
		});

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


		Settings::Get("Engine.Delta_Time", 1 / 60.0, &engineRefreshRate);
		Settings::Get("Engine.Alloted_Render_Time", 0.01, &allotedRenderTime);

		Scheduler::Block();

	}

	

	/* Ray pre process */
	void RayPreProcess() {

		RayEngine::PreProcess();

	}
	
	/* Ray post process */
	void RayPostProcess() {

		RayEngine::PostProcess();

	}

	/* Rasters this object. */
	void Raster() {

		//Backends should handle multithreading
		WindowManager::Draw();

	}

	/* Warmups this object. */
	void Warmup() {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwPollEvents();
		});

		for (auto const& scene : scenes) {
			scene->Build(engineRefreshRate);
		}

		Scheduler::Block();

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
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwPollEvents();
		});

		//poll input after glfw processes all its callbacks (updating some input states)
		InputManager::Poll();

		EventManager::Emit("Update", "Early");

		//Update the engine cameras
		RayEngine::Update();

		//pull cameras into jobs
		EventManager::Emit("Update", "Job Cameras");

		Scheduler::Block();

	}

	/* Late update. */
	void LateUpdate() {

		EventManager::Emit("Update", "Late");

	}

	/* Runs this object. */
	void Run()
	{

		Warmup();

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
			while (accumulator >= engineRefreshRate) {

				EarlyUpdate();

				for (auto const& scene : scenes) {
					scene->Build(engineRefreshRate);
				}
				/*
				for (auto const& scene : scenes){
					PhysicsEngine::Process(scene);
				}*/

				LateUpdate();

				t += engineRefreshRate;
				accumulator -= engineRefreshRate;
			}


			LateFrameUpdate();

			RayPreProcess();

			for (auto const& scene : scenes) {
				RayEngine::Process(scene, engineRefreshRate);
			}

			RayPostProcess();

			Raster();

		}
	}
}

/*
 *    //////////////////////User Interface///////////////////////////.
 *    @param	pressType	Type of the press.
 */

void SoulSignalClose() {
	Soul::running = false;
	WindowManager::SignelClose();
}

/* Soul run. */
void SoulRun() {
	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
		Soul::Run();
	});

	Scheduler::Block();
}

/*
 *    Gets delta time.
 *    @return	The delta time.
 */

double GetDeltaTime() {
	return Soul::engineRefreshRate;
}

/* Initializes Soul. This should be the first command in a program. */
void SoulInit() {
	Soul::Initialize();
}

/* Soul terminate. */
void SoulTerminate() {
	Soul::Terminate();
}

/*
 *    Submit scene.
 *    @param [in,out]	scene	If non-null, the scene.
 */

void SubmitScene(Scene* scene) {
	Soul::scenes.push_back(scene);
}

/*
 *    Removes the scene described by scene.
 *    @param [in,out]	scene	If non-null, the scene.
 */

void RemoveScene(Scene* scene) {
	Soul::scenes.remove(scene);
}

/*
 *    Main entry-point for this application.
 *    @return	Exit-code for the process - 0 for success, else an error code.
 */

int main()
{

	try
	{
		SoulInit();

		EventManager::Listen("Input", "ESCAPE", [](keyState state) {
			if (state == RELEASE) {
				SoulSignalClose();
			}
		});

		uint xSize;
		Settings::Get("MainWindow.Width", uint(800), &xSize);
		uint ySize;
		Settings::Get("MainWindow.Height", uint(450), &ySize);
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

		glm::uvec2 size = glm::uvec2(xSize, ySize);

		

		Window* mainWindow = WindowManager::CreateWindow(type, "main", monitor, xPos, yPos, xSize, ySize);

		uint jobID;
		WindowManager::SetWindowLayout(mainWindow, new SingleLayout(new RenderWidget(jobID)));

		RayJob& job = RayEngine::GetJob(jobID);
		Camera& camera = job.camera;
		camera.position = glm::vec3(DECAMETER * 5, DECAMETER * 5, (DECAMETER) * 5);
		camera.OffsetOrientation(225, 45);

		double deltaTime = GetDeltaTime();
		float moveSpeed = 10 * METER * deltaTime;

		InputManager::AfixMouse(*mainWindow);

		EventManager::Listen("Input", "S", [&camera, &moveSpeed](keyState state) {

			if (state == PRESS || state == REPEAT) {
				camera.position += float(moveSpeed) * -camera.forward;
			}
		});

		EventManager::Listen("Input", "W", [&camera, &moveSpeed](keyState state) {
			if (state == PRESS || state == REPEAT) {
				camera.position += float(moveSpeed) * camera.forward;
			}
		});

		EventManager::Listen("Input", "A", [&camera, &moveSpeed](keyState state) {
			if (state == PRESS || state == REPEAT) {
				camera.position += float(moveSpeed) * -camera.right;
			}
		});

		EventManager::Listen("Input", "D", [&camera, &moveSpeed](keyState state) {
			if (state == PRESS || state == REPEAT) {
				camera.position += float(moveSpeed) * camera.right;
			}
		});

		EventManager::Listen("Input", "Z", [&camera, &moveSpeed](keyState state) {
			if (state == PRESS || state == REPEAT) {
				camera.position += float(moveSpeed) * -glm::vec3(0, 1, 0);
			}
		});

		EventManager::Listen("Input", "X", [&camera, &moveSpeed](keyState state) {
			if (state == PRESS || state == REPEAT) {
				camera.position += float(moveSpeed) * glm::vec3(0, 1, 0);
			}
		});

		EventManager::Listen("Input", "LEFT SHIFT", [deltaTime, &moveSpeed](keyState state) {
			if (state == PRESS || state == REPEAT) {
				moveSpeed = 90 * METER * deltaTime;
			}
			else if (state == RELEASE) {
				moveSpeed = 10 * METER * deltaTime;
			}
		});

		EventManager::Listen("Input", "Mouse Position", [&camera](double x, double y) {
			glm::dvec2 mouseChangeDegrees;
			mouseChangeDegrees.x = x / camera.fieldOfView.x * 4;
			mouseChangeDegrees.y = y / camera.fieldOfView.y * 4;

			camera.OffsetOrientation(mouseChangeDegrees.x, mouseChangeDegrees.y);
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
		scene->AddObject(glm::mat4(), tree);

		Object* plane = new Object("Resources\\Objects\\Plane.obj", whiteGray);
		scene->AddObject(glm::mat4(), plane);

		glm::mat4 transform;
		transform = glm::translate(transform, /*100000000000.0f**/glm::vec3(-(DECAMETER) * 10, DECAMETER * 20, (DECAMETER) * 10));
		transform = glm::scale(transform, /*100000000000.0f**/glm::vec3(1.0f, 1.0f, 1.0f));

		Object* sphere = new Object("Resources\\Objects\\Sphere.obj", light);
		scene->AddObject(transform, sphere);

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