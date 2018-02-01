/////////////////////////Includes/////////////////////////////////

#include "SoulCore.h"


#include "Transput/Settings.h"
#include "Utility/Logger.h"
#include "Engine Core/Frame/Frame.h"
#include "Physics Engine\PhysicsEngine.h"
#include "Compute\ComputeManager.h"
#include "Display\Window\WindowManager.h"
#include "Display\Layout\SingleLayout.h"
#include "Display\Widget\RenderWidget.h"
#include "Multithreading\Scheduler.h"
#include "Events\EventManager.h"
#include "Input/InputManager.h"
#include "Ray Engine/RayEngine.h"

#undef GetJob

namespace Soul {

	/* //////////////////////Variables and Declarations//////////////////. */

	std::vector<std::unique_ptr<Scene>> scenes;

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

	/* Initializes the engine. */
	void Initialize() {

		//create the listener for threads initializeing
		EventManager::Listen("Thread", "Initialize", []()
		{
			ComputeManager::Instance().InitThread();
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
			ComputeManager::Instance().ExtractDevices();
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

		if (!didInit) {
			S_LOG_FATAL("GLFW did not initialize");
		}

		Settings::Get("Engine.Delta_Time", 1 / 60.0, engineRefreshRate);
		Settings::Get("Engine.Alloted_Render_Time", 0.01, allotedRenderTime);

		Scheduler::Block();

	}

	/* Call to deconstuct both the engine and its dependencies. */
	void Terminate() {
		Soul::SynchSystem();

		//Write the settings into a file
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
			Settings::Write("config.ini", TEXT);
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
			ComputeManager::Instance().DestroyDevices();
		});

		Scheduler::Block();

		Scheduler::Terminate();
	}


	/* Ray pre process */
	void RayPreProcess() {

		RayEngine::Instance().PreProcess();

	}

	/* Ray post process */
	void RayPostProcess() {

		RayEngine::Instance().PostProcess();

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

		for (auto& scene : scenes) {
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
		RayEngine::Instance().Update();

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

				for (auto& scene : scenes) {
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

			RayEngine::Instance().Process(*scenes[0], engineRefreshRate);

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
	Soul::scenes.push_back(std::unique_ptr<Scene>(scene));
}

/*
 *    Removes the scene described by scene.
 *    @param [in,out]	scene	If non-null, the scene.
 */

void RemoveScene(Scene scene) {

}

/*
 *    Main entry-point for this application.
 *    @return	Exit-code for the process - 0 for success, else an error code.
 */

int main()
{
	SoulInit();

	EventManager::Listen("Input", "ESCAPE", [](keyState state) {
		if (state == RELEASE) {
			SoulSignalClose();
		}
	});

	uint xSize;
	Settings::Get("MainWindow.Width", uint(800), xSize);
	uint ySize;
	Settings::Get("MainWindow.Height", uint(450), ySize);
	uint xPos;
	Settings::Get("MainWindow.X_Position", uint(0), xPos);
	uint yPos;
	Settings::Get("MainWindow.Y_Position", uint(0), yPos);
	int monitor;
	Settings::Get("MainWindow.Monitor", 0, monitor);

	WindowType type;
	int typeCast;
	Settings::Get("MainWindow.Type", static_cast<int>(WINDOWED), typeCast);
	type = static_cast<WindowType>(typeCast);

	glm::uvec2 size = glm::uvec2(xSize, ySize);


	Window* mainWindow = WindowManager::CreateWindow(type, "main", monitor, xPos, yPos, xSize, ySize);

	uint jobID;
	WindowManager::SetWindowLayout(mainWindow, new SingleLayout(new RenderWidget(jobID)));

	RayJob& job = RayEngine::Instance().GetJob(jobID);
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
		mouseChangeDegrees.x = x / camera.fieldOfView.x * 10;
		mouseChangeDegrees.y = y / camera.fieldOfView.y * 10;

		camera.OffsetOrientation(mouseChangeDegrees.x, mouseChangeDegrees.y);
	});

	Scene* scene = new Scene();

	Material whiteGray;
	whiteGray.diffuse = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
	whiteGray.emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

	Object plane("Resources\\Objects\\plane.obj", whiteGray);
	scene->AddObject(plane);

	SubmitScene(scene);

	SoulRun();

	SoulTerminate();

	return EXIT_SUCCESS;

}