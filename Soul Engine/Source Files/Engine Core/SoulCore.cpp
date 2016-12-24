
/////////////////////////Includes/////////////////////////////////

#include "SoulCore.h"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Raster Engine/RasterBackend.h"
#include "Engine Core/BasicDependencies.h"

#include "Utility/Settings.h"
#include "Utility/Logger.h"
#include "Multithreading\Scheduler.h"
#include "Engine Core/Frame/Frame.h"
#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/InputState.h"
#include "Ray Engine/RayEngine.h"
#include "Physics Engine\PhysicsEngine.h"
//#include "Renderer\Renderer.h"
#include "Bounding Volume Heirarchy/BVH.h"
#include "GPGPU\GPUManager.h"
#include "Multithreading\Scheduler.h"
#include "Display\Window\WindowManager.h"


namespace Soul {

	/////////////////////////Variables and Declarations//////////////////

	//typedef struct renderer {
	//	Renderer* rendererHandle;
	//	RenderType type;
	//	float timeModifier;
	//}renderer;

	//std::vector<renderer> renderObjects;

	std::list<Scene*> scenes;

	//bool usingDefaultCamera;
	/*std::vector<Camera*> cameras;
	Camera* mouseCamera;*/

	int engineRefreshRate;

	/////////////////////////Synchronization///////////////////////////

	void SynchCPU() {
		Scheduler::Wait();
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

		Scheduler::Wait();

		//destroy glfw, needs to wait on the window manager
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
			glfwTerminate();
		});

		Scheduler::Wait();

		Scheduler::Terminate();
	}

	//Initializes the engine
	void Init() {

		//setup the multithreader
		Scheduler::Init();

		//open the config file for the duration of the runtime
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false,[]() {
			Settings::Read("config.ini");
		});

		//extract all available GPU devices
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false,[]() {
			GPUManager::ExtractDevices();
		});

		//Init glfw context for Window handling
		int didInit;
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true,[&]() {
			didInit = glfwInit();
		});

		Scheduler::Wait();

		if (!didInit) {
			Terminate();
		}

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false,[]() {
			RasterBackend::Init();
		});

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false,[]() {
			WindowManager::Init();
		});

		engineRefreshRate = Settings::Get("Engine.Engine_Refresh_Rate", 60);

		Scheduler::Wait();

	}

	void InputToCamera(GLFWwindow* window, Camera* camera) {

		if (camera != nullptr) {

			int width, height;
			glfwGetWindowSize(window, &width, &height);

			/*mouseCamera->OffsetOrientation(
				(float)(InputState::GetInstance().xPos / width * camera->FieldOfView().x),
				(float)(InputState::GetInstance().yPos / height * camera->FieldOfView().y));*/
		}

	}

	void UpdateDefaultCamera(GLFWwindow* window, double deltaTime) {
		double moveSpeed;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
			moveSpeed = 9 * METER * deltaTime;
		}
		else if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) {
			moveSpeed = 1 * METER * deltaTime;
		}
		else {
			moveSpeed = 4.5 * METER * deltaTime;
		}

		//fill with freecam variable
	/*	if (true) {
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
				mouseCamera->OffsetPosition(float(moveSpeed) * -mouseCamera->Forward());
			}
			else if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
				mouseCamera->OffsetPosition(float(moveSpeed) * mouseCamera->Forward());
			}
			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
				mouseCamera->OffsetPosition(float(moveSpeed) * -mouseCamera->Right());
			}
			else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
				mouseCamera->OffsetPosition(float(moveSpeed) * mouseCamera->Right());
			}
			if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
				mouseCamera->OffsetPosition(float(moveSpeed) * -glm::vec3(0, 1, 0));
			}
			else if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
				mouseCamera->OffsetPosition(float(moveSpeed) * glm::vec3(0, 1, 0));
			}
		}*/

	}

	void Warmup() {

		double deltaTime = 1.0 / engineRefreshRate;

		glfwPollEvents();

		for (auto const& scene : scenes) {
			scene->Build(deltaTime);
		}

	}

	void Run()
	{

		Warmup();

		double deltaTime = 1.0 / engineRefreshRate;


		//setup timer info
		double t = 0.0f;
		double currentTime = glfwGetTime();
		double accumulator = 0.0f;

		//stop loop when glfw exit is called
		while (!WindowManager::ShouldClose()) {

			//start frame timers
			double newTime = glfwGetTime();
			double frameTime = newTime - currentTime;

			if (frameTime > 0.25) {
				frameTime = 0.25;
			}

			currentTime = newTime;
			accumulator += frameTime;

			//consumes time created by the renderer
			while (accumulator >= deltaTime) {

				deltaTime = 1.0 / engineRefreshRate;

				//loading and updates for multithreading
				glfwPollEvents();


				//if (usingDefaultCamera){
				//	UpdateDefaultCamera(masterWindow, deltaTime);
				//	InputToCamera(masterWindow, mouseCamera);
				//}

				//apply camera changes to their matrices
				/*for (auto const& cam : cameras){
					cam->UpdateVariables();
				}*/

				/*	for (auto const& scene : scenes){
						scene->Build(deltaTime);
					}

					for (auto const& scene : scenes){
						PhysicsEngine::Process(scene);
					}*/

				t += deltaTime;
				accumulator -= deltaTime;
			}

			/*	for (auto const& rend : renderObjects) {
					int width, height;
					glfwGetWindowSize(masterWindow, &width, &height);
					rend.rendererHandle->RenderSetup({ width, height }, deltaTime);
				}*/

				/*	for (auto const& scene : scenes){
						RayEngine::Clear();
						RayEngine::Process(scene);
					}*/

					/*glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);*/

					//SynchGPU();
					//for (auto const& rend : renderObjects) {
					//	//integration bool
					//	rend.rendererHandle->Render(false);
					//}
			SynchGPU();
			//RayEngine::Clear();
			///////////////////////////////////////////////////////////////////////until vulkan

			/*InputState::GetInstance().ResetOffsets();

			glfwSwapBuffers(masterWindow);*/
			////////////////////////////////////////////////////////////////////////////////////


			WindowManager::Draw();


		}

		//Put Vulkan into idle
		//VulkanBackend::GetInstance().IdleDevice();

		Terminate();

	}
}

/////////////////////////User Interface///////////////////////////

void SoulSignalClose() {
	WindowManager::SignelClose();
}

void SoulRun() {
	Soul::Run();
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

void SubmitScene(Scene* scene) {
	Soul::scenes.push_back(scene);
}

void RemoveScene(Scene* scene) {
	Soul::scenes.remove(scene);
}

int main()
{
	SoulInit();

	//create a Window
	WindowManager::SoulCreateWindow(0, 0, 0, 100, 100);

	InputState::GetInstance().ResetMouse = true;

	SetKey(GLFW_KEY_ESCAPE, SoulSignalClose);

	/*Material* whiteGray = new Material();
	whiteGray->diffuse = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
	whiteGray->emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

	Scene* scene = new Scene();
	AddObject(scene, glm::vec3(0, 0, 0), "Rebellion.obj", whiteGray);

	SubmitScene(scene);*/

	SoulRun();

	return 0;
}