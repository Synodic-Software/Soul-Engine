
/////////////////////////Includes/////////////////////////////////

#include "Engine Core/BasicDependencies.h"
#include "SoulCore.h"
#include "Settings.h"
#include "Multithreading\Scheduler.h"
#include "Engine Core/Frame/Frame.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/Input.h"
#include "Ray Engine/RayEngine.h"
#include "Physics Engine\PhysicsEngine.h"
#include "Renderer\Renderer.h"
#include "Bounding Volume Heirarchy/BVH.h"
#include "Resources\Objects\Hand.h"
#include "Utility\CUDA\CUDADevices.cuh"
#include <array>


#include "Utility\Vulkan\VulkanBackend.h"



namespace Soul {


	/////////////////////////Variables and Declarations//////////////////

	uint seed;

	typedef struct window{
		GLFWwindow* window;
		WindowType type;
	}window;

	typedef struct renderer{
		Renderer* renderer;
		RenderType type;
		float timeModifier;
	}renderer;

	std::vector<window> windows;
	GLFWwindow* masterWindow;

	int monitorCount;
	GLFWmonitor** monitors;

	std::vector<renderer> renderObjects;

	std::list<Scene*> scenes;

	Settings* settings;

	bool usingDefaultCamera;
	std::vector<Camera*> cameras;
	Camera* mouseCamera;

	int engineRefreshRate;

	float CurrentDelta();
	void ClearColor(float, float, float, float);

	/////////////////////////Synchronization///////////////////////////

	void SynchCPU(){

	}
	void SynchGPU(){
		CudaCheck(cudaDeviceSynchronize());
	}
	void Synch(){
		SynchCPU();
		SynchGPU();
	}

	/////////////////////////Hints and Toggles///////////////////////////





	bool RequestRenderSwitch(RenderType newR){
		return true;
	}
	bool RequestWindowSwitch(WindowType newW){
		return true;
	}
	bool RequestScreenSize(glm::uvec2 newScreen){
		return true;
	}

	/////////////////////////Engine Core/////////////////////////////////


	void InitVulkan() {
		VulkanBackend::GetInstance().CreateInstance();
		VulkanBackend::GetInstance().SetupDebugCallback();
		VulkanBackend::GetInstance().CreateSurface(masterWindow);
		VulkanBackend::GetInstance().PickVulkanDevice();
		VulkanBackend::GetInstance().CreateVulkanLogical();
		VulkanBackend::GetInstance().CreateSwapChain(masterWindow);
		VulkanBackend::GetInstance().CreateImageViews();
		VulkanBackend::GetInstance().CreateRenderPass();
		VulkanBackend::GetInstance().CreateDescriptorSetLayout();
		VulkanBackend::GetInstance().CreateGraphicsPipeline();
		VulkanBackend::GetInstance().CreateCommandPool();
		VulkanBackend::GetInstance().CreateDepthResources();
		VulkanBackend::GetInstance().CreateFramebuffers();
		VulkanBackend::GetInstance().CreateTextureImage();
		VulkanBackend::GetInstance().CreateTextureImageView();
		VulkanBackend::GetInstance().CreateTextureSampler();
		VulkanBackend::GetInstance().LoadModel();
		VulkanBackend::GetInstance().CreateVertexBuffer();
		VulkanBackend::GetInstance().CreateIndexBuffer();
		VulkanBackend::GetInstance().CreateUniformBuffer();
		VulkanBackend::GetInstance().CreateDescriptorPool();
		VulkanBackend::GetInstance().CreateDescriptorSet();
		VulkanBackend::GetInstance().CreateCommandBuffers();
		VulkanBackend::GetInstance().CreateSemaphores();
	}

	void InputToCamera(GLFWwindow* window,Camera* camera){

		if (camera != nullptr){

			int width, height;
			glfwGetWindowSize(window, &width, &height);

			mouseCamera->OffsetOrientation(
				(float)(SoulInput::xPos / width * camera->FieldOfView().x),
				(float)(SoulInput::yPos / height * camera->FieldOfView().y));
		}

	}

	void UpdateDefaultCamera(GLFWwindow* window,double deltaTime){
		double moveSpeed;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
			moveSpeed = 9 * METER * deltaTime;
		}
		else if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS){
			moveSpeed = 1 * METER * deltaTime;
		}
		else{
			moveSpeed = 4.5 * METER * deltaTime;
		}

		//fill with freecam variable
		if (true){
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
				mouseCamera->OffsetPosition(float(moveSpeed) * -mouseCamera->Forward());
			}
			else if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
				mouseCamera->OffsetPosition(float(moveSpeed) * mouseCamera->Forward());
			}
			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
				mouseCamera->OffsetPosition(float(moveSpeed) * -mouseCamera->Right());
			}
			else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
				mouseCamera->OffsetPosition(float(moveSpeed) * mouseCamera->Right());
			}
			if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS){
				mouseCamera->OffsetPosition(float(moveSpeed) * -glm::vec3(0, 1, 0));
			}
			else if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS){
				mouseCamera->OffsetPosition(float(moveSpeed) * glm::vec3(0, 1, 0));
			}
		}

	}


	void SetKey(int key, void(*func)(void)){
		SoulInput::SetKey(key, std::bind(func));
	}

	void Run()
	{

		SynchGPU();

		//timer info for loop
		double t = 0.0f;
		double currentTime = glfwGetTime();
		double accumulator = 0.0f;

		int width, height;
		glfwGetWindowSize(masterWindow, &width, &height);
		glfwSetCursorPos(masterWindow, width / 2.0f, height / 2.0f);

		glfwPollEvents();
		
		double deltaTime = 1.0 / engineRefreshRate;

		for (auto const& scene : scenes){
			scene->Build(deltaTime);
		}

		bool test = true;		
		
		//stop loop when glfw exit is called
		while (!glfwWindowShouldClose(masterWindow)){
			double newTime = glfwGetTime();
			double frameTime = newTime - currentTime;
			//std::cout << "FPS:: " <<1.0f / frameTime << std::endl;

			//setting up timers
			if (frameTime > 0.25){
				frameTime = 0.25;
			}
			currentTime = newTime;
			accumulator += frameTime;

			//# of updates based on accumulated time

			while (accumulator >= deltaTime){
				SynchGPU();

				deltaTime = 1.0 / engineRefreshRate;

				//loading and updates for multithreading

				glfwPollEvents();

				if (usingDefaultCamera){
					UpdateDefaultCamera(masterWindow, deltaTime);
					InputToCamera(masterWindow, mouseCamera);
				}



				//apply camera changes to their matrices
				for (auto const& cam : cameras){
					cam->UpdateVariables();
				}
				
				//Update();

				cudaEvent_t start, stop;
				float time;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);

				for (auto const& scene : scenes){
					scene->Build(deltaTime);
				}

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&time, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);

				std::cout << "Building Execution: " << time << "ms" << std::endl;


				for (auto const& scene : scenes){
					PhysicsEngine::Process(scene);
				}


				SoulInput::ResetOffsets();

				t += deltaTime;
				accumulator -= deltaTime;
			}

			for (auto const& rend : renderObjects){
				rend.renderer->RenderSetup(SCREEN_SIZE, camera, deltaTime);
			}

			test = !test;

			for (auto const& scene : scenes){
				RayEngine::Clear();
				RayEngine::Process(scene);
			}

			SynchGPU();
			for (auto const& rend : renderObjects){
				rend.renderer->Render();
			}
		}
	}

	void SoulRun(){
		Run();
	}

}

/////////////////////////User Interface///////////////////////////

//Call to deconstuct both the engine and its dependencies
void SoulShutDown(){
	Soul::Synch();
	RayEngine::Clean();
	CudaCheck(cudaDeviceReset());
	delete Soul::settings;
	glfwTerminate();
}

void AddObject(Scene* scene, glm::vec3& globalPos, const char* file, Material* mat){
	Object* obj = new Object(globalPos, file, mat);
	scene->AddObject(obj);
}
void RemoveObject(void* object){

}

int GetSetting(std::string request){
	return Soul::settings->Retrieve(request);
}

int GetSetting(std::string request, int defaultSet){
	return Soul::settings->Retrieve(request, defaultSet);
}

void SetSetting(std::string rName, int rValue){
	Soul::settings->Set(rName, rValue);
}

//Initializes Soul. This must be called before using variables or 
//any other functions relating to the engine.
void SoulInit(){

	Soul::seed = uint(time(NULL));
	srand(Soul::seed);

	Soul::settings = new Settings("Settings.ini");

	Devices::ExtractDevices();

	Soul::Synch();

	if (!glfwInit()){
		SoulShutDown();
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	RenderType  win = static_cast<RenderType>(GetSetting("Renderer", 2));

	Soul::engineRefreshRate = GetSetting("Engine Refresh Rate", 60);

	Soul::monitors = glfwGetMonitors(&Soul::monitorCount);

	Soul::usingDefaultCamera = true;
	Soul::mouseCamera = new Camera();
	Soul::cameras.push_back(Soul::mouseCamera);

	glfwSetKeyCallback(Soul::masterWindow, SoulInput::InputKeyboardCallback);
	glfwSetScrollCallback(Soul::masterWindow, SoulInput::ScrollCallback);
	glfwSetCursorPosCallback(Soul::masterWindow, SoulInput::UpdateMouseCallback);

}

//the moniter number, and a float from 0-1 of the screen size for each dimension
GLFWwindow* SoulCreateWindow(int monitor, float xSize, float ySize){

	GLFWmonitor* monitorIn = Soul::monitors[monitor];

	WindowType  win = static_cast<WindowType>(GetSetting("Window", 2));


	glfwWindowHint(GLFW_SAMPLES, 0);
	glfwWindowHint(GLFW_VISIBLE, GL_TRUE);

	const GLFWvidmode* mode = glfwGetVideoMode(monitorIn);

	GLFWwindow* windowOut;

	if (win == FULLSCREEN){

		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		glfwWindowHint(GLFW_DECORATED, GL_FALSE);
		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		windowOut = glfwCreateWindow(mode->width, mode->height, "Soul Engine", monitorIn, NULL);

	}
	else if (win == WINDOWED){

		glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
		windowOut = glfwCreateWindow(int(xSize*mode->width), int(ySize*mode->height), "Soul Engine", NULL, NULL);

	}

	else if (win == BORDERLESS){

		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		glfwWindowHint(GLFW_DECORATED, GL_FALSE);

		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		//mainThread = glfwCreateWindow(mode->width, mode->height, "Soul Engine", NULL, NULL);   //<---------actual
		windowOut = glfwCreateWindow(mode->width, mode->height, "Soul Engine", NULL, NULL);

	}
	else{
		throw std::runtime_error("NO Window setting found");
	}



	if (!windowOut){
		throw std::runtime_error("GLFW window failed");
	}


	Soul::windows.push_back({ windowOut, win });

	return windowOut;

	/*glfwSetInputMode(mainThread, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0, SCREEN_SIZE.y / 2.0);


	camera->SetPosition(glm::vec3(-(METER * 2), METER * 2 * 2, -(METER * 2)));
	camera->OffsetOrientation(45, 45);

	glfwSetKeyCallback(mainThread, InputKeyboardCallback);
*/

}

void SubmitScene(Scene* scene){
	Soul::scenes.push_back(scene);
}

void RemoveScene(Scene* scene){
	Soul::scenes.remove(scene);
}

int main()
{
	SoulInit();

	//create a Window
	SoulCreateWindow(0,1.0f,1.0f);

	SetKey(GLFW_KEY_ESCAPE, SoulShutDown);

	Material* whiteGray = new Material();
	whiteGray->diffuse = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
	whiteGray->emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

	Scene* scene = new Scene();
	AddObject(scene, glm::vec3(0, 0, 0), "Rebellion.obj", whiteGray);

	SoulRun();

	return 0;
}