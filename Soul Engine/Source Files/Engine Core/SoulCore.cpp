
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
	RenderType renderer;
	WindowType screen;

	int monitorCount;
	GLFWmonitor** monitors;
	Renderer** renderObjects;
	Settings* settings;
	Camera** cameras;

	int engineRefreshRate;

	float* timeModifiers; //one for each render instance

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
		CreateInstance();
		SetupDebugCallback();
		CreateSurface();
		PickVulkanDevice();
		CreateVulkanLogical();
		CreateSwapChain();
		CreateImageViews();
		CreateRenderPass();
		CreateDescriptorSetLayout();
		CreateGraphicsPipeline();
		CreateCommandPool();
		CreateDepthResources();
		CreateFramebuffers();
		CreateTextureImage();
		CreateTextureImageView();
		CreateTextureSampler();
		LoadModel();
		CreateVertexBuffer();
		CreateIndexBuffer();
		CreateUniformBuffer();
		CreateDescriptorPool();
		CreateDescriptorSet();
		CreateCommandBuffers();
		CreateSemaphores();
	}
	
	void UpdateMouse(){
		double xPos;
		double yPos;
		glfwGetCursorPos(mainThread, &xPos, &yPos);
		xPos -= (SCREEN_SIZE.x / 2.0);
		yPos -= (SCREEN_SIZE.y / 2.0);
		mouseChangeDegrees.x = (float)(xPos / SCREEN_SIZE.x * camera->FieldOfView().x);
		mouseChangeDegrees.y = (float)(yPos / SCREEN_SIZE.y * camera->FieldOfView().y);

		if (freeMouse){
			if (freeCam){
				//set camera for each update
				camera->OffsetOrientation(mouseChangeDegrees.x, mouseChangeDegrees.y);
			}


			glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0f, SCREEN_SIZE.y / 2.0f);
		}
	}

	void UpdateKeys(){
		double moveSpeed;
		if (glfwGetKey(mainThread, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
			moveSpeed = 9 * METER * deltaTime;
		}
		else if (glfwGetKey(mainThread, GLFW_KEY_LEFT_ALT) == GLFW_PRESS){
			moveSpeed = 1 * METER * deltaTime;
		}
		else{
			moveSpeed = 4.5 * METER * deltaTime;
		}


		if (freeCam){
			if (glfwGetKey(mainThread, GLFW_KEY_S) == GLFW_PRESS){
				camera->OffsetPosition(float(moveSpeed) * -camera->Forward());
			}
			else if (glfwGetKey(mainThread, GLFW_KEY_W) == GLFW_PRESS){
				camera->OffsetPosition(float(moveSpeed) * camera->Forward());
			}
			if (glfwGetKey(mainThread, GLFW_KEY_A) == GLFW_PRESS){
				camera->OffsetPosition(float(moveSpeed) * -camera->Right());
			}
			else if (glfwGetKey(mainThread, GLFW_KEY_D) == GLFW_PRESS){
				camera->OffsetPosition(float(moveSpeed) * camera->Right());
			}
			if (glfwGetKey(mainThread, GLFW_KEY_Z) == GLFW_PRESS){
				camera->OffsetPosition(float(moveSpeed) * -glm::vec3(0, 1, 0));
			}
			else if (glfwGetKey(mainThread, GLFW_KEY_X) == GLFW_PRESS){
				camera->OffsetPosition(float(moveSpeed) * glm::vec3(0, 1, 0));
			}
		}

	}


	void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{

		scrollUniform += (float)(yoffset / 50.0);
		if (scrollUniform > 1.0f){
			scrollUniform = 1.0f;
		}
		else if (scrollUniform < 0.0f){
			scrollUniform = 0.0f;
		}
	}


	void SetKey(int key, void(*func)(void)){
		SetKey(key, std::bind(func));
	}

	void Run()
	{

		SoulSynchGPU();
		camera = new Camera();

		SoulSynchGPU();
		SoulCreateWindow(BORDERLESS, SPECTRAL);


		rend = new Renderer(*camera, SCREEN_SIZE);


		glfwSetScrollCallback(mainThread, ScrollCallback);

		//timer info for loop
		double t = 0.0f;
		double currentTime = glfwGetTime();
		double accumulator = 0.0f;

		glfwPollEvents();
		//hub->UpdateObjects(deltaTime);
		//stop loop when glfw exit is called
		glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0f, SCREEN_SIZE.y / 2.0f);


		scene->Build(deltaTime);


		bool test = true;
		while (!runShutdown){
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
				SoulSynchGPU();

				//loading and updates for multithreading

				//set cursor in center

				glfwPollEvents();

				UpdateKeys();

				UpdateMouse();

				camera->UpdateVariables();

				//Update();



				cudaEvent_t start, stop;
				float time;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);

				scene->Build(deltaTime);

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&time, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);

				std::cout << "Building Execution: " << time << "ms" << std::endl;


				PhysicsEngine::Process(scene);




				UpdateTimers();

				t += deltaTime;
				accumulator -= deltaTime;
				//SoulSynch();
			}

			rend->RenderSetup(SCREEN_SIZE, camera, deltaTime, scrollUniform);
			//camera->UpdateVariables();

			test = !test;







			RayEngine::Clear();



			RayEngine::Process(scene);

			//draw
			ClearColor(1.0f, 1.0f, 1.0f, 1.0f);

			SoulSynchGPU();
			rend->Render();
			SoulSynchGPU();

			glfwSwapBuffers(mainThread);
		}

		delete camera;
		delete scene;
		delete rend;

	}
	glm::vec2* GetMouseChange(){
		return &mouseChangeDegrees;
	}

	void UpdateTimers(){
		if (physicsTimer > 0){
			physicsTimer = physicsTimer - deltaTime;
		}
	}

	void ClearColor(float r, float g, float b, float a){
		glClearColor(r, g, b, a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

void AddObject(Scene* scene,glm::vec3& globalPos, const char* file, Material* mat){
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

	Soul::seed = GLuint(time(NULL));
	srand(Soul::seed);

	Soul::settings = new Settings("Settings.ini");

	Devices::ExtractDevices();

	Soul::Synch();

	if (!glfwInit()){
		SoulShutDown();
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	RenderType  win = static_cast<RenderType>(GetSetting("Renderer", 2));

	Soul::monitors = glfwGetMonitors(&Soul::monitorCount);
}


void SoulCreateWindow(int monitor, float xSize, float ySize){

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


	Soul::windows.push_back(windowOut);


	/*glfwSetInputMode(mainThread, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0, SCREEN_SIZE.y / 2.0);

	
	camera->SetPosition(glm::vec3(-(METER * 2), METER * 2 * 2, -(METER * 2)));
	camera->OffsetOrientation(45, 45);

	glfwSetKeyCallback(mainThread, InputKeyboardCallback);
	SetInputWindow(mainThread);*/

}


int main()
	{
		SoulInit();


		SetKey(GLFW_KEY_ESCAPE, std::bind(&SoulShutDown));

		Material* whiteGray = new Material();
		whiteGray->diffuse = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
		whiteGray->emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

		Scene* scene = new Scene();
		AddObject(scene, glm::vec3(0, 0, 0), "Rebellion.obj", whiteGray);

		SoulRun();

		return 0;
	}