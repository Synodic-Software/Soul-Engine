
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

#include "Utility\Vulkan\VulkanUtility.h"


namespace Soul {


/////////////////////////Variables and Declarations//////////////////

	uint seed;
	RenderType renderer;
	WindowType screen;

	std::vector<GLFWwindow*> windows;

	VulkanWrapper<VkInstance> vulkanInstance{ vkDestroyInstance };
	VulkanWrapper<VkDebugReportCallbackEXT> vulkanCallback{ vulkanInstance, DestroyDebugReportCallbackEXT };
	VulkanWrapper<VkSurfaceKHR> vulkanSurface{ instance, vkDestroySurfaceKHR };

	VkPhysicalDevice vulkanDevicesPhysical = VK_NULL_HANDLE;
	VulkanWrapper<VkDevice> vulkanDeviceLogical{ vkDestroyDevice };

	VkQueue vulkanGraphicsQueue;

	const std::vector<const char*> validationLayers = {
		"VK_LAYER_LUNARG_standard_validation"
	};

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

	VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
		auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
		if (func != nullptr) {
			return func(instance, pCreateInfo, pAllocator, pCallback);
		}
		else {
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
		auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
		if (func != nullptr) {
			func(instance, callback, pAllocator);
		}
	}

	struct QueueFamilyIndices {
		int graphicsFamily = -1;

		bool isComplete() {
			return graphicsFamily >= 0;
		}
	};

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

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

	void SetupDebugCallback() {
		if (!enableValidationLayers) return;

		VkDebugReportCallbackCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
		createInfo.pfnCallback = debugCallback;

		if (CreateDebugReportCallbackEXT(vulkanInstance, &createInfo, nullptr, &vulkanCallback) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug callback!");
		}
	}

	void VulkanPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (IsDeviceSuitable(device)) {
				vulkanDevicesPhysical = device;
				break;
			}
		}

		if (vulkanDevicesPhysical == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	bool IsDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = FindQueueFamilies(device);

		return indices.isComplete();
	}

	QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}

	std::vector<const char*> getRequiredExtensions() {
		std::vector<const char*> extensions;

		unsigned int glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		for (unsigned int i = 0; i < glfwExtensionCount; i++) {
			extensions.push_back(glfwExtensions[i]);
		}

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		return extensions;
	}

	bool CheckValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData) {
		std::cerr << "validation layer: " << msg << std::endl;

		return VK_FALSE;
	}

	void CreateVulkanInstance(){

		if (enableValidationLayers && !CheckValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateInstance(&createInfo, nullptr, &vulkanInstance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	void CreateLogicalDevice() {
		QueueFamilyIndices indices = FindQueueFamilies(vulkanDevicesPhysical);

		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily;
		queueCreateInfo.queueCount = 1;

		float queuePriority = 1.0f;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		VkPhysicalDeviceFeatures deviceFeatures = {};

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.pQueueCreateInfos = &queueCreateInfo;
		createInfo.queueCreateInfoCount = 1;

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = 0;

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(vulkanDevicesPhysical, &createInfo, nullptr, &vulkanDeviceLogical) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(vulkanDeviceLogical, indices.graphicsFamily, 0, &vulkanGraphicsQueue);
	}


	void VulkanInit(){
		CreateVulkanInstance();
		SetupDebugCallback();
		VulkanPhysicalDevice();
		CreateLogicalDevice();

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