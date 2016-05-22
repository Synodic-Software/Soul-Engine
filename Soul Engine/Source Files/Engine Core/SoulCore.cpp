
/////////////////////////Includes/////////////////////////////////

#include "Engine Core/BasicDependencies.h"
#include "SoulCore.h"
#include "Settings.h"
#include "Multithreading\Scheduler.h"
#include "Engine Core/Frame/Frame.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/Input.h"
#include "Ray Engine/RayEngine.h"
#include "Renderer\Renderer.h"
#include "Bounding Volume Heirarchy/BVH.h"
#include "Engine Core\Scene\Scene.h"
#include "Resources\Objects\Hand.h"
/////////////////////////Variables///////////////////////////////

GLuint seed;
RenderType renderer;
WindowType window;

GLFWwindow* mainThread;
GLFWwindow* loopThread;
Renderer* rend;
glm::uvec2 SCREEN_SIZE;

Settings* settings;
unsigned int MSAASamples;
Scene* scene;
Camera* camera;

float scrollUniform;

bool runShutdown;
bool freeCam;

float timeModifier = 1.0f;
const float deltaTime = (1.0f / 60.0f);
bool runPhysics;
bool freeMouse;
void TogglePhysics();
void UpdateTimers();
void ClearColor(float,float,float,float);
float physicsTimer;
glm::vec2 mouseChangeDegrees;

/////////////////////////User Interface///////////////////////////

//Initializes Soul. This must be called before using variables or 
//any other functions relating to the engine.
void SoulInit(){
	runShutdown = false;
	scrollUniform = 0.05f;
seed = GLuint(time(NULL));
srand(seed);
settings = new Settings("Settings.ini");
physicsTimer = 0;

runPhysics = false;

if (!glfwInit()){
	SoulTerminate();
}

	//set window
std::string wSet = GetSetting("Window");
if (wSet.compare("WINDOWED") == 0){
	window = WINDOWED;
}
else if (wSet.compare("FULLSCREEN") == 0){
	window = FULLSCREEN;
}
else if (wSet.compare("BORDERLESS") == 0){
	window = BORDERLESS;
}
else{
	SetSetting("Window", "FULLSCREEN");
	window = FULLSCREEN;
}

//window = FULLSCREEN; ///////////////////////////////////////////////////////////////////////// <----------- TEMPORARY

	//set renderer
std::string rSet = GetSetting("Renderer");
/*if (rSet.compare("RASTER") == 0){
	renderer = RASTER;
}
else */if (rSet.compare("SPECTRAL") == 0){
	renderer = SPECTRAL;
}
else if (rSet.compare("PATH") == 0){
	renderer = PATH;
}
//else if (rSet.compare("RAY") == 0){
//	renderer = RAY;
//}
else{
	SetSetting("Renderer", "SPECTRAL");
	renderer = SPECTRAL;
}


int count;
GLFWmonitor** monitors = glfwGetMonitors(&count);

	//set screen size
const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

std::string xScreen = GetSetting("ScreenX");
if (xScreen.compare("")!=0){
	SCREEN_SIZE.x = atoi(xScreen.c_str());
}
else{
	SCREEN_SIZE.x = mode->width;
	SetSetting("ScreenX", std::to_string(mode->width));
}

std::string yScreen = GetSetting("ScreenY");
if (yScreen.compare("") != 0){
	SCREEN_SIZE.y = atoi(yScreen.c_str());
}
else{
	SCREEN_SIZE.y = mode->height;
	SetSetting("ScreenY", std::to_string(mode->height));
}

SCREEN_SIZE = glm::uvec2(SCREEN_SIZE.x / 1.05f, SCREEN_SIZE.y / 1.05f );
												///////////////////////////////////////////////////////////////////////// <----------- TEMPORARY
//msaa samples

std::string msAA = GetSetting("MSAA");
if (msAA.compare("") != 0){
	MSAASamples = atoi(msAA.c_str());
}
else{
	SetSetting("MSAA", std::to_string(16));
	MSAASamples = 16;
}

}

//Call to deconstuct both the engine and its dependencies
void SoulTerminate(){
	runShutdown = true;
}
void SoulShutDown(){
	Scheduler::Terminate();
	RayEngine::Clean();
	CudaCheck(cudaDeviceReset());
	//delete hub;
	delete settings;
	glfwTerminate();
}
void AddObject(Object* object){
	//hub->Add(object);

}
void RemoveObject(Object* object){

	

}

void SoulSynchCPU(FiberTaskingLib::TaskScheduler* sched, COUNTER* counter, uint& size){
	for (uint i = 0; i < size; i++){
		sched->WaitForCounter(counter[i], 0);
	}
}
void SoulSynchGPU(){
	CudaCheck(cudaDeviceSynchronize());
}
void SoulSynch(FiberTaskingLib::TaskScheduler* sched,COUNTER* counter,uint& size){
	SoulSynchCPU(sched, counter, size);
	SoulSynchGPU();
}


void SoulCreateWindow(WindowType windowT, RenderType rendererT){

	freeMouse = true;

	window = windowT;
	SetSetting("Window", "WINDOWED");
	renderer = rendererT;
	SetSetting("Renderer", "PATH");

	glfwWindowHint( GLFW_VISIBLE, GL_FALSE );
	loopThread = glfwCreateWindow(1, 1, "Thread Window", NULL, NULL);


	glfwWindowHint(GLFW_SAMPLES, 0);
	glfwWindowHint(GLFW_VISIBLE, GL_TRUE );

	if (window==FULLSCREEN){

		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		glfwWindowHint(GLFW_DECORATED, GL_FALSE);
		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		mainThread = glfwCreateWindow(mode->width, mode->height, "Soul Engine", glfwGetPrimaryMonitor(), loopThread);

	}
	else if (window == WINDOWED){

		glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
		mainThread = glfwCreateWindow(SCREEN_SIZE.x, SCREEN_SIZE.y, "Soul Engine", NULL, loopThread);

	}
	else if (BORDERLESS){

		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		glfwWindowHint(GLFW_DECORATED, GL_FALSE);

		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		//mainThread = glfwCreateWindow(mode->width, mode->height, "Soul Engine", NULL, loopThread);   <---------actual
		mainThread = glfwCreateWindow(SCREEN_SIZE.x, SCREEN_SIZE.y, "Soul Engine", NULL, loopThread);

	}
	

	if (!mainThread){
        glfwTerminate();
		throw std::runtime_error("GLFW window failed");
    }

	glfwMakeContextCurrent(mainThread);

	// initialise GLEW
	if (glewInit() != GLEW_OK){
		throw std::runtime_error("glewInit failed");
	}

	glewExperimental = GL_TRUE; //stops glew crashing on OSX :-/

	glfwSetInputMode(mainThread, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0, SCREEN_SIZE.y / 2.0);

	//Discard all the errors
	while(glGetError() != GL_NO_ERROR) {}

	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthMask(GL_TRUE);  // turn on
	glDepthFunc(GL_LEQUAL);
	glDepthRange(0.0f, 1.0f);
	glEnable(GL_TEXTURE_2D);

	mouseChangeDegrees = glm::vec2(0);
	SoulSynchGPU();
	// setup camera 
	freeCam = true;
	//camera->SetPosition(glm::vec3(-5.0f*METER, 2.0f*METER, 5.0f*METER));
	camera->SetPosition(glm::vec3(-(DECAMETER * 5), DECAMETER * 5, -(DECAMETER * 5)));
	camera->OffsetOrientation(45, 45);

	//unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
	//threads = new ThreadPool(concurentThreadsSupported);

	//hub = new BVH();



	const GLubyte* renderer = glGetString(GL_VENDOR);



	Material::SetDefaultTexture("SoulDefault.png");

	glfwSetKeyCallback(mainThread, InputKeyboardCallback);
	SetInputWindow(mainThread);
}
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

std::string GetSetting(std::string request){
	return settings->Retrieve(request);
}

void SetSetting(std::string rName, std::string rValue){
	settings->Set(rName, rValue);
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
	if (scrollUniform>1.0f){
		scrollUniform = 1.0f;
	}
	else if (scrollUniform<0.0f){
		scrollUniform = 0.0f;
	}
}

TASK_FUNCTION(Run)
{
	
	SoulSynchGPU();
	SoulInit();
	SoulSynchGPU();
	camera = new Camera();
	

	

	SoulSynchGPU();
	SoulCreateWindow(BORDERLESS, SPECTRAL);


	scene = new Scene();
	rend= new Renderer(*camera,SCREEN_SIZE);

	//Hand* hand = new Hand();

	Material* light = new Material();
	light->diffuse = glm::vec4(1.0f,1.0f,1.0f,1.0f);
	light->emit = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

	Material* whiteGray = new Material();
	whiteGray->diffuse = glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
	whiteGray->emit = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);


	Object* obj = new Object(glm::vec3(0.0f, 0.0f, 0.0f), "Winged_Victory.obj", whiteGray);
	scene->AddObject(obj);

	/*Object* sun = new Object(glm::vec3(5.0f*METER, 5.0f*METER, 5.0f*METER), "sphere.obj", light);
	scene->AddObject(sun);*/

	



	

	//Hand* hand1 = new Hand(glm::vec3(0.0f, 0.0f, 20*METER));
	//Object* handObj1 = hand1;
	//scene->AddObject(handObj1);

	//Hand* hand2 = new Hand(glm::vec3(10*METER, 5*METER, 0.0f));
	//Object* handObj2 = hand2;
	//scene->AddObject(handObj2);


	SetKey(GLFW_KEY_ESCAPE, std::bind(&SoulTerminate));
	SetKey(GLFW_KEY_SPACE, std::bind(&TogglePhysics));

	glfwSetScrollCallback(mainThread, ScrollCallback);

	//timer info for loop
	double t = 0.0f;
	double currentTime = glfwGetTime();
	double accumulator = 0.0f;

	glfwPollEvents();
	//hub->UpdateObjects(deltaTime);
	//stop loop when glfw exit is called
	glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0f, SCREEN_SIZE.y / 2.0f);

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

			//loading and updates for multithreading

			//set cursor in center
			UpdateMouse();

			glfwPollEvents();

			UpdateKeys();

			//Update();

			//RunPhysics();
			UpdateTimers();

			t += deltaTime;
			accumulator -= deltaTime;
			//SoulSynch();
		}

		rend->RenderSetup(SCREEN_SIZE, camera, deltaTime, scrollUniform);
		camera->UpdateVariables();

		//if (test){
			test = !test;
			cudaEvent_t start, stop;
			float time;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			scene->Build();

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		//}


		std::cout << "Building Execution: " << time << "ms" << std::endl;


		RayEngine::Clear();



		RayEngine::Process(scene);

		//draw
		ClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		SoulSynchGPU();
		rend->Render(runPhysics);
		SoulSynchGPU();
		
		glfwSwapBuffers(mainThread);
	}

	delete camera;
	delete light;
	delete whiteGray;
	delete obj;
	delete scene;
	delete rend;
}
glm::vec2* GetMouseChange(){
	return &mouseChangeDegrees;
}

void UpdateTimers(){
	if (physicsTimer>0){
		physicsTimer = physicsTimer - deltaTime;
	}
}

void ClearColor(float r, float g, float b, float a){
	glClearColor(r, g, b, a);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void TogglePhysics(){
	if (physicsTimer <= 0.0f){
		runPhysics = !runPhysics;
		physicsTimer = 0.35f;
	}
}
int main(){

	Scheduler::Start({ Run, nullptr });
	SoulShutDown();
	return 0;
}