
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

/////////////////////////Variables///////////////////////////////

GLuint seed;
RenderType renderer;
WindowType window;

GLFWwindow* mainThread;
GLFWwindow* loopThread;

glm::uvec2 SCREEN_SIZE;

Settings* settings;
unsigned int MSAASamples;
//BVH* hub;
Camera* camera;


bool freeCam;

float timeModifier = 1.0f;
const float deltaTime = (1.0f / 60.0f);
bool runPhysics;
bool freeMouse;
void togglePhysics();
void previousRenderer();
void nextRenderer();
float physicsTimer;
float renderSwitchTimer;
glm::vec2 mouseChangeDegrees;

/////////////////////////User Interface///////////////////////////

//Initializes Soul. This must be called before using variables or 
//any other functions relating to the engine.
void SoulInit(){
seed = GLuint(time(NULL));
srand(seed);
settings = new Settings("Settings.ini");
physicsTimer = 0;
renderSwitchTimer = 0;

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

//msaa samples

std::string msAA = GetSetting("MSAA");
if (msAA.compare("") != 0){
	MSAASamples = atoi(msAA.c_str());
}
else{
	SetSetting("MSAA", std::to_string(16));
	MSAASamples = 16;
}


Scheduler::Pool((std::thread::hardware_concurrency()-1));

}

//Call to deconstuct both the engine and its dependencies
void SoulTerminate(){

	//delete hub;
	delete settings;
	glfwTerminate();

	exit(0);
}
void AddObject(Object* object){
	//hub->Add(object);

}
void RemoveObject(Object* object){

	
}
void SoulCreateWindow(WindowType windowT, RenderType rendererT){

	freeMouse = false;

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
		mainThread = glfwCreateWindow(mode->width, mode->height, "Soul Engine", NULL, loopThread);

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

	// setup camera 
	freeCam = true;

	camera->SetPosition(glm::vec3(-(DECAMETER * 5), DECAMETER * 5, -(DECAMETER * 5)));
	camera->OffsetOrientation(135, 45);

	//unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
	//threads = new ThreadPool(concurentThreadsSupported);

	//hub = new BVH();

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

void Run(void)
{
	
	
	SoulInit();
	cudaDeviceSynchronize();
	camera = new Camera();

	SoulCreateWindow(BORDERLESS, SPECTRAL);

	Renderer rend(*camera,SCREEN_SIZE);
	

	SetKey(GLFW_KEY_ESCAPE, std::bind(&SoulTerminate));
	SetKey(GLFW_KEY_SPACE, std::bind(&togglePhysics));
	SetKey(GLFW_KEY_Q, std::bind(&previousRenderer));
	SetKey(GLFW_KEY_E, std::bind(&nextRenderer));

	//timer info for loop
	double t = 0.0f;
	double currentTime = glfwGetTime();
	double accumulator = 0.0f;

	glfwPollEvents();
	//hub->UpdateObjects(deltaTime);
	//stop loop when glfw exit is called
	glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0f, SCREEN_SIZE.y / 2.0f);
	while (!glfwWindowShouldClose(mainThread)){
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
			double xPos;
			double yPos;
			glfwGetCursorPos(mainThread, &xPos, &yPos);
			xPos -= (SCREEN_SIZE.x / 2.0);
			yPos -= (SCREEN_SIZE.y / 2.0);
			CudaCheck(cudaDeviceSynchronize());
			mouseChangeDegrees.x = (float)(xPos / SCREEN_SIZE.x * camera->FieldOfView().x);
			mouseChangeDegrees.y = (float)(yPos / SCREEN_SIZE.y * camera->FieldOfView().y);
			
			if (freeMouse){
				if (freeCam){
					//set camera for each update
					camera->OffsetOrientation(mouseChangeDegrees.x, mouseChangeDegrees.y);
				}


				glfwSetCursorPos(mainThread, SCREEN_SIZE.x / 2.0f, SCREEN_SIZE.y / 2.0f);
			}


			glfwPollEvents();


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



			//double timeSet = glfwGetTime();

			////hub->SetupObjects();

			//currentTime = glfwGetTime() - currentTime;

			//std::cout << "Setup Objects(ms): " << (float)currentTime * 1000 << std::endl;

			//if (runPhysics){

			//timeSet = glfwGetTime();

			////hub->Physics(deltaTime);

			//currentTime = glfwGetTime() - currentTime;

			//std::cout << "Physics(ms): " << (float)currentTime * 1000 << std::endl;
			//	
			////}
			//timeSet = glfwGetTime();

			////hub->UpdateObjects(deltaTime);

			//currentTime = glfwGetTime() - currentTime;

			//std::cout << "UpdateObjects(ms): " << (float)currentTime * 1000 << std::endl;

			if (physicsTimer>0){
				physicsTimer = physicsTimer - deltaTime;
			
			}
			if (renderSwitchTimer>0){
				renderSwitchTimer = renderSwitchTimer - deltaTime;

			}

			//timeSet = glfwGetTime();

			////hub->CreateHeirarchy(false, deltaTime);

			//currentTime = glfwGetTime() - currentTime;

			//std::cout << "CreateHeirarchy(ms): " << (float)currentTime * 1000 << std::endl;


			t += deltaTime;
			accumulator -= deltaTime;
		}


		rend.RenderRequestChange(SCREEN_SIZE, *camera, 1.0f);

		RayEngine::Process();

		//draw
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		cudaDeviceSynchronize();
		rend.Render();
		cudaDeviceSynchronize();
		//rayEngine->Render(SCREEN_SIZE, hub, camera, deltaTime / 2.0f);

		glfwSwapBuffers(mainThread);
	}
}
glm::vec2* GetMouseChange(){
	return &mouseChangeDegrees;
}
void AttachCamera(Character& character){
	/*freeCam = false;
	camera.AttachCamera(&(character.camera));*/
}
void DetachCamera(){
	/*freeCam = true;
	camera.DetachCamera();*/
}

void SetClearColor(float r, float g, float b, float a){
	glClearColor(r, g, b, a);
}
void togglePhysics(){
	if (physicsTimer <= 0){
		runPhysics = !runPhysics;
		physicsTimer = 0.35f;
	}
}
void nextRenderer(){
	if (renderSwitchTimer <= 0){
		if (renderer == PATH){
			renderer = SPECTRAL;
		}
		else if (renderer == SPECTRAL){
			renderer = PATH;
		}
		renderSwitchTimer = 0.5f;
	}
}
void previousRenderer(){
	if (renderSwitchTimer <= 0){
        if (renderer == SPECTRAL){
			renderer = PATH;
		}
		else if (renderer == PATH){
			renderer = SPECTRAL;
		}
		renderSwitchTimer = 0.5f;
	}
}

int main(){
	Run();
	SoulTerminate();
	return 0;
}