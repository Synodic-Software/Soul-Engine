#include "RenderWidget.h"
#include "Raster Engine\Buffer.h"
#include "GPGPU\GPUManager.h"
#include "Utility/CUDA/CudaHelper.cuh"
#include "CUDA\RenderWidget.cuh"
#include <iostream>
#include "Input/InputState.h"
#include "Events\EventManager.h"

RenderWidget::RenderWidget(Camera* cameraIn)
{
	camera = cameraIn;

	widgetJob = RasterBackend::CreateJob();

	samples = 1.0f;

	//attach shaders to render a quad and apply a texture
	widgetJob->AttachShaders({
		RasterBackend::CreateShader("Resources/Shaders/vertex-shader[Renderer].glsl",VERTEX_SHADER),
		RasterBackend::CreateShader("Resources/Shaders/fragment-shader[Renderer].glsl",FRAGMENT_SHADER)
	});


	//init all uniform data
	widgetJob->RegisterUniform(std::string("screen"));

	uint indices[6];
	float vertices[6 * 4];

	vertices[0] = -1.0f;
	vertices[1] = -1.0f;
	vertices[2] = 0.0f;
	vertices[3] = 1.0f;


	vertices[4] = 1.0f;
	vertices[5] = 1.0f;
	vertices[6] = 0.0f;
	vertices[7] = 1.0f;


	vertices[8] = -1.0f;
	vertices[9] = 1.0f;
	vertices[10] = 0.0f;
	vertices[11] = 1.0f;


	vertices[12] = 1.0f;
	vertices[13] = -1.0f;
	vertices[14] = 0.0f;
	vertices[15] = 1.0f;

	indices[0] = 0;
	indices[1] = 2;
	indices[2] = 1;
	indices[3] = 0;
	indices[4] = 1;
	indices[5] = 3;

	widgetJob->UploadGeometry(vertices, sizeof(vertices), indices, sizeof(indices));

	iCounter = 1;
	integrate = false;
	currentSize = glm::uvec2(312,720);

	uint id = EventManager::Listen("EarlyFrameUpdate", this,&RenderWidget::EarlyFrameUpdate);
	id = EventManager::Listen("LateFrameUpdate", this,&RenderWidget::LateFrameUpdate);

}

void RenderWidget::EarlyFrameUpdate() {

}
void RenderWidget::LateFrameUpdate() {

}

RenderWidget::~RenderWidget()
{

}

void RenderWidget::Draw() {

	InputState::GetInstance().SetKey(GLFW_KEY_SPACE, [&integrate = integrate, &time = time](int action) {
		double newTime = glfwGetTime();
		if (newTime - time > 0.3f) {
			integrate = !integrate;
			time = newTime;
		}

	});


	if (integrate) {
		Integrate(renderSize.x*renderSize.y, (glm::vec4*)buffer->GetData(), (glm::vec4*)accumulator->GetData(), (int*)extraData->GetData(), iCounter);
		iCounter++;
	}
	else {
		iCounter = 1;
	}

	buffer->UnmapResources();
	buffer->BindData(0);
	widgetJob->Draw();

	//add the rayJob back in
	buffer->MapResources();

	//get job values
	widgetJob->SetUniform(std::string("screen"), renderSize);

	RayEngine::ModifyJob(rayJob,*camera);
}

void RenderWidget::RecreateData() {

	//remove the rayJob if it exists
	RayEngine::RemoveJob(rayJob);

	//create the new accumulation Buffer
	accumulator = GPUManager::CreateBuffer(GPUManager::GetBestGPU(), size.x*size.y * sizeof(glm::vec4));

	buffer = GPUManager::CreateRasterBuffer(GPUManager::GetBestGPU(), size.x*size.y * sizeof(glm::vec4));

	extraData = GPUManager::CreateBuffer(GPUManager::GetBestGPU(), size.x*size.y * sizeof(int));


	if (currentSize != size) {
		currentSize = size;
		renderSize = size;
		widgetJob->SetUniform(std::string("screen"), renderSize);
	}

	//update the camera
	camera->SetAspect(renderSize.x / (float)renderSize.y);
	camera->resolution = renderSize;

	//add the ray job with new sizes
	buffer->MapResources();
	rayJob = RayEngine::AddJob(RayCOLOUR, renderSize.x*renderSize.y, true,samples, *camera, buffer->GetData(), (int*)extraData->GetData());
}
