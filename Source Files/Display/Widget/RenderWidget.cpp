#include "RenderWidget.h"
#include "Raster Engine\Buffer.h"
#include "GPGPU\GPUManager.h"
#include "Ray Engine/RayEngine.h"
#include "Utility/CUDA/CudaHelper.cuh"
#include "CUDA\RenderWidget.cuh"
#include <iostream>
#include "Input/InputState.h"

RenderWidget::RenderWidget(Camera* cameraIn)
{
	camera = cameraIn;

	widgetJob = RasterBackend::CreateJob();

	samples = 1;

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
		Integrate(size.x*size.y, (glm::vec4*)buffer->GetData(), (glm::vec4*)accumulator->GetData(), iCounter);
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
	RayEngine::AddRayJob(RayCOLOUR, size.x*size.y, samples, *camera, buffer->GetData());

	camera->currentVert.position = camera->Position();
	CudaCheck(cudaMemcpy(camera->devicePos, &camera->currentVert, sizeof(Vertex), cudaMemcpyHostToDevice));


}

void RenderWidget::RecreateData() {

	//update the camera
	camera->SetAspect(size.x / (float)size.y);
	camera->resolution = size;
	//remove the rayJob if it exists
	//TODO

	//create the new accumulation Buffer
	accumulator = GPUManager::CreateBuffer(GPUManager::GetBestGPU(), size.x*size.y * sizeof(glm::vec4));

	buffer = GPUManager::CreateRasterBuffer(GPUManager::GetBestGPU(), size.x*size.y * sizeof(glm::vec4));

	if (currentSize != size) {
		currentSize = size;
		widgetJob->SetUniform(std::string("screen"), size);
	}

	//add the ray job with new sizes
	buffer->MapResources();
	RayEngine::AddRayJob(RayCOLOUR, size.x*size.y, samples, *camera, buffer->GetData());
}
