#include "ImguiBackend.h"

#include "WindowParameters.h"
#include "Display/Window/WindowModule.h"
#include "Display/Window/Window.h"
#include "Display/Input/InputModule.h"
#include "Rasterer/RasterModule.h"

#include <imgui.h>

ImguiBackend::ImguiBackend(std::shared_ptr<InputModule>& inputModule,
	std::shared_ptr<WindowModule>& windowModule,
	std::shared_ptr<RasterModule>& rasterModule):
	inputModule_(inputModule),
	windowModule_(windowModule), 
	rasterModule_(rasterModule)
{

	ImGui::CreateContext();

	//Set callbacks
	inputModule_->AddMousePositionCallback([](double xPos, double yPos) {
		
		ImGuiIO& inputInfo = ImGui::GetIO();

		inputInfo.MousePos = ImVec2(xPos, yPos);

	});

	inputModule_->AddMouseButtonCallback([](int button, ButtonState state) {

		if (button > 1) {
			return;
		}

		ImGuiIO& inputInfo = ImGui::GetIO();

		inputInfo.MouseDown[button] = state == ButtonState::PRESS;

	});



	ImGuiIO& inputInfo = ImGui::GetIO();

	//TODO: abstract fonts
	//Grab and upload font data
	unsigned char* fontData;
	int textureWidth, textureHeight;
	inputInfo.Fonts->GetTexDataAsRGBA32(&fontData, &textureWidth, &textureHeight);

	// TODO: actual rasterModule upload

}

ImguiBackend::~ImguiBackend()
{

	ImGui::DestroyContext();

}

void ImguiBackend::Update(std::chrono::nanoseconds frameTime)
{

	ImGuiIO& inputInfo = ImGui::GetIO();

	//TODO: use the GUI associated window
	//TODO: via callback
	//Update Display
	WindowParameters& windowParams = windowModule_->GetWindow().Parameters();
	inputInfo.DisplaySize = ImVec2(windowParams.pixelSize.x, windowParams.pixelSize.y);
	inputInfo.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

	//TODO: via callback
	//Update frame timings
	auto frameSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(frameTime);
	inputInfo.DeltaTime = frameSeconds.count();


	ImGui::NewFrame();

	//TODO: Convert retained framework to dear imgui intermediate
	//TODO: Remove hardcoded gui


	ImGui::Render();

	//Upload raster data
	ImDrawData* drawData = ImGui::GetDrawData();

	uint vertexBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
	uint indexBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

	if (vertexBufferSize == 0 || indexBufferSize == 0) {

		return;

	}

	//TODO: actual rasterModule upload

}


void ImguiBackend::Draw()
{

	// Record raster commands
	ImDrawData* drawData = ImGui::GetDrawData();

	// TODO: actual rasterModule recording
	rasterModule_->Draw();


}