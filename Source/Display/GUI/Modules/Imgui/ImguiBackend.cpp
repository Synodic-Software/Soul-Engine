#include "ImguiBackend.h"

#include "WindowParameters.h"
#include "Display/Window/WindowModule.h"
#include "Display/Window/Window.h"

#include <imgui.h>

ImguiBackend::ImguiBackend(std::shared_ptr<InputModule>& inputModule,
	std::shared_ptr<WindowModule>& windowModule):
	inputModule_(inputModule),
	windowModule_(windowModule)
{

	ImGui::CreateContext();


	ImGuiIO& inputInfo = ImGui::GetIO();

	//TODO: abstract fonts
	//Grab and upload font data
	unsigned char* fontData;
	int textureWidth, textureHeight;
	inputInfo.Fonts->GetTexDataAsRGBA32(&fontData, &textureWidth, &textureHeight);



}

ImguiBackend::~ImguiBackend()
{

	ImGui::DestroyContext();

}

void ImguiBackend::Update()
{

	ImGuiIO& inputInfo = ImGui::GetIO();

	//TODO: use the GUI associated window
	WindowParameters& windowParams = windowModule_->GetWindow().Parameters();
	inputInfo.DisplaySize = ImVec2(windowParams.pixelSize.x, windowParams.pixelSize.y);
	inputInfo.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

	ImGui::NewFrame();

	//TODO: Convert retained framework to dear imgui intermediate


	ImGui::Render();

}
