#include "ImguiBackend.h"

#include <imgui.h>


ImguiBackend::ImguiBackend()
{

	ImGui::CreateContext();

}

ImguiBackend::~ImguiBackend()
{

	ImGui::DestroyContext();

}

void ImguiBackend::Update()
{

	ImGuiIO& inputInfo = ImGui::GetIO();

}
