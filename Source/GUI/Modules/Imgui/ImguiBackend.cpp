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