#include "ImguiBackend.h"

#include "WindowParameters.h"
#include "Display/Window/WindowModule.h"
#include "Display/Window/Window.h"
#include "Display/Input/InputModule.h"
#include "Render/RenderGraph/RenderGraphModule.h"

#include <imgui.h>

ImguiBackend::ImguiBackend(std::shared_ptr<InputModule>& inputModule,
	std::shared_ptr<WindowModule>& windowModule,
	std::shared_ptr<RenderGraphModule>& renderGraphModule):
	inputModule_(inputModule),
	windowModule_(windowModule), 
	renderGraphModule_(renderGraphModule)
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

	// TODO: actual upload




	renderGraphModule_->CreatePass("GUI", [&](EntityWriter&) {


		return [=](EntityReader&, CommandList& commandList) {

			// Upload raster data
			ImDrawData* drawData = ImGui::GetDrawData();

			uint vertexBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
			uint indexBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

			if (vertexBufferSize == 0 || indexBufferSize == 0) {

				return;
			}

			// TODO: actual data upload

			// TODO: imgui push constants

			int vertexOffset = 0;
			int indexOffset = 0;

			if (drawData->CmdListsCount > 0) {

				for (int32 i = 0; i < drawData->CmdListsCount; i++) {

					const ImDrawList* imguiCommands = drawData->CmdLists[i];

					for (int32 j = 0; j < imguiCommands->CmdBuffer.Size; j++) {

						const ImDrawCmd* command = &imguiCommands->CmdBuffer[j];

						DrawCommand drawParameters;

						// TODO: fill with imgui scissor and draw info


						commandList.Draw(drawParameters);

						indexOffset += command->ElemCount;
					}

					vertexOffset += imguiCommands->VtxBuffer.Size;
				}
			}
		};
	});

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
	ConvertRetained();
	ImGui::Render();

}

void ImguiBackend::ConvertRetained()
{

	//TODO: Convert retained framework to dear imgui intermediate
	//TODO: Remove hardcoded gui

	if (ImGui::BeginMainMenuBar()) {
		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Exit")) {

				//TODO: Call an exit command

				ImGui::EndMenu();
			}
		}

		ImGui::EndMainMenuBar();
	}

}