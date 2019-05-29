#include "ImguiBackend.h"

#include "WindowParameters.h"
#include "Display/Window/WindowModule.h"
#include "Display/Window/Window.h"
#include "Display/Input/InputModule.h"
#include "Render/RenderGraph/RenderGraphModule.h"

#include <imgui.h>

struct PushBlock {

	glm::vec2 scale;
	glm::vec2 translate;

} pushBlock;

ImguiBackend::ImguiBackend(std::shared_ptr<InputModule>& inputModule,
	std::shared_ptr<WindowModule>& windowModule,
	std::shared_ptr<RenderGraphModule>& renderGraphModule):
	inputModule_(inputModule),
	windowModule_(windowModule), renderGraphModule_(renderGraphModule)
{

	ImGui::CreateContext();

	// Set callbacks
	inputModule_->AddMousePositionCallback([](double xPos, double yPos) {
		ImGuiIO& inputInfo = ImGui::GetIO();

		inputInfo.MousePos = ImVec2(static_cast<float>(xPos), static_cast<float>(yPos));
	});

	inputModule_->AddMouseButtonCallback([](int button, ButtonState state) {
		if (button > 1) {
			return;
		}

		ImGuiIO& inputInfo = ImGui::GetIO();

		inputInfo.MouseDown[button] = state == ButtonState::PRESS;
	});


	ImGuiIO& inputInfo = ImGui::GetIO();

	// TODO: abstract fonts
	// Grab and upload font data
	unsigned char* fontData;
	int textureWidth, textureHeight;
	inputInfo.Fonts->GetTexDataAsRGBA32(&fontData, &textureWidth, &textureHeight);

	// TODO: Create font buffers

	RenderTaskParameters params;
	params.name = "GUI";

	renderGraphModule_->CreateTask(params, [](RenderGraphBuilder& builder) {
		Entity resources = builder.CreateGroup(ResourceGroupType::Default);

		builder.Request<VertexBuffer>(resources);
		builder.Request<IndexBuffer>(resources);
		builder.Request<PushBuffer>(resources);
		builder.Request<RenderView>(resources);

		RenderGraphOutputParameters outputParams;
		outputParams.name = "Final";

		builder.CreateOutput(outputParams);

		return [=](const EntityRegistry& registry, CommandList& commandList) {
			auto& renderView = registry.GetComponent<RenderView>(resources);
			auto& pushConstant = registry.GetComponent<PushBuffer>(resources);

			// Input
			{

				ImGuiIO& io = ImGui::GetIO();
				renderView.width = io.DisplaySize.x;
				renderView.height = io.DisplaySize.y;
				renderView.minDepth = 0.0f;
				renderView.maxDepth = 1.0f;

				pushBlock.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
				pushBlock.translate = glm::vec2(-1.0f);

				UpdateBufferCommand updatePushParameters;
				pushConstant;

				commandList.UpdateBuffer(updatePushParameters);
			}

			ImDrawData* drawData = ImGui::GetDrawData();

			if (drawData->TotalVtxCount == 0 || drawData->TotalIdxCount == 0) {

				return;
			}

			auto& vertexBuffer = registry.GetComponent<VertexBuffer>(resources);
			auto& indexBuffer = registry.GetComponent<IndexBuffer>(resources);

			// Push the data to the buffers
			{
				UpdateBufferCommand updateVertexParameters;
				updateVertexParameters.size = drawData->TotalVtxCount;
				updateVertexParameters.offset = 0;

				UpdateBufferCommand updateIndexParameters;
				updateIndexParameters.size = drawData->TotalIdxCount;
				updateIndexParameters.offset = 0;


				nonstd::span<RenderVertex> vtxDst = vertexBuffer.vertices;
				auto vtxOffest = 0;
				nonstd::span<uint16> idxDst = indexBuffer.indices;
				auto idxOffest = 0;

				for (int32 i = 0; i < drawData->CmdListsCount; i++) {

					const ImDrawList* imguiCommand = drawData->CmdLists[i];

					for (int v = 0; v < imguiCommand->VtxBuffer.Size; ++v) {

						//TODO: Refactor datatypes
						vtxDst[vtxOffest + v].position = glm::vec3(
							imguiCommand->VtxBuffer[v].pos.x, imguiCommand->VtxBuffer[v].pos.y, 0.0f);
						vtxDst[vtxOffest + v].textureCoord = glm::vec2(
							imguiCommand->VtxBuffer[v].uv.x, imguiCommand->VtxBuffer[v].uv.y);
						vtxDst[vtxOffest + v].colour = glm::vec4(imguiCommand->VtxBuffer[v].col/255.0f);

					}

					auto beginIterator = imguiCommand->IdxBuffer.begin() + idxOffest;
					std::copy(beginIterator, beginIterator + imguiCommand->IdxBuffer.Size,
						idxDst.begin() + idxOffest);

					vtxOffest += imguiCommand->VtxBuffer.Size;
					idxOffest += imguiCommand->IdxBuffer.Size;
				}

				// Buffer processing
				commandList.UpdateBuffer(updateVertexParameters);
				commandList.UpdateBuffer(updateIndexParameters);
			}

			// Drawing
			{

				int vertexOffset = 0;
				int indexOffset = 0;

				for (int32 i = 0; i < drawData->CmdListsCount; i++) {

					const ImDrawList* imguiCommands = drawData->CmdLists[i];

					for (int32 j = 0; j < imguiCommands->CmdBuffer.Size; j++) {

						const ImDrawCmd* command = &imguiCommands->CmdBuffer[j];

						DrawCommand drawParameters;

						drawParameters.elementSize = command->ElemCount;
						drawParameters.indexOffset = indexOffset;
						drawParameters.vertexOffset = vertexOffset;
						drawParameters.scissorOffset = {command->ClipRect.x, command->ClipRect.y};
						drawParameters.scissorExtent = {command->ClipRect.z - command->ClipRect.x,
							command->ClipRect.w - command->ClipRect.y};

						drawParameters.vertexBuffer = resources;
						drawParameters.indexBuffer = resources;

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

	// TODO: use the GUI associated window
	// TODO: via callback
	// Update Display
	WindowParameters& windowParams = windowModule_->GetWindow().Parameters();
	inputInfo.DisplaySize = ImVec2(
		static_cast<float>(windowParams.pixelSize.x), static_cast<float>(windowParams.pixelSize.y));
	inputInfo.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

	// TODO: via callback
	// Update frame timings
	auto frameSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(frameTime);
	inputInfo.DeltaTime = frameSeconds.count();


	ImGui::NewFrame();
	ConvertRetained();
	ImGui::Render();
}

void ImguiBackend::ConvertRetained()
{

	// TODO: Convert retained framework to dear imgui intermediate
	// TODO: Remove hardcoded gui

	if (ImGui::BeginMainMenuBar()) {
		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Exit")) {

				// TODO: Call an exit command

				ImGui::EndMenu();
			}
		}

		ImGui::EndMainMenuBar();
	}
}