#include "ImguiBackend.h"

#include "WindowParameters.h"
#include "Display/Window/WindowModule.h"
#include "Display/Window/Window.h"
#include "Display/Input/InputModule.h"
#include "Render/RenderGraph/RenderGraphModule.h"

#include <imgui.h>

//TODO: temporary include
#include "Core/Geometry/Vertex.h"


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




	renderGraphModule_->CreatePass("GUI", [&](EntityWriter& writer) {

		writer;

		return [=](EntityReader& reader, CommandList& commandList) {


			////this is the temporary hardcoded square that what previous hardcoded into the pipeline. 
			////Step-by-step development and refactoring :)

			//std::vector<Vertex> vertices(4);
			//vertices[0].position = {-0.5f, -0.5f, 0.0f};
			//vertices[1].position = {0.5f, -0.5f, 0.0f};
			//vertices[2].position = {0.5f, 0.5f, 0.0f};
			//vertices[3].position = {-0.5f, 0.5f, 0.0f};

			//const std::vector<uint16> indices = {0, 1, 2, 2, 3, 0};

			//// TODO: temporary, refactor
			//VulkanBuffer<Vertex> vertexBuffer_;
			//VulkanBuffer<Vertex> vertexStagingBuffer_;

			//VulkanBuffer<uint16> indexBuffer_;
			//VulkanBuffer<uint16> indexStagingBuffer_;

			//vertexBuffer_(4,
			//vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			//vk::MemoryPropertyFlagBits::eDeviceLocal, device_),
			//vertexStagingBuffer_(4, vk::BufferUsageFlagBits::eTransferSrc,
			//	vk::MemoryPropertyFlagBits::eHostVisible |
			//		vk::MemoryPropertyFlagBits::eHostCoherent,
			//	device_),
			//indexBuffer_(6,
			//	vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			//	vk::MemoryPropertyFlagBits::eDeviceLocal, device_),
			//indexStagingBuffer_(6, vk::BufferUsageFlagBits::eTransferSrc,
			//	vk::MemoryPropertyFlagBits::eHostVisible |
			//		vk::MemoryPropertyFlagBits::eHostCoherent,
			//	device_)

			//Vertex* data = vertexStagingBuffer_.Map();
			//std::memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
			//vertexStagingBuffer_.UnMap();

			//uint16* data2 = indexStagingBuffer_.Map();
			//std::memcpy(data2, indices.data(), sizeof(uint16) * indices.size());
			//indexStagingBuffer_.UnMap();

			//// copy buffer
			//{

			//	VulkanCommandBuffer commandBuffer(, device_, vk::CommandBufferLevel::ePrimary,
			//		vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

			//	commandBuffer.Begin();

			//	vk::BufferCopy copyRegion;
			//	copyRegion.size = sizeof(Vertex) * vertices.size();
			//	commandBuffer.GetCommandBuffer().copyBuffer(
			//		vertexStagingBuffer_.GetBuffer(), vertexBuffer_.GetBuffer(), 1, &copyRegion);

			//	commandBuffer.End();
			//	commandBuffer.Submit();
			//}

			//{
			//	VulkanCommandBuffer commandBuffer(, device_, vk::CommandBufferLevel::ePrimary,
			//		vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

			//	commandBuffer.Begin();

			//	vk::BufferCopy copyRegion;
			//	copyRegion.size = sizeof(Vertex) * indices.size();
			//	commandBuffer.GetCommandBuffer().copyBuffer(
			//		indexStagingBuffer_.GetBuffer(), indexBuffer_.GetBuffer(), 1, &copyRegion);

			//	commandBuffer.End();
			//	commandBuffer.Submit();
			//}






			return;

			//What follows will be the actual imgui render task


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