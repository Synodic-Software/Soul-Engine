#pragma once

#include "Display/GUI/GUIModule.h"

class InputModule;

class ImguiBackend final : public GUIModule {

public:

	ImguiBackend(std::shared_ptr<InputModule>&, std::shared_ptr<WindowModule>&);
	~ImguiBackend() override;

	ImguiBackend(const ImguiBackend&) = delete;
	ImguiBackend(ImguiBackend&&) noexcept = default;

	ImguiBackend& operator=(const ImguiBackend&) = delete;
	ImguiBackend& operator=(ImguiBackend&&) noexcept = default;


	void Update(std::chrono::nanoseconds) override;
	void Draw() override;

private:

	std::shared_ptr<InputModule> inputModule_;
	std::shared_ptr<WindowModule> windowModule_;


};

