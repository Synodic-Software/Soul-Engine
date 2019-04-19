#pragma once

#include "Display/GUI/GUIModule.h"


class ImguiBackend final : public GUIModule {

public:

	ImguiBackend();
	~ImguiBackend() override;

	ImguiBackend(const ImguiBackend&) = delete;
	ImguiBackend(ImguiBackend&&) noexcept = default;

	ImguiBackend& operator=(const ImguiBackend&) = delete;
	ImguiBackend& operator=(ImguiBackend&&) noexcept = default;


};

