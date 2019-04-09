#pragma once

#include "GUI/GUIModule.h"


class ImguiBackend final : public GUIModule {

public:

	ImguiBackend() = default;
	~ImguiBackend() override = default;

	ImguiBackend(const ImguiBackend&) = delete;
	ImguiBackend(ImguiBackend&&) noexcept = default;

	ImguiBackend& operator=(const ImguiBackend&) = delete;
	ImguiBackend& operator=(ImguiBackend&&) noexcept = default;


};

