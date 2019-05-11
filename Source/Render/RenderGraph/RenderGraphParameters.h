#pragma once

#include <string>

struct RenderTaskParameters {

public:
	RenderTaskParameters() = default;
	~RenderTaskParameters() = default;

	std::string name;
};

struct RenderGraphOutputParameters {

public:
	RenderGraphOutputParameters() = default;
	~RenderGraphOutputParameters() = default;

	std::string name;
};

struct RenderGraphInputParameters {

public:
	RenderGraphInputParameters() = default;
	~RenderGraphInputParameters() = default;

	std::string name;
};
