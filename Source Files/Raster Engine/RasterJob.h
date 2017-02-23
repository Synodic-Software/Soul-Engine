#pragma once

#include <map>
#include <string>
#include <vector>
#include "Shader.h"

#include "boost\variant.hpp"
#include "Metrics.h"

typedef boost::variant<int, float, double, bool, uint> RasterVariant;

class RasterJob {
public:
	RasterJob();
	~RasterJob();

	RasterVariant const& operator [](std::string i) const;
	RasterVariant& operator [](std::string i);

	virtual void AttachShaders(const std::vector<Shader*>&)=0;

private:
	std::map<std::string, RasterVariant> shaderUniforms;
	std::vector<Shader*> shaders;
};