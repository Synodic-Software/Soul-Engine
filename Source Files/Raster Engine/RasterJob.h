#pragma once

#include <map>
#include <string>
#include <vector>
#include "Shader.h"

#include "boost\variant.hpp"
#include "Metrics.h"
#include "Utility\Includes\GLMIncludes.h"


typedef boost::variant<
	int, 
	float, 
	double, 
	bool, 
	uint, 
	glm::mat4, 
	glm::vec3, 
	glm::uvec3, 
	glm::vec4, 
	glm::uvec4, 
	glm::vec2, 
	glm::uvec2
> RasterVariant;

class RasterJob {
public:
	RasterJob();
	~RasterJob();

	RasterVariant const& operator [](std::string i) const;
	RasterVariant& operator [](std::string i);

	virtual void AttachShaders(const std::vector<Shader*>&)=0;

protected:
	std::map<std::string, RasterVariant> shaderUniforms;
	std::vector<Shader*> shaders;
private:
	
};