#pragma once

#include <map>
#include <string>
#include <vector>
#include "Shader.h"

#include "boost\variant.hpp"
#include "Metrics.h"
#include "Utility\Includes\GLMIncludes.h"
#include <string>

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

	int const& operator [](std::string i) const;
	int& operator [](std::string i);

	virtual void AttachShaders(const std::vector<Shader*>&)=0;
	virtual void RegisterUniform(const std::string) = 0;
	virtual void SetUniform(const std::string, RasterVariant) = 0;

	virtual void UploadGeometry(float* ,uint ,uint*,uint ) = 0;

	virtual void Draw() = 0;

protected:
	std::map<std::string, int> shaderUniforms;
	std::vector<Shader*> shaders;

private:
	
};