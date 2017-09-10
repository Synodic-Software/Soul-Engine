#pragma once

#include <string>
#include "Utility\Includes\GLFWIncludes.h"

/* Values that represent shader ts. */
enum shader_t {VERTEX_SHADER,FRAGMENT_SHADER};

/* A shader. */
class Shader {
public:

	/*
	 *    Constructor.
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 */

	Shader(std::string, shader_t);
	/* Destructor. */
	virtual ~Shader();

	/*
	 *    Extracts the shader described by parameter1.
	 *    @param	parameter1	The first parameter.
	 */

	void ExtractShader(const std::string&);

protected:

	/* The name */
	std::string name;
	/* The code string */
	std::string codeStr;
	/* The type */
	shader_t type;
	/* Number of references */
	unsigned* referenceCount;
	/* Retains this object. */
	void Retain();
	/* Releases this object. */
	void Release();

private:
	

};