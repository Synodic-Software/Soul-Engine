#pragma once

#include <string>

#include "OpenGLBackend.h"
#include "Raster Engine\RasterJob.h"

/* An open gl job. */
class OpenGLJob : public RasterJob {
public:
	/* Default constructor. */
	OpenGLJob();
	/* Destructor. */
	~OpenGLJob();

	/*
	 *    Attach shaders.
	 *    @param	parameter1	The first parameter.
	 */

	void AttachShaders(const std::vector<Shader*>&);

	/*
	 *    Registers the uniform described by std::string.
	 *    @param	std::string	The standard string.
	 */

	void RegisterUniform(const std::string);

	/*
	 *    Uploads a geometry.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param [in,out]	parameter3	If non-null, the third parameter.
	 *    @param 		 	parameter4	The fourth parameter.
	 */

	void UploadGeometry(float*, uint, uint*, uint);

	/*
	 *    Sets an uniform.
	 *    @param	std::string	The standard string.
	 *    @param	parameter2 	The second parameter.
	 */

	void SetUniform(const std::string, RasterVariant);

	/* Draws this object. */
	void Draw();


private:

	/*
	 *    Gets an attribute.
	 *    @param	parameter1	The first parameter.
	 *    @return	The attribute.
	 */

	GLint GetAttribute(const GLchar*);

	/* The vao */
	GLuint vao;
	/* The vbo */
	GLuint vbo;
	/* The ibo */
	GLuint ibo;

	/* The object */
	GLuint object;

	/* Size of the draw */
	uint drawSize;
};