#pragma once

#include "Shader.h"
#include <vector>
#include <glm/glm.hpp>

namespace shading {

    /**
     Represents an OpenGL program made by linking shaders.
     */
    class ShaderSupport { 
    public:
        /**
         Creates a program by linking a list of tdogl::Shader objects
         
         @param shaders  The shaders to link together to make the program
         
         @throws std::exception if an error occurs.
         
         @see tdogl::Shader
         */
        ShaderSupport(const std::vector<Shader>& shaders);
        ~ShaderSupport();
        
        
        /**
         @result The program's object ID, as returned from glCreateProgram
         */
        GLuint object() const;

        void use() const;

        void stopUsing() const;
        
        /**
         @result The attribute index for the given name, as returned from glGetAttribLocation.
         */
        GLint attrib(const GLchar* attribName) const;
        
        
        /**
         @result The uniform index for the given name, as returned from glGetUniformLocation.
         */
        GLint uniform(const GLchar* uniformName) const;

        /**
         Setters for attribute and uniform variables.

         These are convenience methods for the glVertexAttrib* and glUniform* functions.
         */
#define _SHADING_SHADERSUPPORT_ATTRIB_N_UNIFORM_SETTERS(OGL_TYPE) \
        void setAttrib(const GLint pos, OGL_TYPE v0); \
        void setAttrib(const GLint pos, OGL_TYPE v0, OGL_TYPE v1); \
        void setAttrib(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2); \
        void setAttrib(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2, OGL_TYPE v3); \
\
        void setAttrib1v(const GLint pos, const OGL_TYPE* v); \
        void setAttrib2v(const GLint pos, const OGL_TYPE* v); \
        void setAttrib3v(const GLint pos, const OGL_TYPE* v); \
        void setAttrib4v(const GLint pos, const OGL_TYPE* v); \
\
        void setUniform(const GLint pos, OGL_TYPE v0); \
        void setUniform(const GLint pos, OGL_TYPE v0, OGL_TYPE v1); \
        void setUniform(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2); \
        void setUniform(const GLint pos, OGL_TYPE v0, OGL_TYPE v1, OGL_TYPE v2, OGL_TYPE v3); \
\
        void setUniform1v(const GLint pos, const OGL_TYPE* v, GLsizei count=1); \
        void setUniform2v(const GLint pos, const OGL_TYPE* v, GLsizei count=1); \
        void setUniform3v(const GLint pos, const OGL_TYPE* v, GLsizei count=1); \
        void setUniform4v(const GLint pos, const OGL_TYPE* v, GLsizei count=1); \

        _SHADING_SHADERSUPPORT_ATTRIB_N_UNIFORM_SETTERS(GLfloat)
        _SHADING_SHADERSUPPORT_ATTRIB_N_UNIFORM_SETTERS(GLdouble)
        _SHADING_SHADERSUPPORT_ATTRIB_N_UNIFORM_SETTERS(GLint)
        _SHADING_SHADERSUPPORT_ATTRIB_N_UNIFORM_SETTERS(GLuint)

		void setUniformMatrix2(const GLint pos, const GLfloat* v, GLsizei count = 1, GLboolean transpose = GL_FALSE);
		void setUniformMatrix3(const GLint pos, const GLfloat* v, GLsizei count = 1, GLboolean transpose = GL_FALSE);
		void setUniformMatrix4(const GLint pos, const GLfloat* v, GLsizei count = 1, GLboolean transpose = GL_FALSE);
		void setUniform(const GLint pos, const glm::mat2& m, GLboolean transpose = GL_FALSE);
		void setUniform(const GLint pos, const glm::mat3& m, GLboolean transpose = GL_FALSE);
		void setUniform(const GLint pos, const glm::mat4& m, GLboolean transpose = GL_FALSE);

        
    private:
        GLuint _object;
        
        //copying disabled
        ShaderSupport(const ShaderSupport&);
        const ShaderSupport& operator=(const ShaderSupport&);
    };

}

shading::ShaderSupport*  LoadShaders(const char* vertFilename, const char* controlFilename, const char* evaluationFilename, const char* geometryFilename, const char* fragFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(controlFilename, GL_TESS_CONTROL_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(evaluationFilename, GL_TESS_EVALUATION_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(geometryFilename, GL_GEOMETRY_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
	return new shading::ShaderSupport(shaders);
}

shading::ShaderSupport*  LoadShaders(const char* vertFilename, const char* fragFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
	return new shading::ShaderSupport(shaders);
}
shading::ShaderSupport*  LoadShaders(const char* compFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(compFilename, GL_COMPUTE_SHADER));
	return new shading::ShaderSupport(shaders);
}
shading::ShaderSupport*  LoadShaders(const char* vertFilename, const char* geometryFilename, const char* fragFilename) {
	std::vector<shading::Shader> shaders;
	shaders.push_back(shading::Shader::shaderFromFile(vertFilename, GL_VERTEX_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(geometryFilename, GL_GEOMETRY_SHADER));
	shaders.push_back(shading::Shader::shaderFromFile(fragFilename, GL_FRAGMENT_SHADER));
	return new shading::ShaderSupport(shaders);
}