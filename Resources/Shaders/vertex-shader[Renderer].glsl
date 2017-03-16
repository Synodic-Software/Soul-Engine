#version 430 core
in vec4 vert_VS_in;

uniform mat4 camera;
uniform mat4 model;

void main()
{
	gl_Position=(camera*model)*vert_VS_in;
}