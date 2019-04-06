#version 450
#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 fragColor;

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(inPosition.xy, 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}