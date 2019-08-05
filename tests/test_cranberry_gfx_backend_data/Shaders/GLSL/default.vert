#version 450
#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex
{
    vec4 gl_Position;
};

layout (binding = 0) uniform camera_t
{
    mat4 world_to_view;
	mat4 view_to_projection;
} camera;

layout (location = 0) in vec3 in_Position;

void main()
{
	gl_Position = vec4(in_Position, 1.0) * camera.world_to_view * camera.view_to_projection;
}