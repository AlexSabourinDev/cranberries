#version 450
#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex
{
    vec4 gl_Position;
};

layout(set=0, binding=0) buffer mesh_data_t
{
	float data[];
} mesh;

layout(push_constant) uniform transforms_t
{
	mat4x4 MVP;
	uint vertexOffset;
} transformations;

void main()
{
	uint index = (transformations.vertexOffset + gl_VertexIndex) * 3;
	vec4 pos = vec4(mesh.data[index], mesh.data[index + 1], mesh.data[index + 2], 1.0);
	gl_Position = pos;
}