#version 450
#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex
{
    vec4 gl_Position;
};

struct vertex
{
	vec4 pos;
	vec4 normal;
};

layout(set=0, binding=0) buffer mesh_data_t
{
	vertex data[];
} mesh;

layout(push_constant) uniform transforms_t
{
	mat4x4 VP;
	mat3x4 M;
	uint vertexOffset;
} transformations;

layout(location = 1) out vec3 var_normal;

void main()
{
	uint index = (transformations.vertexOffset + gl_VertexIndex);
	mat4 M4 = mat4(transformations.M);

	mat3 rotation = mat3(M4);

	gl_Position = vec4(mesh.data[index].pos.xyz, 1.0)*M4*transformations.VP;
	var_normal = mesh.data[index].normal.xyz*rotation;
}