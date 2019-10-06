#version 450
#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex
{
    vec4 gl_Position;
};

struct projection_t
{
	float x;
	float y;
	float zA;
	float zB;
};

layout(set=0, binding=0) buffer mesh_data_t
{
	uint mask;
	float data[];
} mesh;

layout(push_constant) uniform transforms_t
{
	mat4x4 viewMatrix;
	projection_t projection;
} transformations;

void main()
{
	uint index = gl_VertexIndex * 3;

	vec4 pos = vec4(mesh.data[index], mesh.data[index + 1], mesh.data[index + 2], 1.0);

	pos = inverse(transformations.viewMatrix) * pos;
	float z = pos.z;
	pos = vec4(pos.x * transformations.projection.x, pos.y * transformations.projection.y, pos.z * transformations.projection.zA + transformations.projection.zB, z);
	pos /= z;
	gl_Position = pos;
}