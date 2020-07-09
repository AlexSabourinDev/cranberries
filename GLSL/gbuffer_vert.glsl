#version 450 core
#pragma shader_stage(vertex)

#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex
{
    vec4 gl_Position;
};

struct vertex
{
	// UV in w of pos and w of normal
	vec4 pos;
	vec4 normal;
};

vec2 unpack_uv(vertex v)
{
	return vec2(v.pos.w, v.normal.w);
}

layout(set=0, binding=0) buffer mesh_data_t
{
	vertex data[];
} mesh;

layout(push_constant) uniform transforms_t
{
	mat4x4 VP;
	mat3x4 M;
	uint vertexOffset;
} perDraw;

layout(location = 2) out vec3 var_normal;
layout(location = 3) out vec2 var_uv;

void main()
{
	uint index = (perDraw.vertexOffset + gl_VertexIndex);
	mat4 M4 = mat4(perDraw.M);

	mat3 rotation = mat3(M4);

	gl_Position = vec4(mesh.data[index].pos.xyz, 1.0)*M4*perDraw.VP;
	var_normal = mesh.data[index].normal.xyz*rotation;
	var_uv = unpack_uv(mesh.data[index]);
}