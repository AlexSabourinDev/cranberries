#version 450 core
#extension GL_ARB_separate_shader_objects : enable

#pragma shader_stage( fragment )

#define crang_max_image_count 100

layout(location = 0) out vec4 out_gbuffer0;
layout(location = 1) out vec4 out_gbuffer1;

layout(set = 1, binding = 0) uniform material_data_t
{
	vec4 albedoTint;
} material;

layout(set = 0, binding = 1) uniform sampler textureSampler;
layout(set = 0, binding = 2) uniform texture2D textures[crang_max_image_count];

layout(location = 2) in vec3 var_normal;
layout(location = 3) in vec2 var_uv;

layout(push_constant) uniform transforms_t
{
	layout(offset = 116) uint textureIndex;
} perDraw;

vec4 pack_normal(vec3 normal)
{
	return vec4(var_normal.xyz*0.5 + 0.5,1.0);
}

void main()
{
	out_gbuffer0 = texture(sampler2D(textures[perDraw.textureIndex], textureSampler), var_uv);
	out_gbuffer1 = pack_normal(var_normal);
}