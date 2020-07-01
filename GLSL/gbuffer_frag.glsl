#version 450
#extension GL_ARB_separate_shader_objects : enable
#pragma shader_stage( fragment )

layout( location = 0 ) out vec4 out_Color;

layout(set = 0, binding = 1) uniform material_data_t
{
	vec4 albedoTint;
} material;

layout(set = 0, binding = 2) uniform sampler2D albedo;

layout(location = 1) in vec3 var_normal;

vec4 pack_normal(vec3 normal)
{
	// Encode sign in last bucket of 8 bit channel
	// Formula for storage is [f * 256 = i]
	// We want our high bit of "i" set, so we need our value to give us 128
	// i = f * 256
	// 128 = f * 256
	// 1/2 = f
	uint signBit = floatBitsToUint(var_normal.z) & 0x8000000;
	return vec4(var_normal.xy*0.5 + 0.5, 0.5*signBit,1.0);
}

void main()
{
	out_Color = pack_normal(var_normal);
}