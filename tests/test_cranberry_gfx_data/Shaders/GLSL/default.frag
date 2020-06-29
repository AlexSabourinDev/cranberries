#version 450
#extension GL_ARB_separate_shader_objects : enable
#pragma shader_stage( fragment )

layout( location = 0 ) out vec4 out_Color;

layout(set = 0, binding = 1) uniform material_data_t
{
	vec4 albedoTint;
} material;

layout(set = 0, binding = 2) uniform sampler2D albedo;

void main()
{
	out_Color = texture(albedo, gl_FragCoord.xy / vec2(100.0, 100.0)) * material.albedoTint;
}