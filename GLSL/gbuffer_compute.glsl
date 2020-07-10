#version 450 core
#pragma shader_stage(compute)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_image_load_store : enable

layout (set = 0, binding = 0, rgba8) uniform readonly image2D gbuffers[2];
layout (set = 0, binding = 1) uniform writeonly image2D swapchain;

vec3 unpack_normal(vec4 normalBuffer)
{
	vec3 normal = vec3(normalBuffer.x, normalBuffer.y, normalBuffer.z) * 2.0f - 1.0;
	return normal;
}

void main()
{
	vec4 gbuffer0Load = imageLoad(gbuffers[0], ivec2(gl_GlobalInvocationID.xy));
	vec4 gbuffer1Load = imageLoad(gbuffers[1], ivec2(gl_GlobalInvocationID.xy));

	vec3 albedo = gbuffer0Load.rgb;
	vec3 lightDir = normalize(vec3(0.707, 0.707, -0.707));
	vec3 normal = unpack_normal(gbuffer1Load);

	vec4 color = vec4(albedo,1.0)*max(dot(normal, lightDir),0.1)*gbuffer1Load.a;
	imageStore(swapchain, ivec2(gl_GlobalInvocationID.xy), color);
}