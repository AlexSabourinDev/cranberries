#version 450
#pragma shader_stage(compute)

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_image_load_store : enable

layout (set = 0, binding = 0, rgba8) uniform readonly image2D gbuffer;
layout (set = 0, binding = 1) uniform writeonly image2D swapchain;

vec3 unpack_normal(vec4 normalBuffer)
{
	vec3 normal = vec3(normalBuffer.x, normalBuffer.y, 0.0) * 2.0f - 1.0;
	normal.z = sqrt(1.0 - normal.x * normal.x - normal.y * normal.y);
	normal.z = normalBuffer.z >= 0.5 ? -normal.z : normal.z;
	return normal;
}

void main()
{
	vec4 gbufferLoad = imageLoad(gbuffer, ivec2(gl_GlobalInvocationID.xy));

	vec3 lightDir = normalize(vec3(0.707, 0.707, -0.707));
	vec3 normal = unpack_normal(gbufferLoad);

	vec4 color = vec4(0.1,0.5,0.8,1.0)*max(dot(normal, lightDir),0.0)*gbufferLoad.a;
	imageStore(swapchain, ivec2(gl_GlobalInvocationID.xy), color);
}