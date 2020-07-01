// Links and things:
// https://patapom.com/blog/BRDF/BRDF%20Models/
// https://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf

// Standard: Z is up

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

#define cran_debug
#ifdef cran_debug
#include <assert.h>
#define cran_assert(a) assert(a)
#else
# define cran_assert(a) (void)(a)
#endif

#define cran_stats
#ifdef cran_stats
#define cran_stat(a) a
#else
#define cran_stat(a)
#endif

#include "stb_image.h"
#include "cranberry_platform.h"
#include "cranberry_loader.h"
#include "cranberry_math.h"

#define CRANPR_IMPLEMENTATION
//#define CRANPR_ENABLED
#include "cranberry_profiler.h"

// Hash a string at compile time, Source on: https://stackoverflow.com/questions/7666509/hash-function-for-string
// Algorithm Source: http://www.cse.yorku.ca/~oz/hash.html by Dan Bernstein
uint32_t djb2(char const* cran_restrict string, uint32_t hash)
{
	return (string[0] == '\0') ? 0 : ((hash << 5) + hash) + string[1] + djb2(&string[1], hash);
}

uint32_t hash(const char* string)
{
	return djb2(string, 5381);
}

// Allocator
typedef struct
{
	// Grows from bottom to top
	uint8_t* mem;
	uint64_t top;
	uint64_t size;
	bool locked;
} crana_stack_t;

void* crana_stack_alloc(crana_stack_t* stack, uint64_t size)
{
	cran_assert(!stack->locked);
	uint8_t* ptr = stack->mem + stack->top;
	stack->top += size + sizeof(uint64_t);
	cran_assert(stack->top <= stack->size);

	*(uint64_t*)ptr = size + sizeof(uint64_t);
	return ptr + sizeof(uint64_t);
}

void crana_stack_free(crana_stack_t* stack, void* memory)
{
	cran_assert(!stack->locked);
	stack->top -= *((uint64_t*)memory - 1);
	cran_assert(stack->top <= stack->size);
}

// Lock our stack, this means we have free range on the memory pointer
// Commit the pointer back to the stack to complete the operation.
// While the stack is locked, alloc and free cannot be called.
void* crana_stack_lock(crana_stack_t* stack)
{
	cran_assert(!stack->locked);
	stack->locked = true;
	return stack->mem + stack->top + sizeof(uint64_t);
}

// Commit our new pointer. You can have moved it forward or backwards
// Make sure that the pointer is still from the same memory pool as the stack
void crana_stack_commit(crana_stack_t* stack, void* memory)
{
	cran_assert(stack->locked);
	cran_assert((uint64_t)((uint8_t*)memory - stack->mem) <= stack->size);
	*(uint64_t*)(stack->mem + stack->top) = (uint8_t*)memory - (stack->mem + stack->top);
	stack->top = (uint8_t*)memory - stack->mem;
	stack->locked = false;
}

// Cancel our lock, all the memory we were planning on commiting can be ignored
void crana_stack_revert(crana_stack_t* stack)
{
	cran_assert(stack->locked);
	stack->locked = false;
}

typedef struct
{
	uint64_t rayCount; // TODO: only reference through atomics
	uint64_t primaryRayCount;
	uint64_t totalTime;
	uint64_t sceneGenerationTime;
	uint64_t renderTime;
	uint64_t intersectionTime;
	uint64_t bvhTraversalTime;
	uint64_t bvhHitCount;
	uint64_t bvhLeafHitCount;
	uint64_t bvhMissCount;
	uint64_t bvhNodeCount;
	uint64_t skyboxTime;
	uint64_t imageSpaceTime;
} render_stats_t;

static void merge_render_stats(render_stats_t* base, render_stats_t const* add)
{
	base->rayCount += add->rayCount;
	base->primaryRayCount += add->primaryRayCount;
	base->totalTime += add->totalTime;
	base->sceneGenerationTime += add->sceneGenerationTime;
	base->renderTime += add->renderTime;
	base->intersectionTime += add->intersectionTime;
	base->bvhTraversalTime += add->bvhTraversalTime;
	base->bvhHitCount += add->bvhHitCount;
	base->bvhLeafHitCount += add->bvhLeafHitCount;
	base->bvhMissCount += add->bvhMissCount;
	base->bvhNodeCount += add->bvhNodeCount;
	base->skyboxTime += add->skyboxTime;
	base->imageSpaceTime += add->imageSpaceTime;
}

typedef struct
{
	uint32_t maxDepth;
	uint32_t samplesPerPixel;
	uint32_t renderWidth;
	uint32_t renderHeight;
	bool useDirectionalMat; // Force Shader To Directional
} render_config_t;
render_config_t renderConfig;

typedef uint32_t random_seed_t;
typedef struct
{
	random_seed_t randomSeed;
	uint32_t depth;
	crana_stack_t stack;
	crana_stack_t scratchStack;

	render_stats_t renderStats;
} render_context_t;

static float micro_to_seconds(uint64_t time)
{
	return (float)time / 1000000.0f;
}


uint32_t lcg_parkmiller(uint32_t* state)
{
	// https://en.wikipedia.org/wiki/Lehmer_random_number_generator
	uint64_t product = (uint64_t)(*state) * 48271;
	uint32_t x = (uint32_t)(product & 0x7fffffff) + (uint32_t)(product >> 31);
	x = (x & 0x7fffffff) + (x >> 31);
	*state = x;
	return x;
}

static uint32_t random(random_seed_t* seed)
{
	return lcg_parkmiller(seed);
}

static float random01f(random_seed_t* seed)
{
	// http://www.iquilezles.org/www/articles/sfrand/sfrand.htm
	union
	{
		float f;
		uint32_t u;
	} res;

	res.u = ((lcg_parkmiller(seed)>>8) | 0x3f800000);
	return res.f-1.0f;
}

static uint32_t randomRange(random_seed_t* seed, uint32_t min, uint32_t max)
{
	uint32_t result = random(seed) % (max - min);
	return result + min;
}

static float rgb_to_luminance(float r, float g, float b)
{
	return (0.2126f*r + 0.7152f*g + 0.0722f*b);
}

static bool sphere_does_ray_intersect(cv3 rayO, cv3 rayD, float sphereR)
{
	float projectedDistance = -cv3_dot(rayO, rayD);
	float distanceToRaySqr = cv3_dot(rayO,rayO) - projectedDistance * projectedDistance;
	return (distanceToRaySqr < sphereR * sphereR);
}

static float sphere_ray_intersection(cv3 rayO, cv3 rayD, float rayMin, float rayMax, float sphereR)
{
	// Calculate our intersection distance
	// With the sphere equation: dot(P-O,P-O) = r^2
	// With the ray equation: P = V * d + A (Where A is the origin of our ray)
	// With P - O = V * d + A - O
	// With C = V, D = A - O
	// P - O = C * d + D
	// The sphere equation becomes dot(C * d + D, C * d + D) = r^2
	// Expanding
	// (Cx * d + Dx)^2 + (Cy * d + Dy)^2 = r^2
	// Cx^2*d^2 + 2*Cx*Dx*d + Dx^2 + Cy^2*d^2 + 2*Cy*Dy*d + Dy^2 = r^2
	// Collecting like terms
	// (Cx^2*d^2 + Cy^2*d^2) + (2*Cx*Dx*d + 2*Cy*Dy*d) + (Dx^2 + Dy^2 - r^2) = 0
	// Pull out d
	// d^2 * (Cx^2+Cy^2) + d * (2*Cx*Dx + 2*Cy*Dy)  + (Dx^2 + Dy^2 - r^2) = 0
	// Rename
	// a = (Cx^2+Cy^2), b = (2*Cx*Dx + 2*Cy*Dy), c = (Dx^2 + Dy^2 - r^2)
	// d^2 * a + d * b + c = 0
	// Solve for d
	cv3 raySphereSpace = rayO;
	float a = rayD.x * rayD.x + rayD.y * rayD.y + rayD.z * rayD.z;
	float b = 2.0f * rayD.x * raySphereSpace.x + 2.0f * rayD.y * raySphereSpace.y + 2.0f * rayD.z * raySphereSpace.z;
	float c = raySphereSpace.x * raySphereSpace.x + raySphereSpace.y * raySphereSpace.y + raySphereSpace.z * raySphereSpace.z - sphereR * sphereR;

	float d1, d2;
	if (!cf_quadratic(a, b, c, &d1, &d2))
	{
		return rayMax;
	}

	// Get our closest point
	float d = d1 < d2 ? d1 : d2;
	if (d > rayMin && d < rayMax)
	{
		return d;
	}

	return rayMax;
}

static float triangle_ray_intersection(cv3 rayO, cv3 rayD, float rayMin, float rayMax, cv3 A, cv3 B, cv3 C, float* out_u, float* out_v, float* out_w)
{
	// Source: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
	cv3 e1 = cv3_sub(B, A);
	cv3 e2 = cv3_sub(C, A);
	cv3 p = cv3_cross(rayD, e2);
	float d = cv3_dot(e1, p);
	if (d < FLT_EPSILON)
	{
		return rayMax;
	}

	float invD = cf_rcp(d);
	cv3 tv = cv3_sub(rayO, A);
	float v = cv3_dot(tv, p) * invD;
	if (v < 0.0f || v > 1.0f)
	{
		return rayMax;
	}

	cv3 q = cv3_cross(tv, e1);
	float w = cv3_dot(rayD, q) * invD;
	if (w < 0.0f || v + w > 1.0f)
	{
		return rayMax;
	}

	float t = cv3_dot(e2, q) * invD;
	if (t > rayMin && t < rayMax)
	{
		*out_u = 1.0f - v - w;
		*out_v = v;
		*out_w = w;
		return t;
	}

	return rayMax;
}

static cv3 sphere_random(random_seed_t* seed)
{
	cv3 p;
	do
	{
		p = cv3_mulf((cv3) { random01f(seed)-0.5f, random01f(seed)-0.5f, random01f(seed)-0.5f }, 2.0f);
	} while (cv3_dot(p, p) <= 1.0f);

	return p;
}

static cv3 hemisphere_surface_random_uniform(float r1, float r2, float* pdf)
{
	float sinTheta = sqrtf(1.0f - r1 * r1); 
	float phi = cran_tao * r2;
	float x = sinTheta * cosf(phi);
	float y = sinTheta * sinf(phi);

	*pdf = cf_rcp(cran_tao);
	return (cv3) { x, y, r1 };
}

static cv3 hemisphere_surface_random_lambert(float r1, float r2, float* pdf)
{
	float theta = acosf(1.0f - 2.0f*r1) * 0.5f;
	float cosTheta = cosf(theta);
	float sinTheta = sinf(theta);
	float phi = cran_tao * r2;

	*pdf = cosTheta * cf_rcp(cran_pi);
	return (cv3) { sinTheta*cosf(phi), sinTheta*sinf(phi), cosTheta };
}

static cv3 hemisphere_surface_random_ggx_h(float r1, float r2, float a)
{
	float cosTheta = sqrtf((1.0f-r1)*cf_rcp(r1*(a*a-1.0f)+1.0f));
	float sinTheta = sqrtf(1.0f - cosTheta*cosTheta);
	float phi = cran_tao * r2;
	return (cv3) { sinTheta*cosf(phi), sinTheta*sinf(phi), cosTheta };
}

static cv3 box_random(random_seed_t* seed)
{
	return cv3_mulf((cv3) { random01f(seed)-0.5f, random01f(seed)-0.5f, random01f(seed)-0.5f }, 2.0f);
}

static cv4 sample_rgb_u8(cv2 uv, uint8_t* cran_restrict image, uint32_t width, uint32_t height, uint32_t offsetX, uint32_t offsetY)
{
	uv.y = cf_frac(uv.y);
	uv.x = cf_frac(uv.x);

	uv.y = uv.y < 0.0f ? 1.0f + uv.y : uv.y;
	uv.x = uv.x < 0.0f ? 1.0f + uv.x : uv.x;

	float readY = uv.y * (float)height;
	float readX = uv.x * (float)width;

	uint32_t y = (uint32_t)floorf(readY) + offsetY;
	y = y >= height ? height - 1 : y;
	uint32_t x = (uint32_t)floorf(readX) + offsetX;
	x = x >= width ? width - 1 : x;
	uint32_t readIndex = (y * width + x) * 3;

	cv4 color;
	color.x = (float)image[readIndex + 0] / 255.0f;
	color.y = (float)image[readIndex + 1] / 255.0f;
	color.z = (float)image[readIndex + 2] / 255.0f;
	color.w = 0.0f;

	return color;
}

static cv4 sample_rg_u8(cv2 uv, uint8_t* cran_restrict image, uint32_t width, uint32_t height, uint32_t offsetX, uint32_t offsetY)
{
	uv.y = cf_frac(uv.y);
	uv.x = cf_frac(uv.x);

	uv.y = uv.y < 0.0f ? 1.0f + uv.y : uv.y;
	uv.x = uv.x < 0.0f ? 1.0f + uv.x : uv.x;

	float readY = uv.y * (float)height;
	float readX = uv.x * (float)width;

	uint32_t y = (uint32_t)floorf(readY) + offsetY;
	y = y >= height ? height - 1 : y;
	uint32_t x = (uint32_t)floorf(readX) + offsetX;
	x = x >= width ? width - 1 : x;
	uint32_t readIndex = (y * width + x) * 2;

	cv4 color;
	color.x = (float)image[readIndex + 0] / 255.0f;
	color.y = (float)image[readIndex + 1] / 255.0f;
	color.z = 0.0f;
	color.w = 0.0f;

	return color;
}

static cv4 sample_r_u8(cv2 uv, uint8_t* cran_restrict image, uint32_t width, uint32_t height, uint32_t offsetX, uint32_t offsetY)
{
	uv.y = cf_frac(uv.y);
	uv.x = cf_frac(uv.x);

	uv.y = uv.y < 0.0f ? 1.0f + uv.y : uv.y;
	uv.x = uv.x < 0.0f ? 1.0f + uv.x : uv.x;

	float readY = uv.y * (float)height;
	float readX = uv.x * (float)width;

	uint32_t y = (uint32_t)floorf(readY) + offsetY;
	y = y >= height ? height - 1 : y;
	uint32_t x = (uint32_t)floorf(readX) + offsetX;
	x = x >= width ? width - 1 : x;
	uint32_t readIndex = y * width + x;

	return (cv4) { (float)image[readIndex] / 255.0f, 0.0f, 0.0f, 0.0f };
}

static cv4 sample_rgba_u8(cv2 uv, uint8_t* cran_restrict image, uint32_t width, uint32_t height, uint32_t offsetX, uint32_t offsetY)
{
	uv.y = cf_frac(uv.y);
	uv.x = cf_frac(uv.x);

	uv.y = uv.y < 0.0f ? 1.0f + uv.y : uv.y;
	uv.x = uv.x < 0.0f ? 1.0f + uv.x : uv.x;

	float readY = uv.y * (float)height;
	float readX = uv.x * (float)width;

	uint32_t y = (uint32_t)floorf(readY) + offsetY;
	y = y >= height ? height - 1 : y;
	uint32_t x = (uint32_t)floorf(readX) + offsetX;
	x = x >= width ? width - 1 : x;
	uint32_t readIndex = (y * width + x) * 4;

	cv4 color;
	color.x = (float)image[readIndex + 0] / 255.0f;
	color.y = (float)image[readIndex + 1] / 255.0f;
	color.z = (float)image[readIndex + 2] / 255.0f;
	color.w = (float)image[readIndex + 3] / 255.0f;

	return color;
}

static cv4 sample_rgba_f32(cv2 uv, uint8_t* cran_restrict image, uint32_t width, uint32_t height, uint32_t offsetX, uint32_t offsetY)
{
	uv.y = cf_frac(uv.y);
	uv.x = cf_frac(uv.x);

	uv.y = uv.y < 0.0f ? 1.0f + uv.y : uv.y;
	uv.x = uv.x < 0.0f ? 1.0f + uv.x : uv.x;

	float readY = uv.y * (float)height;
	float readX = uv.x * (float)width;

	uint32_t y = (uint32_t)floorf(readY) + offsetY;
	y = y >= height ? height - 1 : y;
	uint32_t x = (uint32_t)floorf(readX) + offsetX;
	x = x >= width ? width - 1 : x;
	uint32_t readIndex = (y * width + x) * 4;

	cv4 color;
	color.x = ((float*)image)[readIndex + 0];
	color.y = ((float*)image)[readIndex + 1];
	color.z = ((float*)image)[readIndex + 2];
	color.w = ((float*)image)[readIndex + 3];

	return color;
}

static cv4 sample_rgb_f32(cv2 uv, uint8_t* cran_restrict image, uint32_t width, uint32_t height, uint32_t offsetX, uint32_t offsetY)
{
	uv.y = cf_frac(uv.y);
	uv.x = cf_frac(uv.x);

	uv.y = uv.y < 0.0f ? 1.0f + uv.y : uv.y;
	uv.x = uv.x < 0.0f ? 1.0f + uv.x : uv.x;

	float readY = uv.y * (float)height;
	float readX = uv.x * (float)width;

	uint32_t y = (uint32_t)floorf(readY) + offsetY;
	y = y >= height ? height - 1 : y;
	uint32_t x = (uint32_t)floorf(readX) + offsetX;
	x = x >= width ? width - 1 : x;
	uint32_t readIndex = (y * width + x) * 3;

	cv4 color;
	color.x = ((float*)image)[readIndex + 0];
	color.y = ((float*)image)[readIndex + 1];
	color.z = ((float*)image)[readIndex + 2];
	color.w = 0.0f;

	return color;
}

typedef enum
{
	texture_r_u8,
	texture_rg_u8,
	texture_rgb_u8,
	texture_rgb_f32,
	texture_rgba_f32,
	texture_rgba_u8
} texture_format_e;

typedef struct
{
	// 0 is an invalid texture. [1->max_texture_count] is our valid range
	uint32_t id;
} texture_id_t;

typedef enum
{
	sample_nearest,
	sample_bilinear,
	sample_bump,
} sampling_type_e;

typedef struct
{
	sampling_type_e type;
} sampler_t;

typedef struct
{
	uint8_t* cran_restrict data;
	int32_t width;
	int32_t height;
	int32_t stride;
	texture_format_e format;
} texture_t;

// TODO: Improve our texture cache system,
// We likely want a set amount of memory and page out textures as needed
#define max_texture_count 100
typedef struct
{
	uint32_t hashes[max_texture_count];
	texture_t textures[max_texture_count];
	uint32_t nextTexture;
} texture_store_t;

texture_id_t texture_request(texture_store_t* store, char const* cran_restrict path)
{
	cran_assert(store->nextTexture != max_texture_count);

	uint32_t textureHash = hash(path);

	for (uint32_t i = 0; i < store->nextTexture; i++)
	{
		if (store->hashes[i] == textureHash)
		{
			return (texture_id_t) { i + 1 };
		}
	}

	store->hashes[store->nextTexture] = textureHash;
	texture_t* texture = &store->textures[store->nextTexture];

	texture->data = stbi_load(path, &texture->width, &texture->height, &texture->stride, 0);

	texture_format_e formats[] =
	{
		[1] = texture_r_u8,
		[2] = texture_rg_u8,
		[3] = texture_rgb_u8,
		[4] = texture_rgba_u8
	};
	texture->format = formats[texture->stride];

	return (texture_id_t) { ++store->nextTexture };
}

texture_id_t texture_request_f32(texture_store_t* store, char const* cran_restrict path)
{
	cran_assert(store->nextTexture != max_texture_count);

	uint32_t textureHash = hash(path);

	for (uint32_t i = 0; i < store->nextTexture; i++)
	{
		if (store->hashes[i] == textureHash)
		{
			return (texture_id_t) { i + 1 };
		}
	}

	store->hashes[store->nextTexture] = textureHash;
	texture_t* texture = &store->textures[store->nextTexture];

	texture->data = (uint8_t* cran_restrict)stbi_loadf(path, &texture->width, &texture->height, &texture->stride, 0);

	cran_assert(texture->stride == 3);
	texture_format_e formats[] =
	{
		[3] = texture_rgb_f32,
		[4] = texture_rgba_f32
	};
	texture->format = formats[texture->stride];

	return (texture_id_t) { ++store->nextTexture };
}

typedef cv4(sampler_func_t)(cv2, uint8_t* cran_restrict image, uint32_t width, uint32_t height, uint32_t offsetX, uint32_t offsetY);
cv4 sampler_sample(texture_store_t const* store, sampler_t sampler, texture_id_t textureid, cv2 uv)
{
	if (textureid.id == 0)
	{
		if (sampler.type == sample_bump)
		{
			return (cv4) { 0 };
		}
		else
		{
			// No texture is fully white because it makes shaders easier to write.
			return (cv4) { 1.0f, 1.0f, 1.0f, 1.0f };
		}
	}

	cran_assert(textureid.id <= store->nextTexture);
	sampler_func_t* samplers[] =
	{
		[texture_r_u8] = &sample_r_u8,
		[texture_rg_u8] = &sample_rg_u8,
		[texture_rgb_u8] = &sample_rgb_u8,
		[texture_rgba_u8] = &sample_rgba_u8,
		[texture_rgb_f32] = &sample_rgb_f32,
		[texture_rgba_f32] = &sample_rgba_f32
	};

	texture_t const* texture = &store->textures[textureid.id - 1];
	if (sampler.type == sample_nearest)
	{
		return samplers[texture->format](uv, texture->data, texture->width, texture->height, 0, 0);
	}
	else if (sampler.type == sample_bilinear)
	{
		cv4 s00 = samplers[texture->format](uv, texture->data, texture->width, texture->height, 0, 0);
		cv4 s01 = samplers[texture->format](uv, texture->data, texture->width, texture->height, 0, 1);
		cv4 s10 = samplers[texture->format](uv, texture->data, texture->width, texture->height, 1, 0);
		cv4 s11 = samplers[texture->format](uv, texture->data, texture->width, texture->height, 1, 1);

		float wf = cf_frac((float)texture->width * uv.x);
		float hf = cf_frac((float)texture->height * uv.y);

		wf = wf < 0.0f ? 1.0f + wf : wf;
		hf = hf < 0.0f ? 1.0f + hf : hf;
		return (cv4)
		{
			cf_bilinear(s00.x, s01.x, s10.x, s11.x, wf, hf),
			cf_bilinear(s00.y, s01.y, s10.y, s11.y, wf, hf),
			cf_bilinear(s00.z, s01.z, s10.z, s11.z, wf, hf),
			cf_bilinear(s00.w, s01.w, s10.w, s11.w, wf, hf)
		};
	}
	else if (sampler.type == sample_bump)
	{
		float s00 = samplers[texture->format](uv, texture->data, texture->width, texture->height, 0, 0).x;
		float s10 = samplers[texture->format](uv, texture->data, texture->width, texture->height, 1, 0).x;
		float s01 = samplers[texture->format](uv, texture->data, texture->width, texture->height, 0, 1).x;

		return (cv4) { s10 - s00, s01 - s00, 0.0f, 0.0f };
	}

	return (cv4) { 0 };
}

cv4 sampler_sample_3D(texture_store_t const* store, sampler_t sampler, texture_id_t texture, cv3 direction)
{
	float azimuth, theta;
	cv3_to_spherical(direction, &azimuth, &theta);
	return sampler_sample(store, sampler, texture, (cv2) { azimuth * cran_rtao, theta * cran_rpi });
}

typedef struct
{
	float* cran_restrict cdf2d;
	float* cran_restrict luminance2d;

	float* cran_restrict cdf1d;
	float* cran_restrict sum1d;
	float sumTotal;

	int32_t width;
	int32_t height;
} sphere_importance_t;

static cv3 importance_sample_hdr(sphere_importance_t importanceData, float* cran_restrict outBias, random_seed_t* seed)
{
	float ycdf = random01f(seed);
	int y = 0;
	for (; y < importanceData.height; y++)
	{
		if (importanceData.cdf1d[y] >= ycdf)
		{
			break;
		}
	}

	float xcdf = random01f(seed);
	int x = 0;
	for (; x < importanceData.width; x++)
	{
		if (importanceData.cdf2d[y*importanceData.width + x] >= xcdf)
		{
			break;
		}
	}

	float biasy = importanceData.sum1d[y] * cf_rcp(importanceData.sumTotal);
	float biasx = importanceData.luminance2d[y * importanceData.width + x] * cf_rcp(importanceData.sum1d[y]);
	float bias = biasy * biasx;

	cv3 direction = cv3_from_spherical(((float)y / (float)importanceData.height)*cran_pi, ((float)x / (float)importanceData.width)*cran_tao, 1.0f);

	*outBias = bias;
	return direction;
}

static float light_attenuation(cv3 l, cv3 r)
{
	return cf_rcp(1.0f + cv3_sqrlength(cv3_sub(l, r)));
}

// TODO: Refine our scene description
typedef struct
{
	union
	{
		struct
		{
			uint16_t left;
			uint16_t right;
		} jumps;

		uint32_t index;
	} indices;
} bvh_jump_t;

typedef struct
{
	caabb* bounds;
	bvh_jump_t* jumps;
	uint32_t count;
	uint32_t leafCount;
} bvh_t;

typedef enum
{
	material_lambert,
	material_mirror,
	material_directional,
	material_microfacet,
	material_count
} material_type_e;

typedef struct
{
	cv3 albedoTint;
	sampler_t albedoSampler;
	sampler_t bumpSampler;
	texture_id_t albedoTexture;
	texture_id_t bumpTexture;
} material_lambert_t;

typedef struct
{
	cv3 albedoTint;
	cv3 specularTint;
	sampler_t albedoSampler;
	sampler_t bumpSampler;
	sampler_t glossSampler;
	sampler_t maskSampler;
	texture_id_t albedoTexture;
	texture_id_t bumpTexture;
	texture_id_t glossTexture;
	texture_id_t maskTexture;
	float refractiveIndex;
	float gloss;
} material_microfacet_t;

typedef struct
{
	cv3 color;
} material_mirror_t;

typedef struct
{
	uint16_t dataIndex;
	material_type_e typeIndex;
} material_index_t;
static_assert(material_count < 255, "Only 255 renderable types are supported.");

typedef struct
{
	uint16_t dataIndex;
} renderable_index_t;

typedef struct
{
	cv3 pos;
	uint32_t renderableIndex;
} instance_t;

typedef struct
{
	cranl_mesh_t data;
	material_index_t* materialIndices;
} mesh_source_t;

typedef struct
{
	mesh_source_t* sources;
	uint32_t count;
} mesh_store_t;

typedef struct
{
	uint32_t meshSourceIndex;
	bvh_t bvh;
} mesh_t;

typedef struct
{
	void* cran_restrict materials[material_count];
	mesh_t* cran_restrict renderables;

	struct
	{
		instance_t* data;
		uint32_t count;
	} instances;

	struct
	{
		sampler_t sampler;
		sphere_importance_t importance;
		texture_id_t texture;
	} environment;

	bvh_t bvh;

	mesh_store_t meshStore;
	texture_store_t textureStore;
} ray_scene_t;

typedef struct
{
	caabb bound;
	uint32_t index;
} index_aabb_pair_t;

typedef struct
{
	cv3 surface;
	cv3 normal;
	cv3 tangent;
	cv3 bitangent;
	cv3 viewDir;
	cv2 uv;
	uint64_t triangleId;
} shader_inputs_t;

typedef struct
{
	cv3 light;
	cv3 shadedSurface; // Allows the shader to modify the hit surface
} shader_outputs_t;

typedef shader_outputs_t(material_shader_t)(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs);

static material_shader_t shader_lambert;
static material_shader_t shader_mirror;
static material_shader_t shader_directional;
static material_shader_t shader_microfacet;
material_shader_t* shaders[material_count] =
{
	shader_lambert,
	shader_mirror,
	shader_directional,
	shader_microfacet,
};

static int index_aabb_sort_min_x(const void* cran_restrict l, const void* cran_restrict r)
{
	const index_aabb_pair_t* cran_restrict left = (const index_aabb_pair_t*)l;
	const index_aabb_pair_t* cran_restrict right = (const index_aabb_pair_t*)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)copysignf(1.0f, caabb_centroid(left->bound, caabb_x) - caabb_centroid(right->bound, caabb_x));
}

static int index_aabb_sort_min_y(const void* cran_restrict l, const void* cran_restrict r)
{
	const index_aabb_pair_t* cran_restrict left = (const index_aabb_pair_t* cran_restrict)l;
	const index_aabb_pair_t* cran_restrict right = (const index_aabb_pair_t* cran_restrict)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)copysignf(1.0f, caabb_centroid(left->bound, caabb_y) - caabb_centroid(right->bound, caabb_y));
}

static int index_aabb_sort_min_z(const void* cran_restrict l, const void* cran_restrict r)
{
	const index_aabb_pair_t* left = (const index_aabb_pair_t* cran_restrict)l;
	const index_aabb_pair_t* right = (const index_aabb_pair_t* cran_restrict)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)copysignf(1.0f, caabb_centroid(left->bound, caabb_z) - caabb_centroid(right->bound, caabb_z));
}

static bvh_t build_bvh(render_context_t* context, index_aabb_pair_t* leafs, uint32_t leafCount)
{
	cran_assert(leafCount > 0);

	int(*sortFuncs[3])(const void* cran_restrict l, const void* cran_restrict r) = { index_aabb_sort_min_x, index_aabb_sort_min_y, index_aabb_sort_min_z };

	typedef struct
	{
		index_aabb_pair_t* start;
		uint32_t count;
		uint16_t* parentIndex;
	} bvh_workgroup_t;

	// Simple ring buffer
	uint32_t workgroupSize = 1000000;
	bvh_workgroup_t* bvhWorkgroup = (bvh_workgroup_t*)crana_stack_alloc(&context->scratchStack, sizeof(bvh_workgroup_t) * workgroupSize);
	bvh_workgroup_t* leafWorkgroup = (bvh_workgroup_t*)crana_stack_alloc(&context->scratchStack, sizeof(bvh_workgroup_t) * leafCount);
	uint32_t workgroupQueueEnd = 0;
	uint32_t leafWorkgroupEnd = 0;

	if (leafCount == 1)
	{
		leafWorkgroup[0].start = leafs;
		leafWorkgroup[0].count = leafCount;
		leafWorkgroup[0].parentIndex = NULL;
		leafWorkgroupEnd = 1;
	}
	else
	{
		bvhWorkgroup[0].start = leafs;
		bvhWorkgroup[0].count = leafCount;
		bvhWorkgroup[0].parentIndex = NULL;
		workgroupQueueEnd = 1;
	}

	// Add to this list first as a workspace, allows us to allocate one at a time
	// Once we're done, we can split the data into a more memory efficient format
	typedef struct
	{
		caabb bound;
		bvh_jump_t jump;
	} bvh_pair_t;

	// Lock our stack, we're free to advance our pointer as much as we please
	bvh_pair_t* buildingBVHStart = (bvh_pair_t*)crana_stack_lock(&context->scratchStack);
	bvh_pair_t* buildingBVHIter = buildingBVHStart;
	for (uint32_t workgroupIter = 0; workgroupIter != workgroupQueueEnd; workgroupIter = (workgroupIter + 1) % workgroupSize) // TODO: constant for workgroup size
	{
		index_aabb_pair_t* start = bvhWorkgroup[workgroupIter].start;
		uint32_t count = bvhWorkgroup[workgroupIter].count;

		if (bvhWorkgroup[workgroupIter].parentIndex != NULL)
		{
			*(bvhWorkgroup[workgroupIter].parentIndex) = (uint16_t)(buildingBVHIter - buildingBVHStart);
			assert((buildingBVHIter - buildingBVHStart) < UINT16_MAX);
		}

		caabb bounds = start[0].bound;
		for (uint32_t i = 1; i < count; i++)
		{
			bounds.min = cv3_min(start[i].bound.min, bounds.min);
			bounds.max = cv3_max(start[i].bound.max, bounds.max);
		}

		cran_assert(count > 1);
		buildingBVHIter->bound = bounds;
		buildingBVHIter->jump = (bvh_jump_t)
		{
			.indices.index = start[0].index,
		};

		// TODO: Since we're doing all the iteration work in the sort, maybe we could also do the partitioning in the sort?
		cv2 axisSpan[3];
		for (uint32_t axis = 0; axis < 3; axis++)
		{
			for (uint32_t i = 0; i < count; i++)
			{
				axisSpan[axis].x = fminf(axisSpan[axis].x, (&start[i].bound.max.x)[axis] - (&start[i].bound.min.x)[axis] * 0.5f);
				axisSpan[axis].y = fmaxf(axisSpan[axis].y, (&start[i].bound.max.x)[axis] - (&start[i].bound.min.x)[axis] * 0.5f);
			}
		}

		uint32_t axis;
		if (axisSpan[0].y - axisSpan[0].x > axisSpan[1].y - axisSpan[1].x)
		{
			axis = 0;
		}
		else if (axisSpan[1].y - axisSpan[1].x > axisSpan[2].y - axisSpan[2].x)
		{
			axis = 1;
		}
		else
		{
			axis = 2;
		}

		qsort(start, count, sizeof(index_aabb_pair_t), sortFuncs[axis]);

#define cran_bucket_count 12
		caabb bucketBounds[cran_bucket_count] = { 0 };
		uint32_t bucketCount[cran_bucket_count] = { 0 };

		for (uint32_t i = 0; i < count; i++)
		{
			float p = (caabb_centroid(start[i].bound, axis) - (&bounds.min.x)[axis])/caabb_side(bounds,axis);
			uint32_t bucketIndex = (uint32_t)(p*cran_bucket_count);
			bucketIndex = bucketIndex == cran_bucket_count ? bucketIndex - 1 : bucketIndex;
			if (bucketCount[bucketIndex] == 0)
			{
				bucketBounds[bucketIndex] = start[i].bound;
			}
			else
			{
				bucketBounds[bucketIndex] = caabb_merge(bucketBounds[bucketIndex], start[i].bound);
			}
			
			bucketCount[bucketIndex]++;
		}

		float sah[cran_bucket_count-1] = { 0 };
		for (uint32_t i = 0; i < cran_bucket_count-1; i++)
		{
			caabb left = bucketBounds[0];
			uint32_t leftCount = 0;
			caabb right = bucketBounds[cran_bucket_count - 1];
			uint32_t rightCount = 0;

			for (uint32_t b = 0; b <= i; b++)
			{
				// Don't merge empty buckets
				if (bucketCount[b] > 0)
				{
					left = caabb_merge(left, bucketBounds[b]);
					leftCount += bucketCount[b];
				}
			}

			for (uint32_t b = i+1; b < cran_bucket_count; b++)
			{
				if (bucketCount[b] > 0)
				{
					right = caabb_merge(right, bucketBounds[b]);
					rightCount += bucketCount[b];
				}
			}

			// SAH
			const float traversalRelativeCost = 0.25f;
			sah[i] = traversalRelativeCost+((float)leftCount * caabb_surface_area(left) + (float)rightCount * caabb_surface_area(right)) / caabb_surface_area(bounds);
		}

		// Find our lowest cost bucket
		float min = sah[0];
		uint32_t minIndex = 0;
		for (uint32_t i = 1; i < cran_bucket_count-1; i++)
		{
			if (sah[i] < min)
			{
				minIndex = i;
				min = sah[i];
			}
		}

		float bucketSize = caabb_side(bounds, axis) / (float)cran_bucket_count;
		float split = (float)minIndex * bucketSize + bucketSize + (&bounds.min.x)[axis];

		uint32_t centerIndex = count/2;
		for (uint32_t i = 0; i < count; i++)
		{
			if (caabb_centroid(start[i].bound, axis) > split)
			{
				centerIndex = i;
				break;
			}
		}

		// If we're trying to split it into nothing, just split it in half
		if (centerIndex == 0)
		{
			centerIndex = count / 2;
		}

		bool isLeaf = centerIndex == 1;
		if (!isLeaf)
		{
			bvhWorkgroup[workgroupQueueEnd].start = start;
			bvhWorkgroup[workgroupQueueEnd].count = centerIndex;
			cran_assert(bvhWorkgroup[workgroupQueueEnd].count > 0);
			bvhWorkgroup[workgroupQueueEnd].parentIndex = &buildingBVHIter->jump.indices.jumps.left;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % workgroupSize;
			cran_assert(workgroupQueueEnd != workgroupIter);
		}
		else
		{
			leafWorkgroup[leafWorkgroupEnd].start = start;
			leafWorkgroup[leafWorkgroupEnd].count = centerIndex;
			cran_assert(leafWorkgroup[leafWorkgroupEnd].count > 0);
			leafWorkgroup[leafWorkgroupEnd].parentIndex = &buildingBVHIter->jump.indices.jumps.left;
			leafWorkgroupEnd++;
			cran_assert(leafWorkgroupEnd <= leafCount);
		}

		isLeaf = (count - centerIndex == 1);
		if (!isLeaf)
		{
			bvhWorkgroup[workgroupQueueEnd].start = start + centerIndex;
			bvhWorkgroup[workgroupQueueEnd].count = count - centerIndex;
			cran_assert(bvhWorkgroup[workgroupQueueEnd].count > 0);

			bvhWorkgroup[workgroupQueueEnd].parentIndex = &buildingBVHIter->jump.indices.jumps.right;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % workgroupSize;
			cran_assert(workgroupQueueEnd != workgroupIter);
		}
		else
		{
			leafWorkgroup[leafWorkgroupEnd].start = start + centerIndex;
			leafWorkgroup[leafWorkgroupEnd].count = count - centerIndex;
			cran_assert(leafWorkgroup[leafWorkgroupEnd].count > 0);
			leafWorkgroup[leafWorkgroupEnd].parentIndex = &buildingBVHIter->jump.indices.jumps.right;
			leafWorkgroupEnd++;
			cran_assert(leafWorkgroupEnd <= leafCount);
		}
		buildingBVHIter++;
	}

	// Add all the leafs to the end of it
	cran_assert(leafWorkgroupEnd == leafCount);
	for (uint32_t i = 0; i < leafCount; i++)
	{
		index_aabb_pair_t* start = leafWorkgroup[i].start;
		uint32_t count = leafWorkgroup[i].count;

		if (leafWorkgroup[i].parentIndex != NULL)
		{
			*(leafWorkgroup[i].parentIndex) = (uint16_t)(buildingBVHIter - buildingBVHStart);
			assert((buildingBVHIter - buildingBVHStart) < UINT16_MAX);
		}

		caabb bounds = start[0].bound;
		cran_assert(count == 1);
		buildingBVHIter->bound = bounds;
		buildingBVHIter->jump = (bvh_jump_t)
		{
			.indices.index = start[0].index,
		};

		buildingBVHIter++;
	}

	uint32_t bvhSize = (uint32_t)(buildingBVHIter - buildingBVHStart);
	bvh_t builtBVH =
	{
		.bounds = crana_stack_alloc(&context->stack, sizeof(caabb) * bvhSize),
		.jumps = crana_stack_alloc(&context->stack, sizeof(bvh_jump_t) * bvhSize),
		.count = bvhSize,
		.leafCount = leafCount
	};

	for (uint32_t i = 0; i < bvhSize; i++)
	{
		builtBVH.bounds[i] = buildingBVHStart[i].bound;
		builtBVH.jumps[i] = buildingBVHStart[i].jump;
	}

	// Don't bother commiting our lock, we don't need this anymore
	crana_stack_revert(&context->scratchStack);
	crana_stack_free(&context->scratchStack, bvhWorkgroup);
	crana_stack_free(&context->scratchStack, leafWorkgroup);

	cran_stat(context->renderStats.bvhNodeCount = bvhSize);
	return builtBVH;
}

// Allocates candidates from bottom stack
// Uses top stack for working memory
// TODO: Maybe 2 seperate stacks would be better than using a single stack with 2 apis
static uint32_t traverse_bvh(render_context_t* context, bvh_t const* bvh, cv3 rayO, cv3 rayD, float rayMin, float rayMax, uint32_t** candidates)
{
	cranpr_begin("bvh", "traverse");

	if (bvh->count == 0)
	{
		*candidates = crana_stack_alloc(&context->stack, 1);
		return 0;
	}

	*candidates = crana_stack_lock(&context->stack); // Allocate nothing, but we're going to be growing it
	uint32_t* candidateIter = *candidates;

	cran_stat(uint64_t traversalStartTime = cranpl_timestamp_micro());

	uint32_t* testQueueIter = crana_stack_lock(&context->scratchStack);
	uint32_t* testQueueEnd = testQueueIter+1;

	*testQueueIter = 0;
	while (testQueueEnd > testQueueIter)
	{
		cv3l boundMins = { 0 };
		cv3l boundMaxs = { 0 };

		uint32_t activeLaneCount = min((uint32_t)(testQueueEnd - testQueueIter), cran_lane_count);
		for (uint32_t i = 0; i < activeLaneCount; i++)
		{
			uint32_t nodeIndex = testQueueIter[i];
			cv3l_set(&boundMins, bvh->bounds[nodeIndex].min, i);
			cv3l_set(&boundMaxs, bvh->bounds[nodeIndex].max, i);
		}

		uint32_t intersections = caabb_does_ray_intersect_lanes(rayO, rayD, rayMin, rayMax, boundMins, boundMaxs);
		uint32_t activeLaneCountMask = (1 << activeLaneCount) - 1;

		if (intersections > 0)
		{
			intersections = intersections & activeLaneCountMask;

#define _MM_SHUFFLE_EPI8(i3,i2,i1,i0) _mm_set_epi8(i3*4+3,i3*4+2,i3*4+1,i3*4,i2*4+3,i2*4+2,i2*4+1,i2*4,i1*4+3,i1*4+2,i1*4+1,i1*4,i0*4+3,i0*4+2,i0*4+1,i0*4)
			__m128i shuffles[16] = 
			{
				_MM_SHUFFLE_EPI8(0,0,0,0), _MM_SHUFFLE_EPI8(0,0,0,0), _MM_SHUFFLE_EPI8(0,0,0,1), _MM_SHUFFLE_EPI8(0,0,1,0), // 0000, 0001, 0010, 0011
				_MM_SHUFFLE_EPI8(0,0,0,2), _MM_SHUFFLE_EPI8(0,0,2,0), _MM_SHUFFLE_EPI8(0,0,2,1), _MM_SHUFFLE_EPI8(0,2,1,0), // 0100, 0101, 0110, 0111
				_MM_SHUFFLE_EPI8(0,0,0,3), _MM_SHUFFLE_EPI8(0,0,3,0), _MM_SHUFFLE_EPI8(0,0,3,1), _MM_SHUFFLE_EPI8(0,3,1,0), // 1000, 1001, 1010, 1011
				_MM_SHUFFLE_EPI8(0,0,3,2), _MM_SHUFFLE_EPI8(0,3,2,0), _MM_SHUFFLE_EPI8(0,3,2,1), _MM_SHUFFLE_EPI8(3,2,1,0)  // 1100, 1101, 1110, 1111
			};

			__m128i queueIndices = _mm_load_si128((__m128i*)testQueueIter);

			uint32_t leafLine = bvh->count - bvh->leafCount;
			uint32_t childIndexMask;
			uint32_t parentIndexMask;
			{
				__m128i isParent = _mm_cmplt_epi32(queueIndices, _mm_set_epi32(leafLine, leafLine, leafLine, leafLine));
				parentIndexMask = _mm_movemask_ps(_mm_castsi128_ps(isParent));
				childIndexMask = ~parentIndexMask & 0x0F;

				parentIndexMask = parentIndexMask & intersections;
				childIndexMask = childIndexMask & intersections;
			}

			uint32_t leafCount = __popcnt(childIndexMask);
			uint32_t parentCount = __popcnt(parentIndexMask);
			__m128i childIndices = _mm_shuffle_epi8(queueIndices, shuffles[childIndexMask]);
			__m128i parentIndices = _mm_shuffle_epi8(queueIndices, shuffles[parentIndexMask]);

			cran_stat(context->renderStats.bvhHitCount += leafCount+parentCount); 
			cran_stat(context->renderStats.bvhLeafHitCount += leafCount);
			cran_stat(context->renderStats.bvhMissCount += activeLaneCount-(leafCount+parentCount));

			union
			{
				uint32_t i[4];
				__m128i v;
			} indexUnion;

			indexUnion.v = childIndices;
			for (uint32_t i = 0; i < leafCount; i++)
			{
				uint32_t nodeIndex = indexUnion.i[i];
				candidateIter[i] = bvh->jumps[nodeIndex].indices.index;
			}
			candidateIter+=leafCount;

			indexUnion.v = parentIndices;
			for (uint32_t i = 0; i < parentCount; i++)
			{
				uint32_t nodeIndex = indexUnion.i[i];
				cran_assert(nodeIndex < bvh->count);
				testQueueEnd[i*2] = bvh->jumps[nodeIndex].indices.jumps.left;
				testQueueEnd[i*2 + 1] = bvh->jumps[nodeIndex].indices.jumps.right;
			}
			testQueueEnd += parentCount * 2;
		}
		else
		{
			cran_stat(context->renderStats.bvhMissCount += activeLaneCount);
		}


		testQueueIter += activeLaneCount;
	}

	crana_stack_revert(&context->scratchStack);
	crana_stack_commit(&context->stack, candidateIter);

	cran_stat(context->renderStats.bvhTraversalTime += cranpl_timestamp_micro() - traversalStartTime);

	cranpr_end("bvh", "traverse");
	return (uint32_t)(candidateIter - *candidates);
}

static void generate_scene(render_context_t* context, ray_scene_t* scene)
{
	cranpr_begin("scene", "generate");
	cran_stat(uint64_t startTime = cranpl_timestamp_micro());

	static mesh_source_t meshSource;
	// Mesh
	mesh_t* meshes;
	{
		// TODO: We likely don't want a stack allocator here
		// clean up would be too tedious, think of a way to encapsulate meshes
		meshSource.data = cranl_obj_load("sponza.obj", cranl_flip_yz | cranl_cm_to_m,
			(cranl_allocator_t)
			{
				.instance = &context->stack,
				.alloc = &crana_stack_alloc,
				.free = &crana_stack_free
			});

		meshes = crana_stack_alloc(&context->stack, sizeof(mesh_t) * meshSource.data.groups.count);
		for (uint32_t groupIndex = 0; groupIndex < meshSource.data.groups.count; groupIndex++)
		{
			meshes[groupIndex].meshSourceIndex = 0;

			uint32_t faceStart = meshSource.data.groups.groupOffsets[groupIndex];
			uint32_t faceEnd = (groupIndex + 1) < meshSource.data.groups.count ? meshSource.data.groups.groupOffsets[groupIndex + 1] : meshSource.data.faces.count;
			if (faceEnd - faceStart == 0)
			{
				meshes[groupIndex] = (mesh_t) { 0 };
				continue;
			}

			index_aabb_pair_t* leafs = crana_stack_alloc(&context->scratchStack, sizeof(index_aabb_pair_t) * (faceEnd - faceStart));
			for (uint32_t i = faceStart, leafIndex = 0; i < faceEnd; i++, leafIndex++)
			{
				leafs[leafIndex].index = i;

				uint32_t vertIndexA = meshSource.data.faces.vertexIndices[i * 3 + 0];
				uint32_t vertIndexB = meshSource.data.faces.vertexIndices[i * 3 + 1];
				uint32_t vertIndexC = meshSource.data.faces.vertexIndices[i * 3 + 2];

				cv3 vertA, vertB, vertC;
				memcpy(&vertA, meshSource.data.vertices.data + vertIndexA * 3, sizeof(cv3));
				memcpy(&vertB, meshSource.data.vertices.data + vertIndexB * 3, sizeof(cv3));
				memcpy(&vertC, meshSource.data.vertices.data + vertIndexC * 3, sizeof(cv3));

				leafs[leafIndex].bound.min = cv3_min(cv3_min(vertA, vertB), vertC);
				leafs[leafIndex].bound.max = cv3_max(cv3_max(vertA, vertB), vertC);

				// If our bounds have no volume, add a surrounding shell
				if (fabsf(leafs[leafIndex].bound.max.x - leafs[leafIndex].bound.min.x) < FLT_EPSILON)
				{
					leafs[leafIndex].bound.max.x += 0.001f;
					leafs[leafIndex].bound.min.x -= 0.001f;
				}

				if (fabsf(leafs[leafIndex].bound.max.y - leafs[leafIndex].bound.min.y) < FLT_EPSILON)
				{
					leafs[leafIndex].bound.max.y += 0.001f;
					leafs[leafIndex].bound.min.y -= 0.001f;
				}

				if (fabsf(leafs[leafIndex].bound.max.z - leafs[leafIndex].bound.min.z) < FLT_EPSILON)
				{
					leafs[leafIndex].bound.max.z += 0.001f;
					leafs[leafIndex].bound.min.z -= 0.001f;
				}
			}

			meshes[groupIndex].bvh = build_bvh(context, leafs, faceEnd-faceStart);
			crana_stack_free(&context->scratchStack, leafs);
		}
	}

	texture_store_t textureStore = { 0 };

	// Environment map
	sampler_t backgroundSampler = (sampler_t) {.type = sample_nearest};
	texture_id_t backgroundTextureId;
	sphere_importance_t backgroundImportance;
	{
		backgroundTextureId = texture_request_f32(&textureStore, "background_4k.hdr");
		texture_t const* texture = &textureStore.textures[backgroundTextureId.id - 1];

		backgroundImportance.cdf2d = (float*)crana_stack_alloc(&context->stack, sizeof(float) * texture->width * texture->height);
		backgroundImportance.luminance2d = (float*)crana_stack_alloc(&context->stack, sizeof(float) * texture->width * texture->height);

		backgroundImportance.cdf1d = (float*)crana_stack_alloc(&context->stack, sizeof(float) * texture->height);
		backgroundImportance.sum1d = (float*)crana_stack_alloc(&context->stack, sizeof(float) * texture->height);

		float ysum = 0.0f;
		for (int y = 0; y < texture->height; y++)
		{
			float xsum = 0.0f;
			for (int x = 0; x < texture->width; x++)
			{
				int index = (y * texture->width) + x;

				float* cran_restrict pixel = &((float*)texture->data)[index * texture->stride];
				float luminance = rgb_to_luminance(pixel[0], pixel[1], pixel[2]);

				xsum += luminance;
				backgroundImportance.luminance2d[index] = luminance;
				backgroundImportance.cdf2d[index] = xsum;
			}

			for (int x = 0; x < texture->width; x++)
			{
				int index = (y * texture->width) + x;
				backgroundImportance.cdf2d[index] = backgroundImportance.cdf2d[index] * cf_rcp(xsum);
			}

			ysum += xsum;
			backgroundImportance.cdf1d[y] = ysum;
			backgroundImportance.sum1d[y] = xsum;
		}

		for (int y = 0; y < texture->height; y++)
		{
			backgroundImportance.cdf1d[y] = backgroundImportance.cdf1d[y]  * cf_rcp(ysum);
		}

		backgroundImportance.sumTotal = ysum;
	}

	// materials
	material_microfacet_t* microfacets;
	static material_mirror_t mirror;
	{
		mirror = (material_mirror_t){ .color = (cv3) {1.0f, 1.0f, 1.0f} };

		cran_assert(meshSource.data.materialLibraries.count == 1); // Assume only one material library for now
		cranl_material_lib_t matLib = cranl_obj_mat_load(meshSource.data.materialLibraries.names[0], (cranl_allocator_t)
			{
				.instance = &context->stack,
				.alloc = &crana_stack_alloc,
				.free = &crana_stack_free
			});

		microfacets = crana_stack_alloc(&context->stack, sizeof(material_microfacet_t) * matLib.count);
		memset(microfacets, 0, sizeof(material_microfacet_t) * matLib.count);
		for (uint32_t i = 0; i < matLib.count; i++)
		{
			microfacets[i].albedoTint = (cv3) { matLib.materials[i].albedo[0], matLib.materials[i].albedo[1], matLib.materials[i].albedo[2] };
			microfacets[i].refractiveIndex = matLib.materials[i].refractiveIndex;
			microfacets[i].specularTint = (cv3) { matLib.materials[i].specular[0], matLib.materials[i].specular[1], matLib.materials[i].specular[2] };
			microfacets[i].gloss = 0.0f;

			microfacets[i].albedoSampler = (sampler_t) {.type = sample_bilinear };
			if (matLib.materials[i].albedoMap != NULL)
			{
				microfacets[i].albedoTexture = texture_request(&textureStore, matLib.materials[i].albedoMap);
			}

			microfacets[i].bumpSampler = (sampler_t) {.type = sample_bump };
			if (matLib.materials[i].bumpMap != NULL)
			{
				microfacets[i].bumpTexture = texture_request(&textureStore, matLib.materials[i].bumpMap);
			}

			microfacets[i].glossSampler = (sampler_t) {.type = sample_bilinear };
			if (matLib.materials[i].glossMap != NULL)
			{
				microfacets[i].glossTexture = texture_request(&textureStore, matLib.materials[i].glossMap);

			}

			microfacets[i].maskSampler = (sampler_t) {.type = sample_nearest };
			if (matLib.materials[i].maskMap != NULL)
			{
				microfacets[i].maskTexture = texture_request(&textureStore, matLib.materials[i].maskMap);
			}
		}

		material_index_t* materialIndices = crana_stack_alloc(&context->stack, sizeof(material_index_t) * meshSource.data.materials.count);
		for (uint32_t i = 0; i < meshSource.data.materials.count; i++)
		{
			uint16_t dataIndex = 0;
			for (; dataIndex < matLib.count; dataIndex++)
			{
				if (strcmp(matLib.materials[dataIndex].name, meshSource.data.materials.materialNames[i]) == 0)
				{
					break;
				}
			}

			materialIndices[i] = (material_index_t) { .dataIndex = dataIndex, .typeIndex = material_microfacet };
		}

		meshSource.materialIndices = materialIndices;
	}

	instance_t* instances = crana_stack_alloc(&context->stack, sizeof(instance_t)*meshSource.data.groups.count);
	for (uint32_t i = 0; i < meshSource.data.groups.count; i++)
	{
		instances[i] = (instance_t)
		{
			.pos = { 0.0f, 0.0f, 0.0f },
			.renderableIndex = i
		};
	}

	// Output our scene
	*scene = (ray_scene_t)
	{
		.instances =
		{
			.data = instances,
			.count = meshSource.data.groups.count
		},
		.renderables = meshes,
		.materials =
		{
			[material_microfacet] = microfacets,
			[material_mirror] = &mirror
		},
		.meshStore =
		{
			.sources = &meshSource,
			.count = 1,
		},
		.textureStore = textureStore,
		.environment =
		{
			.importance = backgroundImportance,
			.texture = backgroundTextureId,
			.sampler = backgroundSampler,
		}
	};

	// BVH
	{
		uint32_t leafCount = scene->instances.count;
		index_aabb_pair_t* leafs = crana_stack_alloc(&context->scratchStack, sizeof(index_aabb_pair_t) * leafCount);
		for (uint32_t i = 0; i < leafCount; i++)
		{
			cv3 pos = scene->instances.data[i].pos;
			uint32_t renderableIndex = scene->instances.data[i].renderableIndex;

			leafs[i].index = i;

			mesh_t* mesh = &scene->renderables[renderableIndex];
			for (uint32_t boundIndex = 0; boundIndex < mesh->bvh.count; boundIndex++)
			{
				leafs[i].bound.min = cv3_min(leafs[i].bound.min, cv3_add(pos, mesh->bvh.bounds[boundIndex].min));
				leafs[i].bound.max = cv3_max(leafs[i].bound.max, cv3_add(pos, mesh->bvh.bounds[boundIndex].max));
			}
		}

		scene->bvh = build_bvh(context, leafs, leafCount);
		crana_stack_free(&context->scratchStack, leafs);
	}

	cran_stat(context->renderStats.sceneGenerationTime = cranpl_timestamp_micro() - startTime);
	cranpr_end("scene", "generate");
}

typedef struct
{
	cv3 light;
	cv3 surface;
	bool hit;
} ray_hit_t;

static ray_hit_t cast_scene(render_context_t* context, ray_scene_t const* scene, cv3 rayO, cv3 rayD, uint64_t sourceTriangleId)
{
	cranpr_begin("scene", "cast");

	context->depth++;
	if (context->depth >= renderConfig.maxDepth)
	{
		context->depth--;
		cranpr_end("scene", "cast");
		return (ray_hit_t) { 0 };
	}

	cran_stat(context->renderStats.rayCount++);

	const float NoRayIntersection = FLT_MAX;

	struct
	{
		float distance;
		cv3 normal;
		cv3 tangent;
		cv3 bitangent;
		cv3 barycentric;
		cv2 uv;
		material_index_t materialIndex;
		uint64_t triangleId;
	} closestHitInfo = { 0 };
	closestHitInfo.distance = NoRayIntersection;

	cran_stat(uint64_t intersectionStartTime = cranpl_timestamp_micro());

	// Candidates
	{
		uint32_t* candidates;
		uint32_t candidateCount = traverse_bvh(context, &scene->bvh, rayO, rayD, 0.0f, NoRayIntersection, &candidates);
		for (uint32_t i = 0; i < candidateCount; i++)
		{
			uint32_t candidateIndex = candidates[i];

			cv3 instancePos = scene->instances.data[candidateIndex].pos;
			uint32_t renderableIndex = scene->instances.data[candidateIndex].renderableIndex;

			cv3 rayInstanceO = cv3_sub(rayO, instancePos);

			float intersectionDistance = 0.0f;

			mesh_t* mesh = &scene->renderables[renderableIndex];
			mesh_source_t* sourceMesh = &scene->meshStore.sources[mesh->meshSourceIndex];
			{
				uint32_t* meshCandidates;
				uint32_t meshCandidateCount = traverse_bvh(context, &mesh->bvh, rayO, rayD, 0.0f, NoRayIntersection, &meshCandidates);
				
				cranpr_begin("scene", "cast-triangles");
				for (uint32_t faceCandidate = 0; faceCandidate < meshCandidateCount; faceCandidate++)
				{
					// TODO: Lanes
					uint32_t faceIndex = meshCandidates[faceCandidate];

					uint64_t triangleId = ((uint64_t)faceIndex | (uint64_t)candidateIndex << 32);
					if (sourceTriangleId == triangleId) // disallow self intersection
					{
						continue;
					}

					uint32_t vertIndexA = sourceMesh->data.faces.vertexIndices[faceIndex * 3 + 0];
					uint32_t vertIndexB = sourceMesh->data.faces.vertexIndices[faceIndex * 3 + 1];
					uint32_t vertIndexC = sourceMesh->data.faces.vertexIndices[faceIndex * 3 + 2];

					cv3 vertA, vertB, vertC;
					memcpy(&vertA, sourceMesh->data.vertices.data + vertIndexA * 3, sizeof(cv3));
					memcpy(&vertB, sourceMesh->data.vertices.data + vertIndexB * 3, sizeof(cv3));
					memcpy(&vertC, sourceMesh->data.vertices.data + vertIndexC * 3, sizeof(cv3));

					float u, v, w;
					intersectionDistance = triangle_ray_intersection(rayInstanceO, rayD, 0.0f, NoRayIntersection, vertA, vertB, vertC, &u, &v, &w);
					if (intersectionDistance < closestHitInfo.distance)
					{
						closestHitInfo.distance = intersectionDistance;
						closestHitInfo.triangleId = triangleId;
						closestHitInfo.barycentric = (cv3) { u, v, w };
					}
				}
				cranpr_end("scene", "cast-triangles");
				crana_stack_free(&context->stack, meshCandidates);
			}
		}

		// Only populate the rest of our closest hit info object after we traversed our BVH and mesh
		if (closestHitInfo.triangleId != 0)
		{
			uint32_t candidateIndex = closestHitInfo.triangleId >> 32ull;
			uint32_t faceIndex = closestHitInfo.triangleId & 0xFFFFFFFF;
			uint32_t renderableIndex = scene->instances.data[candidateIndex].renderableIndex;
			mesh_t* mesh = &scene->renderables[renderableIndex];

			mesh_source_t* sourceMesh = &scene->meshStore.sources[mesh->meshSourceIndex];

			uint32_t materialIndex = 0;
			material_index_t* materialIndices = sourceMesh->materialIndices;
			for (; materialIndex < sourceMesh->data.materials.count; materialIndex++)
			{
				if (faceIndex < sourceMesh->data.materials.materialBoundaries[materialIndex])
				{
					break;
				}
			}
			closestHitInfo.materialIndex = materialIndices[materialIndex - 1];

			float u = closestHitInfo.barycentric.x;
			float v = closestHitInfo.barycentric.y;
			float w = closestHitInfo.barycentric.z;

			uint32_t normalIndexA = sourceMesh->data.faces.normalIndices[faceIndex * 3 + 0];
			uint32_t normalIndexB = sourceMesh->data.faces.normalIndices[faceIndex * 3 + 1];
			uint32_t normalIndexC = sourceMesh->data.faces.normalIndices[faceIndex * 3 + 2];
			cv3 normalA, normalB, normalC;
			memcpy(&normalA, sourceMesh->data.normals.data + normalIndexA * 3, sizeof(cv3));
			memcpy(&normalB, sourceMesh->data.normals.data + normalIndexB * 3, sizeof(cv3));
			memcpy(&normalC, sourceMesh->data.normals.data + normalIndexC * 3, sizeof(cv3));

			closestHitInfo.normal = cv3_add(cv3_add(cv3_mulf(normalA, u), cv3_mulf(normalB, v)), cv3_mulf(normalC, w));

			uint32_t uvIndexA = sourceMesh->data.faces.uvIndices[faceIndex * 3 + 0];
			uint32_t uvIndexB = sourceMesh->data.faces.uvIndices[faceIndex * 3 + 1];
			uint32_t uvIndexC = sourceMesh->data.faces.uvIndices[faceIndex * 3 + 2];
			cv2 uvA, uvB, uvC;
			memcpy(&uvA, sourceMesh->data.uvs.data + uvIndexA * 2, sizeof(cv2));
			memcpy(&uvB, sourceMesh->data.uvs.data + uvIndexB * 2, sizeof(cv2));
			memcpy(&uvC, sourceMesh->data.uvs.data + uvIndexC * 2, sizeof(cv2));

			closestHitInfo.uv = cv2_add(cv2_add(cv2_mulf(uvA, u), cv2_mulf(uvB, v)), cv2_mulf(uvC, w));

			uint32_t vertIndexA = sourceMesh->data.faces.vertexIndices[faceIndex * 3 + 0];
			uint32_t vertIndexB = sourceMesh->data.faces.vertexIndices[faceIndex * 3 + 1];
			uint32_t vertIndexC = sourceMesh->data.faces.vertexIndices[faceIndex * 3 + 2];

			cv3 vertA, vertB, vertC;
			memcpy(&vertA, sourceMesh->data.vertices.data + vertIndexA * 3, sizeof(cv3));
			memcpy(&vertB, sourceMesh->data.vertices.data + vertIndexB * 3, sizeof(cv3));
			memcpy(&vertC, sourceMesh->data.vertices.data + vertIndexC * 3, sizeof(cv3));

			float du0 = uvB.x - uvA.x;
			float dv0 = uvB.y - uvA.y;
			float du1 = uvC.x - uvA.x;
			float dv1 = uvC.y - uvA.y;

			// TODO: Hack, will have to find a better way to do this.
			// Getting repeating UV value indices causing 0s to appear
			du0 = du0 == 0.0f ? 0.0001f : du0;
			dv0 = dv0 == 0.0f ? 0.0001f : dv0;
			du1 = du1 == 0.0f ? 0.0001f : du1;
			dv1 = dv1 == 0.0f ? 0.0001f : dv1;

			cv3 e0 = cv3_sub(vertB, vertA);
			cv3 e1 = cv3_sub(vertC, vertA);

			// e0=du0T+dv0B (1)
			// e1=du1T+dv1B (2)
			// solve for B
			// (e0-du0T)/dv0
			// plug into (2)
			// e1=du1T+dv1(e0-du0T)/dv0
			// solve for T
			// e1=du1dv0T/dv0+dv1e0/dv0-dv1du0T/dv0
			// dv0e1=du1dv0T+dv1e0-dv1du0T
			// dv0e1-dv1e0=du1dv0T-dv1du0T
			// dv0e1-dv1e0=T(du1dv0-dv1du0)
			// T = (dv0e1-dv1e0)/(dv0du1-dv1du0)

			closestHitInfo.tangent = cv3_mulf(cv3_sub(cv3_mulf(e1, dv0), cv3_mulf(e0, dv1)), cf_rcp(fmaxf(dv0*du1 - dv1*du0, 0.0001f)));
			cran_assert(cv3_sqrlength(closestHitInfo.tangent) > 0.0f);
			closestHitInfo.tangent = cv3_normalize(closestHitInfo.tangent);
			closestHitInfo.bitangent = cv3_normalize(cv3_cross(closestHitInfo.normal, closestHitInfo.tangent));
		}

		crana_stack_free(&context->stack, candidates);
	}
	cran_stat(context->renderStats.intersectionTime += cranpl_timestamp_micro() - intersectionStartTime);

	if (closestHitInfo.triangleId != 0)
	{
		material_index_t materialIndex = closestHitInfo.materialIndex;
		cv3 intersectionPoint = cv3_add(rayO, cv3_mulf(rayD, closestHitInfo.distance));

		uint32_t shaderIndex = renderConfig.useDirectionalMat ? material_directional : materialIndex.typeIndex;
		shader_outputs_t shaderResults = shaders[shaderIndex](scene->materials[materialIndex.typeIndex], materialIndex.dataIndex, context, scene,
			(shader_inputs_t)
			{
				.surface = intersectionPoint,
				.normal = closestHitInfo.normal,
				.tangent = closestHitInfo.tangent,
				.bitangent = closestHitInfo.bitangent,
				.viewDir = rayD,
				.uv = closestHitInfo.uv,
				.triangleId = closestHitInfo.triangleId
			});
		context->depth--;

		cranpr_end("scene", "cast");
		return (ray_hit_t)
		{
			.light = shaderResults.light,
			.surface = shaderResults.shadedSurface,
			.hit = true
		};
	}

	cran_stat(uint64_t skyboxStartTime = cranpl_timestamp_micro());
	cv3 skybox = (cv3) { 1000.0f, 1000.0f, 1000.0f };// sample_hdr(rayD, backgroundSampler);
	cran_stat(context->renderStats.skyboxTime += cranpl_timestamp_micro() - skyboxStartTime);

	context->depth--;

	cranpr_end("scene", "cast");
	return (ray_hit_t)
	{
		.light = skybox,
		.surface = cv3_add(rayO, cv3_mulf(rayD, 1000.0f)),
		.hit = false
	};
}

static shader_outputs_t shader_directional(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	(void)materialData;
	(void)materialIndex;
	(void)context;
	(void)scene;

	return (shader_outputs_t)
	{
		.light = cv3_mulf((cv3) { 1.0f, 1.0f, 1.0f }, fmaxf(cv3_dot(inputs.normal, (cv3) { 0.0f, 0.707f, 0.707f }), 0.2f)),
		.shadedSurface = inputs.surface,
	};
}

// Do we want to handle this some other way?
static shader_outputs_t shader_lambert(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	cranpr_begin("shader", "lambert");
	material_lambert_t lambertData = ((const material_lambert_t* cran_restrict)materialData)[materialIndex];
	// TODO: Consider iteration instead of recursion

	cv3 normal = cv3_normalize(inputs.normal);
	cv4 partialDerivative = sampler_sample(&scene->textureStore, lambertData.bumpSampler, lambertData.bumpTexture, inputs.uv);
	normal = cv3_cross(cv3_add(inputs.tangent, cv3_mulf(normal, partialDerivative.x)), cv3_add(inputs.bitangent, cv3_mulf(normal, partialDerivative.y)));
	normal = cv3_normalize(normal);
	cran_assert(cv3_dot(normal, inputs.normal) >= 0.0f);

	float r1 = random01f(&context->randomSeed);
	float r2 = random01f(&context->randomSeed);
	float pdf;
	cv3 castDir = hemisphere_surface_random_lambert(r1,r2,&pdf);
	castDir = cm3_rotate_cv3(cm3_basis_from_normal(normal), castDir);
	ray_hit_t result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);

	const bool ImportanceSampling = false;
	if (!result.hit && ImportanceSampling)
	{
		float bias;
		castDir = importance_sample_hdr(scene->environment.importance, &bias, &context->randomSeed);
		result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);
		result.light = cv3_mulf(result.light, cf_rcp(bias));
	}
	else
	{
		result.light = cv3_mulf(result.light, cf_rcp(pdf));
	}

	cv3 albedoTint = lambertData.albedoTint;
	cv4 samplerAlbedo = sampler_sample(&scene->textureStore, lambertData.albedoSampler, lambertData.albedoTexture, inputs.uv);

	cv3 albedo = cv3_mul(albedoTint, (cv3) { samplerAlbedo.x, samplerAlbedo.y, samplerAlbedo.z });
	cv3 light = cv3_mulf(result.light, fmaxf(cv3_dot(castDir, normal), 0.0f));

	cranpr_end("shader", "lambert");
	return (shader_outputs_t)
	{
		.light = cv3_mul(light, cv3_mulf(albedo, cran_rpi)),
		.shadedSurface = inputs.surface,
	};
}

static shader_outputs_t shader_microfacet(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	cranpr_begin("shader", "microfacet");
	material_microfacet_t microfacetData = ((const material_microfacet_t* cran_restrict)materialData)[materialIndex];
	// TODO: Consider iteration instead of recursion

	if (microfacetData.maskTexture.id != 0)
	{
		float mask = sampler_sample(&scene->textureStore, microfacetData.maskSampler, microfacetData.maskTexture, inputs.uv).x;
		if (mask == 0.0f)
		{
			ray_hit_t result = cast_scene(context, scene, inputs.surface, inputs.viewDir, inputs.triangleId);
			return (shader_outputs_t)
			{
				.light = result.light,
				.shadedSurface = result.surface,
			};
		}
	}

	cv3 viewDir = cv3_inverse(cv3_normalize(inputs.viewDir));

	cv3 normal = cv3_normalize(inputs.normal);
	cv4 partialDerivative = sampler_sample(&scene->textureStore, microfacetData.bumpSampler, microfacetData.bumpTexture, inputs.uv);
	normal = cv3_cross(cv3_add(inputs.tangent, cv3_mulf(normal, partialDerivative.x)), cv3_add(inputs.bitangent, cv3_mulf(normal, partialDerivative.y)));
	normal = cv3_normalize(normal);
	cran_assert(cv3_dot(normal, inputs.normal) >= 0.0f);

	float gloss = microfacetData.gloss;
	if (microfacetData.glossTexture.id != 0)
	{
		gloss = sampler_sample(&scene->textureStore, microfacetData.glossSampler, microfacetData.glossTexture, inputs.uv).x;
	}
	float roughness = fmaxf(1.0f - gloss, 0.01f);

	float r1 = random01f(&context->randomSeed);
	float r2 = random01f(&context->randomSeed);
	cv3 h = hemisphere_surface_random_ggx_h(r1, r2, roughness);
	h = cm3_rotate_cv3(cm3_basis_from_normal(normal), h);

	// If we don't want any specular, don't bother.
	float specularAmount = cv3_sqrlength(microfacetData.specularTint) > 0.0f ? 1.0f : 0.0f;
	float F = cmi_fresnel_schlick(1.0f, microfacetData.refractiveIndex, h, viewDir) * specularAmount;

	float pdf = 0.0f;
	cv3 light;
	if(random01f(&context->randomSeed) >= F)
	{
		float lr1 = random01f(&context->randomSeed);
		float lr2 = random01f(&context->randomSeed);
		float diffusePDF;
		cv3 castDir = hemisphere_surface_random_lambert(lr1, lr2, &diffusePDF);
		castDir = cm3_rotate_cv3(cm3_basis_from_normal(normal), castDir);

		ray_hit_t result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);
		if (result.hit)
		{
			result.light = cv3_mulf(result.light, light_attenuation(result.surface, inputs.surface));
		}

		light = cv3_mulf(result.light, fmaxf(cv3_dot(castDir, normal), 0.0f));

		cv3 albedoTint = microfacetData.albedoTint;
		cv4 samplerAlbedo = sampler_sample(&scene->textureStore, microfacetData.albedoSampler, microfacetData.albedoTexture, inputs.uv);

		cv3 albedo = cv3_mul(albedoTint, (cv3) { samplerAlbedo.x, samplerAlbedo.y, samplerAlbedo.z });
		light = cv3_mul(light, cv3_mulf(albedo, cran_rpi));

		pdf = diffusePDF * (1.0f - F);
	}
	else
	{
		// https://schuttejoe.github.io/post/ggximportancesamplingpart1/
		// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
		// specular BRDF
		cv3 castDir = cv3_reflect(cv3_inverse(viewDir), h);

		float chn = cv3_dot(h, normal);
		float cvn = cv3_dot(viewDir, normal);
		float cln = cv3_dot(castDir, normal);

		// GGX
		float t = chn * chn*(roughness*roughness - 1.0f) + 1.0f;
		float D = (roughness*roughness)*cf_rcp(cran_pi*t*t);
		// Smith shadowing
		float Gv = cvn*sqrtf(roughness*roughness+(1.0f - roughness*roughness)*cln*cln);
		float Gl = cln*sqrtf(roughness*roughness+(1.0f - roughness*roughness)*cvn*cvn);
		float G = 2.0f*cln*cvn / (Gv + Gl);

		// F(L,H)D(H)G(L,V,H)/(4*N.L*V.N)
		float brdf = F*G*D*cf_rcp(4.0f*cv3_dot(castDir, normal)*cv3_dot(viewDir, normal));
		
		ray_hit_t result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);
		if (result.hit)
		{
			result.light = cv3_mulf(result.light, light_attenuation(result.surface, inputs.surface));
		}

		light = cv3_mulf(result.light, brdf*fmaxf(cv3_dot(castDir, normal), 0.0f));
		light = cv3_mul(light, microfacetData.specularTint); // TODO: absolutely no physics here but makes the render look nicer.

		pdf = D*chn*cf_rcp(4.0f*cv3_dot(castDir,h))*F;
	}
	light = cv3_mulf(light, cf_rcp(pdf));

	cranpr_end("shader", "microfacet");
	return (shader_outputs_t)
	{
		.light = light,
		.shadedSurface = inputs.surface,
	};
}

static shader_outputs_t shader_mirror(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	cranpr_begin("shader", "mirror");
	material_mirror_t mirrorData = ((const material_mirror_t* cran_restrict)materialData)[materialIndex];

	cv3 castDir = cv3_reflect(inputs.viewDir, inputs.normal);
	ray_hit_t result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);

	float lambertCosine = fmaxf(0.0f, cv3_dot(cv3_normalize(castDir), inputs.normal));

	// \int_\omega c*dirac(v,reflect(l,n))*v.n d\omega = 1
	// c*reflect(l,n).n=1
	// c=1/reflect(l,n)
	float normalization = cf_rcp(lambertCosine); // cancels our lambert cosine
	cv3 sceneCast = cv3_mulf(result.light, lambertCosine * normalization);

	cranpr_end("shader", "mirror");
	return (shader_outputs_t)
	{
		.light = cv3_mul(sceneCast, mirrorData.color),
		.shadedSurface = inputs.surface,
	};
}

typedef struct
{
	ray_scene_t scene;

	cv3 origin;
	cv3 right;
	cv3 up;
	cv3 forward;
	float near;

	int32_t imgStride;
	int32_t imgWidth;
	int32_t imgHeight;
	int32_t halfImgWidth;
	int32_t halfImgHeight;

	float xStep;
	float yStep;
} render_data_t;

typedef struct
{
	int32_t xStart;
	int32_t xEnd;
	int32_t yStart;
	int32_t yEnd;
} render_chunk_t;

typedef struct
{
	render_chunk_t* chunks;
	int32_t count;
	cranpl_atomic_int_t next;
} render_queue_t;

typedef struct
{
	render_data_t const* renderData;
	render_context_t context;
	render_queue_t* renderQueue;

	float* cran_restrict hdrOutput;
} thread_context_t;

static void render_scene_async(void* cran_restrict data)
{
	thread_context_t* threadContext = (thread_context_t*)data;
	render_context_t* renderContext = &threadContext->context;
	render_data_t const* renderData = threadContext->renderData;
	render_queue_t* renderQueue = threadContext->renderQueue;

	int32_t chunkIdx = cranpl_atomic_increment(&renderQueue->next);
	for (; chunkIdx < renderQueue->count; chunkIdx = cranpl_atomic_increment(&renderQueue->next))
	{
		render_chunk_t chunk = renderQueue->chunks[chunkIdx];

		// Sample our scene for every pixel in the bitmap. Do we want to upsample?
		for (int32_t y = chunk.yStart; y < chunk.yEnd; y++)
		{
			float yOff = renderData->yStep * (float)y;
			for (int32_t x = chunk.xStart; x < chunk.xEnd; x++)
			{
				float xOff = renderData->xStep * (float)x;

				cv3 sceneColor = { 0 };
				for (uint32_t i = 0; i < renderConfig.samplesPerPixel; i++)
				{
					cran_stat(renderContext->renderStats.primaryRayCount++);

					float randX = xOff + renderData->xStep * (random01f(&renderContext->randomSeed) - 0.5f);
					float randY = yOff + renderData->yStep * (random01f(&renderContext->randomSeed) - 0.5f);

					// Construct our ray as a vector going from our origin to our near plane
					// V = F*n + R*ix*worldWidth/imgWidth + U*iy*worldHeight/imgHeight
					cv3 rayDir = cv3_add(cv3_mulf(renderData->forward, renderData->near), cv3_add(cv3_mulf(renderData->right, randX), cv3_mulf(renderData->up, randY)));

					ray_hit_t hit = cast_scene(renderContext, &renderData->scene, renderData->origin, rayDir, ~0ull);
					sceneColor = cv3_add(sceneColor, cv3_mulf(hit.light, cf_rcp((float)renderConfig.samplesPerPixel)));
				}

				int32_t imgIdx = ((y + renderData->halfImgHeight) * renderData->imgWidth + (x + renderData->halfImgWidth)) * renderData->imgStride;
				threadContext->hdrOutput[imgIdx + 0] = sceneColor.x;
				threadContext->hdrOutput[imgIdx + 1] = sceneColor.y;
				threadContext->hdrOutput[imgIdx + 2] = sceneColor.z;
				threadContext->hdrOutput[imgIdx + 3] = 1.0f;
			}
		}
	}

	cranpr_flush_thread_buffer();
}

int main()
{
	cranpr_init();
	cranpr_begin("cranberray","main");

	renderConfig = (render_config_t)
	{
		.maxDepth = 10,
		.samplesPerPixel = 4,
		.renderWidth = 420,
		.renderHeight = 360,
		.useDirectionalMat = false
	};

	// 3GB for persistent memory
	// 1GB for scratch
	render_context_t mainRenderContext =
	{
		.randomSeed = 57,
		.stack =
		{
			.mem = malloc(1024ull * 1024ull * 1024ull * 3),
			.size = 1024ull*1024ull*1024ull*3
		},
		.scratchStack =
		{
			.mem = malloc(1024 * 1024 * 1024),
			.size = 1024*1024*1024
		}
	};

	static render_data_t mainRenderData;
	uint64_t startTime = cranpl_timestamp_micro();

	generate_scene(&mainRenderContext, &mainRenderData.scene);

	mainRenderData.imgWidth = renderConfig.renderWidth;
	mainRenderData.imgHeight = renderConfig.renderHeight;
	mainRenderData.imgStride = 4;
	mainRenderData.halfImgWidth = mainRenderData.imgWidth / 2;
	mainRenderData.halfImgHeight = mainRenderData.imgHeight / 2;

	// TODO: How do we want to express our camera?
	// Currently simply using the near plane.
	mainRenderData.near = 1.0f;
	float nearHeight = 1.0f;
	float nearWidth = nearHeight * (float)mainRenderData.imgWidth / (float)mainRenderData.imgHeight;
	mainRenderData.xStep = nearWidth / (float)mainRenderData.imgWidth;
	mainRenderData.yStep = nearHeight / (float)mainRenderData.imgHeight;

	mainRenderData.origin = (cv3){ 1.0f, 0.0f, 0.5f };
	mainRenderData.forward = (cv3){ .x = -1.0f,.y = 0.0f,.z = 0.0f };
	mainRenderData.right = (cv3){ .x = 0.0f,.y = 1.0f,.z = 0.0f };
	mainRenderData.up = (cv3){ .x = 0.0f,.y = 0.0f,.z = 1.0f };

	float* cran_restrict hdrImage = crana_stack_alloc(&mainRenderContext.stack, mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride * sizeof(float));

	cran_stat(uint64_t renderStartTime = cranpl_timestamp_micro());

	static render_queue_t mainRenderQueue = { 0 };
	{
		mainRenderQueue.chunks = crana_stack_alloc(&mainRenderContext.stack, sizeof(render_chunk_t) * mainRenderData.imgHeight);
		for (int32_t i = 0; i < mainRenderData.imgHeight; i++)
		{
			mainRenderQueue.chunks[i] = (render_chunk_t)
			{
				.xStart = -mainRenderData.halfImgWidth,
				.xEnd = mainRenderData.halfImgWidth,
				.yStart = -mainRenderData.halfImgHeight + i,
				.yEnd = -mainRenderData.halfImgHeight + i + 1
			};
		}
		mainRenderQueue.count = mainRenderData.imgHeight;
	}

	uint64_t threadStackSize = 1024 * 1024 * 1024 / cranpl_get_core_count() / 2;

	thread_context_t* threadContexts = crana_stack_alloc(&mainRenderContext.stack, sizeof(thread_context_t) * cranpl_get_core_count());
	void** threadHandles = crana_stack_alloc(&mainRenderContext.stack, sizeof(void*) * cranpl_get_core_count() - 1);
	for (uint32_t i = 0; i < cranpl_get_core_count(); i++)
	{
		threadContexts[i] = (thread_context_t)
		{
			.renderData = &mainRenderData,
			.renderQueue = &mainRenderQueue,
			.context = 
			{
				.randomSeed = random(&mainRenderContext.randomSeed),
				.stack = 
				{
					.mem = crana_stack_alloc(&mainRenderContext.stack, threadStackSize),
					.size = threadStackSize
				},
				.scratchStack = 
				{
					.mem = crana_stack_alloc(&mainRenderContext.stack, threadStackSize),
					.size = threadStackSize
				}
			},
			.hdrOutput = hdrImage
		};
	}

	for (uint32_t i = 0; i < cranpl_get_core_count() - 1; i++)
	{
		threadHandles[i] = cranpl_create_thread(&render_scene_async, &threadContexts[i]);
	}

	render_scene_async(&threadContexts[cranpl_get_core_count() - 1]);

	for (uint32_t i = 0; i < cranpl_get_core_count() - 1; i++)
	{
		cranpl_wait_on_thread(threadHandles[i]);
	}

	cran_stat(mainRenderContext.renderStats.renderTime = (cranpl_timestamp_micro() - renderStartTime));

	// Image Space Effects
	bool enableImageSpace = true;
	if(enableImageSpace)
	{
		cran_stat(uint64_t imageSpaceStartTime = cranpl_timestamp_micro());
		// reinhard tonemapping
		for (int32_t y = 0; y < mainRenderData.imgHeight; y++)
		{
			for (int32_t x = 0; x < mainRenderData.imgWidth; x++)
			{
				int32_t readIndex = (y * mainRenderData.imgWidth + x) * mainRenderData.imgStride;

				hdrImage[readIndex + 0] = hdrImage[readIndex + 0] / (hdrImage[readIndex + 0] + 1.0f);
				hdrImage[readIndex + 1] = hdrImage[readIndex + 1] / (hdrImage[readIndex + 1] + 1.0f);
				hdrImage[readIndex + 2] = hdrImage[readIndex + 2] / (hdrImage[readIndex + 2] + 1.0f);
			}
		}
		cran_stat(mainRenderContext.renderStats.imageSpaceTime = cranpl_timestamp_micro() - imageSpaceStartTime);
	}

	mainRenderContext.renderStats.totalTime = cranpl_timestamp_micro() - startTime;

	// Convert HDR to 8 bit bitmap
	{
		uint8_t* cran_restrict bitmap = crana_stack_alloc(&mainRenderContext.scratchStack, mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride);
		for (int32_t i = 0; i < mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride; i+=mainRenderData.imgStride)
		{
			bitmap[i + 0] = (uint8_t)(255.99f * sqrtf(fminf(hdrImage[i + 2], 1.0f)));
			bitmap[i + 1] = (uint8_t)(255.99f * sqrtf(fminf(hdrImage[i + 1], 1.0f)));
			bitmap[i + 2] = (uint8_t)(255.99f * sqrtf(fminf(hdrImage[i + 0], 1.0f)));
			bitmap[i + 3] = (uint8_t)(255.99f * hdrImage[i + 3]);
		}

		cranpl_write_bmp("render.bmp", bitmap, mainRenderData.imgWidth, mainRenderData.imgHeight);
		cranpl_open_file_with_default_app("render.bmp");
		crana_stack_free(&mainRenderContext.scratchStack, bitmap);
	}

	for (uint32_t i = 0; i < cranpl_get_core_count(); i++)
	{
		merge_render_stats(&mainRenderContext.renderStats, &threadContexts[i].context.renderStats);
	}

	// Print stats
	{
		FILE* fileHandle;
		errno_t error = fopen_s(&fileHandle, "cranberray_stats.txt", "w");
		if (error == 0)
		{
			fprintf(fileHandle, "Total Time: %f\n", micro_to_seconds(mainRenderContext.renderStats.totalTime));
#ifdef cran_stats
			fprintf(fileHandle, "\tScene Generation Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.sceneGenerationTime), (float)mainRenderContext.renderStats.sceneGenerationTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
			fprintf(fileHandle, "\tRender Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.renderTime), (float)mainRenderContext.renderStats.renderTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
			fprintf(fileHandle, "----------\n");
			fprintf(fileHandle, "Accumulated Threading Data\n");
			fprintf(fileHandle, "\t\tIntersection Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.intersectionTime), (float)mainRenderContext.renderStats.intersectionTime / (float)mainRenderContext.renderStats.renderTime * 100.0f);
			fprintf(fileHandle, "\t\t\tBVH Traversal Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.bvhTraversalTime), (float)mainRenderContext.renderStats.bvhTraversalTime / (float)mainRenderContext.renderStats.intersectionTime * 100.0f);
			fprintf(fileHandle, "\t\t\t\tBVH Tests: %" PRIu64 "\n", mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount);
			fprintf(fileHandle, "\t\t\t\t\tBVH Hits: %" PRIu64 "[%.2f%%]\n", mainRenderContext.renderStats.bvhHitCount, (float)mainRenderContext.renderStats.bvhHitCount / (float)(mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount) * 100.0f);
			fprintf(fileHandle, "\t\t\t\t\t\tBVH Leaf Hits: %" PRIu64 "[%.2f%%]\n", mainRenderContext.renderStats.bvhLeafHitCount, (float)mainRenderContext.renderStats.bvhLeafHitCount / (float)mainRenderContext.renderStats.bvhHitCount * 100.0f);
			fprintf(fileHandle, "\t\t\t\t\tBVH Misses: %" PRIu64 "[%.2f%%]\n", mainRenderContext.renderStats.bvhMissCount, (float)mainRenderContext.renderStats.bvhMissCount / (float)(mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount) * 100.0f);
			fprintf(fileHandle, "\t\tSkybox Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.skyboxTime), (float)mainRenderContext.renderStats.skyboxTime / (float)mainRenderContext.renderStats.renderTime * 100.0f);
			fprintf(fileHandle, "----------\n");
			fprintf(fileHandle, "\tImage Space Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.imageSpaceTime), (float)mainRenderContext.renderStats.imageSpaceTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
			fprintf(fileHandle, "\n");
			fprintf(fileHandle, "MRays/seconds: %f\n", (float)mainRenderContext.renderStats.rayCount / micro_to_seconds(mainRenderContext.renderStats.renderTime) / 1000000.0f);
			fprintf(fileHandle, "Rays Fired: %" PRIu64 "\n", mainRenderContext.renderStats.rayCount);
			fprintf(fileHandle, "\tCamera Rays Fired: %" PRIu64 " [%.2f%%]\n", mainRenderContext.renderStats.primaryRayCount, (float)mainRenderContext.renderStats.primaryRayCount / (float)mainRenderContext.renderStats.rayCount * 100.0f);
			fprintf(fileHandle, "\tBounce Rays Fired: %" PRIu64 " [%.2f%%]\n", mainRenderContext.renderStats.rayCount - mainRenderContext.renderStats.primaryRayCount, (float)(mainRenderContext.renderStats.rayCount - mainRenderContext.renderStats.primaryRayCount) / (float)mainRenderContext.renderStats.rayCount * 100.0f);
			fprintf(fileHandle, "\n");
			fprintf(fileHandle, "BVH\n");
			fprintf(fileHandle, "\tBVH Node Count: %" PRIu64 "\n", mainRenderContext.renderStats.bvhNodeCount);
			fprintf(fileHandle, "Memory\n");
			fprintf(fileHandle, "\tMain Stack\n");
			fprintf(fileHandle, "\t\tStack Size: %" PRIu64 "\n", mainRenderContext.stack.size);
			fprintf(fileHandle, "\t\tFinal Stack Top: %" PRIu64 " [%.2f%%]\n", mainRenderContext.stack.top, (float)mainRenderContext.stack.top / (float)mainRenderContext.stack.size*100.0f);
			fprintf(fileHandle, "\tScratch Stack\n");
			fprintf(fileHandle, "\t\tStack Size: %" PRIu64 "\n", mainRenderContext.scratchStack.size);
			fprintf(fileHandle, "\t\tStack Top: %" PRIu64 " [%.2f%%]\n", mainRenderContext.scratchStack.top, (float)mainRenderContext.scratchStack.top / (float)mainRenderContext.scratchStack.size*100.0f);
#endif // cran_stats
			fclose(fileHandle);
		}
	}

	// Not worrying about individual memory cleanup, stack allocator is cleaned up in one swoop anyways.
	free(mainRenderContext.stack.mem);
	free(mainRenderContext.scratchStack.mem);

	for (uint32_t i = 0; i < mainRenderData.scene.textureStore.nextTexture; i++)
	{
		stbi_image_free(mainRenderData.scene.textureStore.textures[i].data);
	}

	cranpr_end("cranberray","main");
	cranpr_flush_thread_buffer();
	cranpr_write_to_file("cranberray_profile.json");

	cranpr_terminate();
	return 0;
}
