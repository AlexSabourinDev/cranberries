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
#include <assert.h>
#include <string.h>
#include <time.h>

#include "stb_image.h"
#include "cranberry_platform.h"
#include "cranberry_loader.h"
#include "cranberry_math.h"

#define CRANPR_IMPLEMENTATION
//#define CRANPR_ENABLED
#include "cranberry_profiler.h"

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
	assert(!stack->locked);
	uint8_t* ptr = stack->mem + stack->top;
	stack->top += size;
	assert(stack->top <= stack->size);
	return ptr;
}

// TODO: Deallocation is error prone, what if we change the size of the alloc but not the release?
// Tricky because I've decided to allow the allocations to grow and still reference the initial allocation
// pointer as long as no intermediate allocations are made to the allocator. (TODO: Likely want to either change the scheme, or add a key type mechanism)
void crana_stack_free(crana_stack_t* stack, uint64_t size)
{
	assert(!stack->locked);
	stack->top -= size;
	assert(stack->top <= stack->size);
}

// Lock our stack, this means we have free range on the memory pointer
// Commit the pointer back to the stack to complete the operation.
// While the stack is locked, alloc and free cannot be called.
void* crana_stack_lock(crana_stack_t* stack)
{
	assert(!stack->locked);
	stack->locked = true;
	return stack->mem + stack->top;
}

// Commit our new pointer. You can have moved it forward or backwards
// Make sure that the pointer is still from the same memory pool as the stack
void crana_stack_commit(crana_stack_t* stack, void* memory)
{
	assert(stack->locked);
	assert((uint64_t)((uint8_t*)memory - stack->mem) <= stack->size);
	stack->top = (uint8_t*)memory - stack->mem;
	stack->locked = false;
}

// Cancel our lock, all the memory we were planning on commiting can be ignored
void crana_stack_revert(crana_stack_t* stack)
{
	assert(stack->locked);
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

static cv3 box_random(random_seed_t* seed)
{
	return cv3_mulf((cv3) { random01f(seed)-0.5f, random01f(seed)-0.5f, random01f(seed)-0.5f }, 2.0f);
}

typedef struct
{
	float* cran_restrict image;

	float* cran_restrict cdf2d;
	float* cran_restrict luminance2d;

	float* cran_restrict cdf1d;
	float* cran_restrict sum1d;
	float sumTotal;

	int32_t width;
	int32_t height;
	int32_t stride;
} sampler_hdr_t;

typedef struct
{
	uint8_t* cran_restrict image;

	int32_t width;
	int32_t height;
	int32_t stride;
} sampler_u8_t;

static cv3 sample_rgb_f32(cv2 uv, sampler_u8_t sampler)
{
	// TODO: Bilerp
	uv.y = cf_frac(uv.y);
	uv.x = cf_frac(uv.x);

	uv.y = uv.y < 0.0f ? 1.0f + uv.y : uv.y;
	uv.x = uv.x < 0.0f ? 1.0f + uv.x : uv.x;

	float readY = uv.y * (float)sampler.height;
	float readX = uv.x * (float)sampler.width;
	int32_t readIndex = ((int32_t)floorf(readY) * sampler.width + (int32_t)floorf(readX)) * sampler.stride;

	cv3 color;
	color.x = (float)sampler.image[readIndex + 0] / 255.0f;
	color.y = (float)sampler.image[readIndex + 1] / 255.0f;
	color.z = (float)sampler.image[readIndex + 2] / 255.0f;

	return color;
}


static cv3 sample_hdr(cv3 v, sampler_hdr_t sampler)
{
	float azimuth, theta;
	cv3_to_spherical(v, &azimuth, &theta);

	// TODO: lerp
	int32_t readY = (int32_t)(fminf(theta * cran_rpi, 0.999f) * (float)sampler.height);
	int32_t readX = (int32_t)(fminf(azimuth * cran_rtao, 0.999f) * (float)sampler.width);
	int32_t readIndex = (readY * sampler.width + readX) * sampler.stride;

	cv3 color;
	color.x = sampler.image[readIndex + 0];
	color.y = sampler.image[readIndex + 1];
	color.z = sampler.image[readIndex + 2];

	return color;
}

static cv3 importance_sample_hdr(sampler_hdr_t imageset, float* cran_restrict outBias, random_seed_t* seed)
{
	float ycdf = random01f(seed);
	int y = 0;
	for (; y < imageset.height; y++)
	{
		if (imageset.cdf1d[y] >= ycdf)
		{
			break;
		}
	}

	float xcdf = random01f(seed);
	int x = 0;
	for (; x < imageset.width; x++)
	{
		if (imageset.cdf2d[y*imageset.width + x] >= xcdf)
		{
			break;
		}
	}

	float biasy = imageset.sum1d[y] * cf_rcp(imageset.sumTotal);
	float biasx = imageset.luminance2d[y * imageset.width + x] * cf_rcp(imageset.sum1d[y]);
	float bias = biasy * biasx;

	cv3 direction = cv3_from_spherical(((float)y / (float)imageset.height)*cran_pi, ((float)x / (float)imageset.width)*cran_tao, 1.0f);

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
			uint32_t left;
			uint32_t right;
		} jumps;

		uint32_t index;
	} indices;
	bool isLeaf;
} bvh_jump_t;

typedef struct
{
	caabb* bounds;
	bvh_jump_t* jumps;
} bvh_t;

typedef enum
{
	material_lambert,
	material_mirror,
	material_count
} material_type_e;

typedef struct
{
	cv3 albedo;
} material_lambert_t;

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
	cranl_mesh_t data;
	bvh_t bvh;
	material_index_t* materialIndices;
} mesh_t;

typedef struct
{
	cv3 pos;
	uint32_t renderableIndex;
} instance_t;

typedef struct
{
	void* cran_restrict materials[material_count];
	mesh_t* cran_restrict renderables;

	struct
	{
		instance_t* data;
		uint32_t count;
	} instances;

	bvh_t bvh; // TODO: BVH!
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
	cv3 viewDir;
	cv2 uv;
	uint64_t triangleId;
} shader_inputs_t;

typedef cv3(material_shader_t)(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs);

static material_shader_t shader_lambert;
static material_shader_t shader_mirror;
material_shader_t* shaders[material_count] =
{
	shader_lambert,
	shader_mirror
};

static int index_aabb_sort_min_x(const void* cran_restrict l, const void* cran_restrict r)
{
	const index_aabb_pair_t* cran_restrict left = (const index_aabb_pair_t*)l;
	const index_aabb_pair_t* cran_restrict right = (const index_aabb_pair_t*)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)(left->bound.min.x - right->bound.min.x);
}

static int index_aabb_sort_min_y(const void* cran_restrict l, const void* cran_restrict r)
{
	const index_aabb_pair_t* cran_restrict left = (const index_aabb_pair_t* cran_restrict)l;
	const index_aabb_pair_t* cran_restrict right = (const index_aabb_pair_t* cran_restrict)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)(left->bound.min.y - right->bound.min.y);
}

static int index_aabb_sort_min_z(const void* cran_restrict l, const void* cran_restrict r)
{
	const index_aabb_pair_t* left = (const index_aabb_pair_t* cran_restrict)l;
	const index_aabb_pair_t* right = (const index_aabb_pair_t* cran_restrict)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)(left->bound.min.z - right->bound.min.z);
}

static bvh_t build_bvh(render_context_t* context, index_aabb_pair_t* leafs, uint32_t leafCount)
{
	assert(leafCount > 0);

	int(*sortFuncs[3])(const void* cran_restrict l, const void* cran_restrict r) = { index_aabb_sort_min_x, index_aabb_sort_min_y, index_aabb_sort_min_z };

	typedef struct
	{
		index_aabb_pair_t* start;
		uint32_t count;
		uint32_t* parentIndex;
	} bvh_workgroup_t;

	// Simple ring buffer
	uint32_t workgroupSize = 100000;
	bvh_workgroup_t* bvhWorkgroup = (bvh_workgroup_t*)crana_stack_alloc(&context->scratchStack, sizeof(bvh_workgroup_t) * workgroupSize);
	uint32_t workgroupQueueEnd = 1;

	bvhWorkgroup[0].start = leafs;
	bvhWorkgroup[0].count = leafCount;
	bvhWorkgroup[0].parentIndex = NULL;

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
			*(bvhWorkgroup[workgroupIter].parentIndex) = (uint32_t)(buildingBVHIter - buildingBVHStart);
		}

		caabb bounds = start[0].bound;
		for (uint32_t i = 1; i < count; i++)
		{
			bounds.min = cv3_min(start[i].bound.min, bounds.min);
			bounds.max = cv3_max(start[i].bound.max, bounds.max);
		}

		bool isLeaf = (count == 1);
		buildingBVHIter->bound = bounds;
		buildingBVHIter->jump = (bvh_jump_t)
		{
			.indices.index = start[0].index,
			.isLeaf = isLeaf
		};

		if (!isLeaf)
		{
			// TODO: Since we're doing all the iteration work in the sort, maybe we could also do the partitioning in the sort?
			uint32_t axis = randomRange(&context->randomSeed, 0, 3);
			qsort(start, count, sizeof(index_aabb_pair_t), sortFuncs[axis]);

			float boundsCenter = ((&bounds.max.x)[axis] - (&bounds.min.x)[axis]) * 0.5f;
			uint32_t centerIndex = count / 2;
			for (uint32_t i = 0; i < count; i++)
			{
				float centroid = ((&start[i].bound.max.x)[axis] - (&start[i].bound.min.x)[axis]) * 0.5f;
				if (centroid > boundsCenter)
				{
					centerIndex = i;
					break;
				}
			}

			bvhWorkgroup[workgroupQueueEnd].start = start;
			bvhWorkgroup[workgroupQueueEnd].count = centerIndex;
			bvhWorkgroup[workgroupQueueEnd].parentIndex = &buildingBVHIter->jump.indices.jumps.left;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % workgroupSize;
			assert(workgroupQueueEnd != workgroupIter);

			bvhWorkgroup[workgroupQueueEnd].start = start + centerIndex;
			bvhWorkgroup[workgroupQueueEnd].count = count - centerIndex;
			bvhWorkgroup[workgroupQueueEnd].parentIndex = &buildingBVHIter->jump.indices.jumps.right;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % workgroupSize;
			assert(workgroupQueueEnd != workgroupIter);
		}

		buildingBVHIter++;
	}

	uint32_t bvhSize = (uint32_t)(buildingBVHIter - buildingBVHStart);
	bvh_t builtBVH =
	{
		.bounds = crana_stack_alloc(&context->stack, sizeof(caabb) * bvhSize),
		.jumps = crana_stack_alloc(&context->stack, sizeof(bvh_jump_t) * bvhSize),
	};

	for (uint32_t i = 0; i < bvhSize; i++)
	{
		// TODO: Pack leaf nodes at the end
		builtBVH.bounds[i] = buildingBVHStart[i].bound;
		builtBVH.jumps[i] = buildingBVHStart[i].jump;
	}

	// Don't bother commiting our lock, we don't need this anymore
	crana_stack_revert(&context->scratchStack);
	crana_stack_free(&context->scratchStack, sizeof(bvh_workgroup_t) * workgroupSize);

	context->renderStats.bvhNodeCount = bvhSize;
	return builtBVH;
}

// Allocates candidates from bottom stack
// Uses top stack for working memory
// TODO: Maybe 2 seperate stacks would be better than using a single stack with 2 apis
static uint32_t traverse_bvh(render_context_t* context, bvh_t const* bvh, cv3 rayO, cv3 rayD, float rayMin, float rayMax, uint32_t** candidates)
{
	cranpr_begin("bvh", "traverse");

	*candidates = crana_stack_lock(&context->stack); // Allocate nothing, but we're going to be growing it
	uint32_t* candidateIter = *candidates;

	uint64_t traversalStartTime = cranpl_timestamp_micro();

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
		if (intersections > 0)
		{
			for (uint32_t i = 0; i < activeLaneCount; i++)
			{
				if (intersections & (1 << i))
				{
					uint32_t nodeIndex = testQueueIter[i];

					context->renderStats.bvhHitCount++; 

					// TODO: What if instead of testing our isLeaf flag, all leaf nodes were packed at the end of the
					// tree array? Then we wouldn't need to wait to load our jump into memory and we can work purely with our index.
					// We have to have loaded our index into memory recently for our bounds anyways.
					bool isLeaf = bvh->jumps[nodeIndex].isLeaf;
					if (isLeaf)
					{
						context->renderStats.bvhLeafHitCount++;

						// Grows our candidate pointer
						*candidateIter = bvh->jumps[nodeIndex].indices.index;
						candidateIter++;
					}
					else
					{
						*testQueueEnd = bvh->jumps[nodeIndex].indices.jumps.left;
						testQueueEnd++;
						*testQueueEnd = bvh->jumps[nodeIndex].indices.jumps.right;
						testQueueEnd++;
					}
				}
				else
				{
					context->renderStats.bvhMissCount++;
				}
			}
		}
		else
		{
			context->renderStats.bvhMissCount += activeLaneCount;
		}


		testQueueIter += activeLaneCount;
	}

	crana_stack_revert(&context->scratchStack);
	crana_stack_commit(&context->stack, candidateIter);

	context->renderStats.bvhTraversalTime += cranpl_timestamp_micro() - traversalStartTime;

	cranpr_end("bvh", "traverse");
	return (uint32_t)(candidateIter - *candidates);
}

static void generate_scene(render_context_t* context, ray_scene_t* scene)
{
	cranpr_begin("scene", "generate");
	uint64_t startTime = cranpl_timestamp_micro();

	static material_lambert_t lamberts[3] = { {.albedo = { 0.8f, 0.9f, 1.0f } },  {.albedo = { 0.1f, 0.1f, 0.1f } }, {.albedo = {1.0f, 1.0f, 1.0f} } };
	static material_mirror_t mirrors[2] = { {.color = { 1.0f, 1.0f, 1.0f } }, { .color = { 0.1f, 0.8f, 0.5f } } };

	static material_index_t materialIndices[] = 
	{
		{.dataIndex = 0,.typeIndex = material_mirror },
		{.dataIndex = 2,.typeIndex = material_lambert },
		{.dataIndex = 2,.typeIndex = material_lambert },
		{.dataIndex = 0,.typeIndex = material_lambert },
	};

	static instance_t instances[1];
	static mesh_t mesh;

	// Mesh
	{
		// TODO: We likely don't want a stack allocator here
		// clean up would be too tedious, think of a way to encapsulate meshes
		mesh.data = cranl_obj_load("mitsuba-sphere.obj", cranl_flip_yz,
			(cranl_allocator_t)
			{
				.instance = &context->stack,
				.alloc = &crana_stack_alloc,
				.free = &crana_stack_free
			});
		uint32_t meshLeafCount = mesh.data.faces.count;
		index_aabb_pair_t* leafs = crana_stack_alloc(&context->scratchStack, sizeof(index_aabb_pair_t) * meshLeafCount);

		for (uint32_t i = 0; i < meshLeafCount; i++)
		{
			leafs[i].index = i;

			uint32_t vertIndexA = mesh.data.faces.vertexIndices[i * 3 + 0];
			uint32_t vertIndexB = mesh.data.faces.vertexIndices[i * 3 + 1];
			uint32_t vertIndexC = mesh.data.faces.vertexIndices[i * 3 + 2];

			cv3 vertA, vertB, vertC;
			memcpy(&vertA, mesh.data.vertices.data + vertIndexA * 3, sizeof(cv3));
			memcpy(&vertB, mesh.data.vertices.data + vertIndexB * 3, sizeof(cv3));
			memcpy(&vertC, mesh.data.vertices.data + vertIndexC * 3, sizeof(cv3));

			leafs[i].bound.min = cv3_min(cv3_min(vertA, vertB), vertC);
			leafs[i].bound.max = cv3_max(cv3_max(vertA, vertB), vertC);

			// If our bounds have no volume, add a surrounding shell
			if (fabsf(leafs[i].bound.max.x - leafs[i].bound.min.x) < FLT_EPSILON)
			{
				leafs[i].bound.max.x += 0.001f;
				leafs[i].bound.min.x -= 0.001f;
			}

			if (fabsf(leafs[i].bound.max.y - leafs[i].bound.min.y) < FLT_EPSILON)
			{
				leafs[i].bound.max.y += 0.001f;
				leafs[i].bound.min.y -= 0.001f;
			}

			if (fabsf(leafs[i].bound.max.z - leafs[i].bound.min.z) < FLT_EPSILON)
			{
				leafs[i].bound.max.z += 0.001f;
				leafs[i].bound.min.z -= 0.001f;
			}
		}

		mesh.bvh = build_bvh(context, leafs, meshLeafCount);
		crana_stack_free(&context->scratchStack, sizeof(index_aabb_pair_t) * meshLeafCount);
	}
	mesh.materialIndices = materialIndices;

	for (uint32_t i = 0; i < 1; i++)
	{
		instances[i] = (instance_t)
		{
			.pos = { 0.0f, 0.0f, 0.0f },
			.renderableIndex = 0
		};
	}

	// Output our scene
	*scene = (ray_scene_t)
	{
		.instances =
		{
			.data = instances,
			.count = 1
		},
		.renderables = &mesh,
		.materials =
		{
			[material_lambert] = lamberts,
			[material_mirror] = mirrors
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

			mesh_t* meshData = &scene->renderables[renderableIndex];
			for (uint32_t vert = 0; vert < meshData->data.vertices.count; vert++)
			{
				// TODO: type pun here
				cv3 vertex;
				memcpy(&vertex, meshData->data.vertices.data + vert * 3, sizeof(cv3));

				leafs[i].bound.min = cv3_min(leafs[i].bound.min, cv3_add(vertex, pos));
				leafs[i].bound.max = cv3_max(leafs[i].bound.max, cv3_add(vertex, pos));
			}
		}

		scene->bvh = build_bvh(context, leafs, leafCount);
		crana_stack_free(&context->scratchStack, sizeof(index_aabb_pair_t) * leafCount);
	}

	context->renderStats.sceneGenerationTime = cranpl_timestamp_micro() - startTime;
	cranpr_end("scene", "generate");
}

typedef struct
{
	cv3 light;
	cv3 surface;
	bool hit;
} ray_hit_t;

sampler_hdr_t backgroundSampler;
sampler_u8_t checkerboardSampler;

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

	context->renderStats.rayCount++;

	const float NoRayIntersection = FLT_MAX;

	struct
	{
		float distance;
		cv3 normal;
		cv2 uv;
		material_index_t materialIndex;
		uint64_t triangleId;
	} closestHitInfo = { 0 };
	closestHitInfo.distance = NoRayIntersection;

	uint64_t intersectionStartTime = cranpl_timestamp_micro();

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
			material_index_t* materialIndices = mesh->materialIndices;

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

					uint32_t vertIndexA = mesh->data.faces.vertexIndices[faceIndex * 3 + 0];
					uint32_t vertIndexB = mesh->data.faces.vertexIndices[faceIndex * 3 + 1];
					uint32_t vertIndexC = mesh->data.faces.vertexIndices[faceIndex * 3 + 2];

					cv3 vertA, vertB, vertC;
					memcpy(&vertA, mesh->data.vertices.data + vertIndexA * 3, sizeof(cv3));
					memcpy(&vertB, mesh->data.vertices.data + vertIndexB * 3, sizeof(cv3));
					memcpy(&vertC, mesh->data.vertices.data + vertIndexC * 3, sizeof(cv3));

					float u, v, w;
					intersectionDistance = triangle_ray_intersection(rayInstanceO, rayD, 0.0f, NoRayIntersection, vertA, vertB, vertC, &u, &v, &w);
					if (intersectionDistance < closestHitInfo.distance)
					{
						uint32_t materialIndex = 0;
						for (; materialIndex < mesh->data.materials.count; materialIndex++)
						{
							if (faceIndex < mesh->data.materials.materialBoundaries[materialIndex])
							{
								break;
							}
						}
						closestHitInfo.materialIndex = materialIndices[materialIndex - 1];
						closestHitInfo.distance = intersectionDistance;
						closestHitInfo.triangleId = triangleId;

						uint32_t normalIndexA = mesh->data.faces.normalIndices[faceIndex * 3 + 0];
						uint32_t normalIndexB = mesh->data.faces.normalIndices[faceIndex * 3 + 1];
						uint32_t normalIndexC = mesh->data.faces.normalIndices[faceIndex * 3 + 2];
						cv3 normalA, normalB, normalC;
						memcpy(&normalA, mesh->data.normals.data + normalIndexA * 3, sizeof(cv3));
						memcpy(&normalB, mesh->data.normals.data + normalIndexB * 3, sizeof(cv3));
						memcpy(&normalC, mesh->data.normals.data + normalIndexC * 3, sizeof(cv3));

						closestHitInfo.normal = cv3_add(cv3_add(cv3_mulf(normalA, u), cv3_mulf(normalB, v)), cv3_mulf(normalC, w));

						uint32_t uvIndexA = mesh->data.faces.uvIndices[faceIndex * 3 + 0];
						uint32_t uvIndexB = mesh->data.faces.uvIndices[faceIndex * 3 + 1];
						uint32_t uvIndexC = mesh->data.faces.uvIndices[faceIndex * 3 + 2];
						cv2 uvA, uvB, uvC;
						memcpy(&uvA, mesh->data.uvs.data + uvIndexA * 2, sizeof(cv2));
						memcpy(&uvB, mesh->data.uvs.data + uvIndexB * 2, sizeof(cv2));
						memcpy(&uvC, mesh->data.uvs.data + uvIndexC * 2, sizeof(cv2));

						closestHitInfo.uv = cv2_add(cv2_add(cv2_mulf(uvA, u), cv2_mulf(uvB, v)), cv2_mulf(uvC, w));
					}
				}
				cranpr_end("scene", "cast-triangles");
				crana_stack_free(&context->stack, meshCandidateCount * sizeof(uint32_t));
			}
		}

		crana_stack_free(&context->stack, candidateCount * sizeof(uint32_t));
	}
	context->renderStats.intersectionTime += cranpl_timestamp_micro() - intersectionStartTime;

	if (closestHitInfo.distance != NoRayIntersection)
	{
		material_index_t materialIndex = closestHitInfo.materialIndex;
		cv3 intersectionPoint = cv3_add(rayO, cv3_mulf(rayD, closestHitInfo.distance));

		cv3 light = shaders[materialIndex.typeIndex](scene->materials[materialIndex.typeIndex], materialIndex.dataIndex, context, scene,
			(shader_inputs_t)
			{
				.surface = intersectionPoint,
				.normal = closestHitInfo.normal,
				.viewDir = rayD,
				.uv = closestHitInfo.uv,
				.triangleId = closestHitInfo.triangleId
			});
		context->depth--;

		cranpr_end("scene", "cast");
		return (ray_hit_t)
		{
			.light = light,
			.surface = intersectionPoint,
			.hit = true
		};
	}

	uint64_t skyboxStartTime = cranpl_timestamp_micro();
	cv3 skybox = (cv3) { 1.0f, 1.0f, 1.0f };// sample_hdr(rayD, backgroundSampler);
	context->renderStats.skyboxTime += cranpl_timestamp_micro() - skyboxStartTime;

	context->depth--;

	cranpr_end("scene", "cast");
	return (ray_hit_t)
	{
		.light = skybox,
		.surface = cv3_add(rayO, cv3_mulf(rayD, 10000.0f)),
		.hit = false
	};
}

// Do we want to handle this some other way?
static cv3 shader_lambert(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	cranpr_begin("shader", "lambert");
	material_lambert_t lambertData = ((const material_lambert_t* cran_restrict)materialData)[materialIndex];
	// TODO: Consider iteration instead of recursion

	float r1 = random01f(&context->randomSeed);
	float r2 = random01f(&context->randomSeed);
	float pdf;
	cv3 castDir = hemisphere_surface_random_lambert(r1,r2,&pdf);
	castDir = cm3_rotate_cv3(cm3_basis_from_normal(inputs.normal), castDir);
	ray_hit_t result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);

	const bool ImportanceSampling = false;
	if (!result.hit && ImportanceSampling)
	{
		float bias;
		castDir = importance_sample_hdr(backgroundSampler, &bias, &context->randomSeed);
		result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);
		result.light = cv3_mulf(result.light, cf_rcp(bias));
	}
	else
	{
		result.light = cv3_mulf(result.light, cf_rcp(pdf));
	}

	cv3 albedo = (cv3) {1.0f,1.0f,1.0f};// sample_rgb_f32(inputs.uv, checkerboardSampler);
	albedo = cv3_mul(albedo,lambertData.albedo);

	float attenuation = result.hit ? light_attenuation(result.surface, inputs.surface) : 1.0f;
	cv3 light = cv3_mulf(result.light, fmaxf(cv3_dot(castDir, inputs.normal), 0.0f) * attenuation);

	cranpr_end("shader", "lambert");
	return cv3_mul(light, cv3_mulf(albedo, cran_rpi));
}

static cv3 shader_mirror(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	cranpr_begin("shader", "mirror");
	material_mirror_t mirrorData = ((const material_mirror_t* cran_restrict)materialData)[materialIndex];

	cv3 castDir = cv3_reflect(inputs.viewDir, inputs.normal);
	ray_hit_t result = cast_scene(context, scene, inputs.surface, castDir, inputs.triangleId);

	float lambertCosine = fmaxf(0.0f, cv3_dot(cv3_normalize(castDir), inputs.normal));
	float attenuation = result.hit ? light_attenuation(result.surface, inputs.surface) : 1.0f;
	cv3 sceneCast = cv3_mulf(cv3_mulf(result.light, lambertCosine), attenuation);

	cranpr_end("shader", "mirror");
	return cv3_mul(sceneCast, mirrorData.color);
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
					renderContext->renderStats.primaryRayCount++;

					float randX = xOff + renderData->xStep * (random01f(&renderContext->randomSeed) * 0.5f - 0.5f);
					float randY = yOff + renderData->yStep * (random01f(&renderContext->randomSeed) * 0.5f - 0.5f);

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
		.maxDepth = 99,
		.samplesPerPixel = 4,
		.renderWidth = 1024,
		.renderHeight = 768
	};

	// 3GB for persistent memory
	// 1GB for scratch
	render_context_t mainRenderContext =
	{
		.randomSeed = (uint32_t)time(0),
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

	// Checkerboad
	{
		checkerboardSampler.image = stbi_load("checkerboard.png", &checkerboardSampler.width, &checkerboardSampler.height, &checkerboardSampler.stride, 0);
	}

	// Environment map
	{
		backgroundSampler.image = stbi_loadf("background_4k.hdr", &backgroundSampler.width, &backgroundSampler.height, &backgroundSampler.stride, 0);
		backgroundSampler.cdf2d = (float*)crana_stack_alloc(&mainRenderContext.stack, sizeof(float) * backgroundSampler.width * backgroundSampler.height);
		backgroundSampler.luminance2d = (float*)crana_stack_alloc(&mainRenderContext.stack, sizeof(float) * backgroundSampler.width * backgroundSampler.height);

		backgroundSampler.cdf1d = (float*)crana_stack_alloc(&mainRenderContext.stack, sizeof(float) * backgroundSampler.height);
		backgroundSampler.sum1d = (float*)crana_stack_alloc(&mainRenderContext.stack, sizeof(float) * backgroundSampler.height);

		float ysum = 0.0f;
		for (int y = 0; y < backgroundSampler.height; y++)
		{
			float xsum = 0.0f;
			for (int x = 0; x < backgroundSampler.width; x++)
			{
				int index = (y * backgroundSampler.width) + x;

				float* cran_restrict pixel = &backgroundSampler.image[index * backgroundSampler.stride];
				float luminance = rgb_to_luminance(pixel[0], pixel[1], pixel[2]);

				xsum += luminance;
				backgroundSampler.luminance2d[index] = luminance;
				backgroundSampler.cdf2d[index] = xsum;
			}

			for (int x = 0; x < backgroundSampler.width; x++)
			{
				int index = (y * backgroundSampler.width) + x;
				backgroundSampler.cdf2d[index] = backgroundSampler.cdf2d[index] * cf_rcp(xsum);
			}

			ysum += xsum;
			backgroundSampler.cdf1d[y] = ysum;
			backgroundSampler.sum1d[y] = xsum;
		}

		for (int y = 0; y < backgroundSampler.height; y++)
		{
			backgroundSampler.cdf1d[y] = backgroundSampler.cdf1d[y]  * cf_rcp(ysum);
		}

		backgroundSampler.sumTotal = ysum;
	}

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

	mainRenderData.origin = (cv3){ 0.0f, -3.5f, 0.0f };
	mainRenderData.forward = (cv3){ .x = 0.0f,.y = 1.0f,.z = 0.0f };
	mainRenderData.right = (cv3){ .x = 1.0f,.y = 0.0f,.z = 0.0f };
	mainRenderData.up = (cv3){ .x = 0.0f,.y = 0.0f,.z = 1.0f };

	float* cran_restrict hdrImage = crana_stack_alloc(&mainRenderContext.stack, mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride * sizeof(float));

	uint64_t renderStartTime = cranpl_timestamp_micro();

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

	mainRenderContext.renderStats.renderTime = (cranpl_timestamp_micro() - renderStartTime);

	// Image Space Effects
	bool enableImageSpace = true;
	if(enableImageSpace)
	{
		uint64_t imageSpaceStartTime = cranpl_timestamp_micro();
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
		mainRenderContext.renderStats.imageSpaceTime = cranpl_timestamp_micro() - imageSpaceStartTime;
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
		crana_stack_free(&mainRenderContext.scratchStack, mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride);
	}

	for (uint32_t i = 0; i < cranpl_get_core_count(); i++)
	{
		merge_render_stats(&mainRenderContext.renderStats, &threadContexts[i].context.renderStats);
	}

	// Print stats
	{
		system("cls");
		printf("Total Time: %f\n", micro_to_seconds(mainRenderContext.renderStats.totalTime));
		printf("\tScene Generation Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.sceneGenerationTime), (float)mainRenderContext.renderStats.sceneGenerationTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
		printf("\tRender Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.renderTime), (float)mainRenderContext.renderStats.renderTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
		printf("----------\n");
		printf("Accumulated Threading Data\n");
		printf("\t\tIntersection Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.intersectionTime), (float)mainRenderContext.renderStats.intersectionTime / (float)mainRenderContext.renderStats.renderTime * 100.0f);
		printf("\t\t\tBVH Traversal Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.bvhTraversalTime), (float)mainRenderContext.renderStats.bvhTraversalTime / (float)mainRenderContext.renderStats.intersectionTime * 100.0f);
		printf("\t\t\t\tBVH Tests: %" PRIu64 "\n", mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount);
		printf("\t\t\t\t\tBVH Hits: %" PRIu64 "[%.2f%%]\n", mainRenderContext.renderStats.bvhHitCount, (float)mainRenderContext.renderStats.bvhHitCount/(float)(mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount) * 100.0f);
		printf("\t\t\t\t\t\tBVH Leaf Hits: %" PRIu64 "[%.2f%%]\n", mainRenderContext.renderStats.bvhLeafHitCount, (float)mainRenderContext.renderStats.bvhLeafHitCount/(float)mainRenderContext.renderStats.bvhHitCount * 100.0f);
		printf("\t\t\t\t\tBVH Misses: %" PRIu64 "[%.2f%%]\n", mainRenderContext.renderStats.bvhMissCount, (float)mainRenderContext.renderStats.bvhMissCount/(float)(mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount) * 100.0f);
		printf("\t\tSkybox Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.skyboxTime), (float)mainRenderContext.renderStats.skyboxTime / (float)mainRenderContext.renderStats.renderTime * 100.0f);
		printf("----------\n");
		printf("\tImage Space Time: %f [%.2f%%]\n", micro_to_seconds(mainRenderContext.renderStats.imageSpaceTime), (float)mainRenderContext.renderStats.imageSpaceTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
		printf("\n");
		printf("MRays/seconds: %f\n", (float)mainRenderContext.renderStats.rayCount / micro_to_seconds(mainRenderContext.renderStats.renderTime) / 1000000.0f);
		printf("Rays Fired: %" PRIu64 "\n", mainRenderContext.renderStats.rayCount);
		printf("\tCamera Rays Fired: %" PRIu64 " [%.2f%%]\n", mainRenderContext.renderStats.primaryRayCount, (float)mainRenderContext.renderStats.primaryRayCount / (float)mainRenderContext.renderStats.rayCount * 100.0f);
		printf("\tBounce Rays Fired: %" PRIu64 " [%.2f%%]\n", mainRenderContext.renderStats.rayCount - mainRenderContext.renderStats.primaryRayCount, (float)(mainRenderContext.renderStats.rayCount - mainRenderContext.renderStats.primaryRayCount) / (float)mainRenderContext.renderStats.rayCount * 100.0f);
		printf("\n");
		printf("BVH\n");
		printf("\tBVH Node Count: %" PRIu64 "\n", mainRenderContext.renderStats.bvhNodeCount);
		printf("Memory\n");
		printf("\tMain Stack\n");
		printf("\t\tStack Size: %" PRIu64 "\n", mainRenderContext.stack.size);
		printf("\t\tFinal Stack Top: %" PRIu64 " [%.2f%%]\n", mainRenderContext.stack.top, (float)mainRenderContext.stack.top/(float)mainRenderContext.stack.size*100.0f);
		printf("\tScratch Stack\n");
		printf("\t\tStack Size: %" PRIu64 "\n", mainRenderContext.scratchStack.size);
		printf("\t\tStack Top: %" PRIu64 " [%.2f%%]\n", mainRenderContext.scratchStack.top, (float)mainRenderContext.scratchStack.top/(float)mainRenderContext.scratchStack.size*100.0f);
	}

	// Not worrying about individual memory cleanup, stack allocator is cleaned up in one swoop anyways.
	free(mainRenderContext.stack.mem);
	free(mainRenderContext.scratchStack.mem);

	stbi_image_free(checkerboardSampler.image);
	stbi_image_free(backgroundSampler.image);

	cranpr_end("cranberray","main");
	cranpr_flush_thread_buffer();
	cranpr_write_to_file("cranberray_profile.json");

	cranpr_terminate();
	return 0;
}
