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

#include "stb_image.h"
#include "cranberry_platform.h"
#include "cranberry_loader.h"
#include "cranberry_math.h"

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

typedef struct
{
	int32_t randomSeed;
	uint32_t depth;

	render_stats_t renderStats;
} render_context_t;

static float micro_to_seconds(uint64_t time)
{
	return (float)time / 1000000.0f;
}

static float random01f(int32_t* seed)
{
	assert(*seed != 0);

	// http://www.iquilezles.org/www/articles/sfrand/sfrand.htm
	union
	{
		float f;
		int32_t i;
	} res;

	*seed *= 16807;
	uint32_t randomNumber = *seed;
	res.i = ((randomNumber>>9) | 0x3f800000);
	return res.f - 1.0f;
}

static uint32_t randomRange(int32_t* seed, uint32_t min, uint32_t max)
{
	float result = random01f(seed) * (float)(max - min);
	return (uint32_t)result + min;
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

static cv3 sphere_random(int32_t* seed)
{
	cv3 p;
	do
	{
		p = cv3_mulf((cv3) { random01f(seed)-0.5f, random01f(seed)-0.5f, random01f(seed)-0.5f }, 2.0f);
	} while (cv3_dot(p, p) <= 1.0f);

	return p;
}

static cv3 sample_hdr(cv3 v, float* image, int32_t imgWidth, int32_t imgHeight, int32_t imgStride)
{
	float azimuth, theta;
	cv3_to_spherical(v, &azimuth, &theta);

	int32_t readY = (int32_t)(fminf(theta * cran_rpi, 0.999f) * (float)imgHeight);
	int32_t readX = (int32_t)(fminf(azimuth * cran_rtao, 0.999f) * (float)imgWidth);
	int32_t readIndex = (readY * imgWidth + readX) * imgStride;

	// TODO: Don't just clamp to 1, remap our image later on. For now this is fine.
	cv3 color;
	color.x = image[readIndex + 0];
	color.y = image[readIndex + 1];
	color.z = image[readIndex + 2];
	return color;
}

float light_attenuation(cv3 l, cv3 r)
{
	return cf_rcp(1.0f + cv3_length(cv3_sub(l, r)));
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
	cv3 color;
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
	int(*sortFuncs[3])(const void* cran_restrict l, const void* cran_restrict r) = { index_aabb_sort_min_x, index_aabb_sort_min_y, index_aabb_sort_min_z };

	struct
	{
		index_aabb_pair_t* start;
		uint32_t count;
		uint32_t* parentIndex;
	} bvhWorkgroup[10000];
	uint32_t workgroupQueueEnd = 1;

	bvhWorkgroup[0].start = leafs;
	bvhWorkgroup[0].count = leafCount;
	bvhWorkgroup[0].parentIndex = NULL;

	uint32_t bvhWriteIter = 0;
	bvh_t builtBVH =
	{
		.bounds = malloc(sizeof(caabb) * 100000), // TODO: Actually get an accurate size?/Release this memory.
		.jumps = malloc(sizeof(bvh_jump_t) * 100000),
	};

	for (uint32_t workgroupIter = 0; workgroupIter != workgroupQueueEnd; workgroupIter = (workgroupIter + 1) % 10000) // TODO: constant for workgroup size
	{
		index_aabb_pair_t* start = bvhWorkgroup[workgroupIter].start;
		uint32_t count = bvhWorkgroup[workgroupIter].count;

		if (bvhWorkgroup[workgroupIter].parentIndex != NULL)
		{
			*(bvhWorkgroup[workgroupIter].parentIndex) = bvhWriteIter;
		}

		caabb bounds = start[0].bound;
		for (uint32_t i = 1; i < count; i++)
		{
			bounds.min = cv3_min(start[i].bound.min, bounds.min);
			bounds.max = cv3_max(start[i].bound.max, bounds.max);
		}

		bool isLeaf = (count == 1);
		builtBVH.bounds[bvhWriteIter] = bounds;
		builtBVH.jumps[bvhWriteIter] = (bvh_jump_t)
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
			bvhWorkgroup[workgroupQueueEnd].parentIndex = &builtBVH.jumps[bvhWriteIter].indices.jumps.left;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % 10000; // TODO: Constant for workgroup size
			assert(workgroupQueueEnd != workgroupIter);

			bvhWorkgroup[workgroupQueueEnd].start = start + centerIndex;
			bvhWorkgroup[workgroupQueueEnd].count = count - centerIndex;
			bvhWorkgroup[workgroupQueueEnd].parentIndex = &builtBVH.jumps[bvhWriteIter].indices.jumps.right;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % 10000; // TODO: Constant for workgroup size
			assert(workgroupQueueEnd != workgroupIter);
		}

		bvhWriteIter++;
		assert(bvhWriteIter < 100000);
	}

	context->renderStats.bvhNodeCount = bvhWriteIter;
	return builtBVH;
}

static uint32_t traverse_bvh(render_stats_t* renderStats, bvh_t const* bvh, cv3 rayO, cv3 rayD, float rayMin, float rayMax, uint32_t* candidates, uint32_t maxInstances)
{
	uint64_t traversalStartTime = cranpl_timestamp_micro();

	uint32_t iter = 0;

	// TODO: custom allocator
	uint32_t* testQueue = malloc(sizeof(uint32_t) * 10000); // TODO: Currently we just allow 1000 stack pushes. Fix this!
	uint32_t testQueueSize = 1;
	uint32_t testQueueIter = 0;

	testQueue[0] = 0;
	while ((int32_t)testQueueSize - (int32_t)testQueueIter > 0)
	{
		cv3l boundMins = { 0 };
		cv3l boundMaxs = { 0 };

		uint32_t activeLaneCount = min(testQueueSize - testQueueIter, cran_lane_count);
		for (uint32_t i = 0; i < activeLaneCount; i++)
		{
			uint32_t nodeIndex = testQueue[testQueueIter + i];
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
					uint32_t nodeIndex = testQueue[testQueueIter + i];

					renderStats->bvhHitCount++;

					// All our leaves are packed at the end of the 
					bool isLeaf = bvh->jumps[nodeIndex].isLeaf;
					if (isLeaf)
					{
						renderStats->bvhLeafHitCount++;

						candidates[iter] = bvh->jumps[nodeIndex].indices.index;
						iter++;
						assert(iter <= maxInstances);
					}
					else
					{
						testQueue[testQueueSize] = bvh->jumps[nodeIndex].indices.jumps.left;
						testQueueSize++;
						testQueue[testQueueSize] = bvh->jumps[nodeIndex].indices.jumps.right;
						testQueueSize++;

						assert(testQueueSize < 10000);
					}
				}
				else
				{
					renderStats->bvhMissCount++;
				}
			}
		}
		else
		{
			renderStats->bvhMissCount += activeLaneCount;
		}


		testQueueIter += activeLaneCount;
	}

	free(testQueue);
	renderStats->bvhTraversalTime += cranpl_timestamp_micro() - traversalStartTime;
	return iter;
}

static void generate_scene(render_context_t* context, ray_scene_t* scene)
{
	uint64_t startTime = cranpl_timestamp_micro();

	static material_lambert_t lamberts[3] = { {.color = { 0.8f, 0.9f, 1.0f } },  {.color = { 0.1f, 0.1f, 0.1f } }, {.color = {1.0f, 1.0f, 1.0f} } };
	static material_mirror_t mirrors[2] = { {.color = { 0.8f, 1.0f, 1.0f } }, { .color = { 0.1f, 0.8f, 0.5f } } };

	static material_index_t materialIndices[] = 
	{
		{.dataIndex = 1,.typeIndex = material_lambert },
		{.dataIndex = 0,.typeIndex = material_mirror },
		{.dataIndex = 2,.typeIndex = material_lambert },
		{.dataIndex = 0,.typeIndex = material_lambert },
	};

	static instance_t instances[1];
	static mesh_t mesh;

	// Mesh
	{
		mesh.data = cranl_obj_load("mori_knob.obj", cranl_flip_yz);
		uint32_t meshLeafCount = mesh.data.faces.count;
		index_aabb_pair_t* leafs = malloc(sizeof(index_aabb_pair_t) * meshLeafCount);

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
		free(leafs);
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
		index_aabb_pair_t* leafs = malloc(sizeof(index_aabb_pair_t) * leafCount);
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
		free(leafs);
	}

	context->renderStats.sceneGenerationTime = cranpl_timestamp_micro() - startTime;
}

typedef struct
{
	cv3 light;
	cv3 surface;
} ray_hit_t;

int backgroundWidth, backgroundHeight, backgroundStride;
float* cran_restrict background;
static ray_hit_t cast_scene(render_context_t* context, ray_scene_t const* scene, cv3 rayO, cv3 rayD)
{
	context->depth++;
	if (context->depth >= renderConfig.maxDepth)
	{
		context->depth--;
		return (ray_hit_t) { 0 };
	}

	context->renderStats.rayCount++;

	const float NoRayIntersection = FLT_MAX;

	struct
	{
		float distance;
		cv3 normal;
		material_index_t materialIndex;
	} closestHitInfo = { 0 };
	closestHitInfo.distance = NoRayIntersection;

	// TODO: Traverse our BVH instead - Done
	// TODO: Gather list of candidate BVHs - Done
	// TODO: Sort list of instances by type
	// TODO: find closest intersection in candidates
	// TODO: ?!?!?
	uint64_t intersectionStartTime = cranpl_timestamp_micro();

	uint32_t candidates[1000]; // TODO: Max candidates of 1000?
	uint32_t candidateCount = traverse_bvh(&context->renderStats, &scene->bvh, rayO, rayD, 0.0f, NoRayIntersection, candidates, 1000);
	for (uint32_t i = 0; i < candidateCount; i++)
	{
		uint32_t candidateIndex = candidates[i];

		cv3 instancePos = scene->instances.data[candidateIndex].pos;
		uint32_t renderableIndex = scene->instances.data[candidateIndex].renderableIndex;

		cv3 rayInstanceO = cv3_sub(rayO, instancePos);

		float intersectionDistance = 0.0f;

		mesh_t* mesh = &scene->renderables[renderableIndex];
		material_index_t* materialIndices = mesh->materialIndices;

		uint32_t meshCandidates[1000]; // TODO:
		uint32_t meshCandidateCount = traverse_bvh(&context->renderStats, &mesh->bvh, rayO, rayD, 0.0f, NoRayIntersection, meshCandidates, 1000);
		for (uint32_t faceCandidate = 0; faceCandidate < meshCandidateCount; faceCandidate++)
		{
			// TODO: Lanes
			uint32_t faceIndex = meshCandidates[faceCandidate];

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

				uint32_t normalIndexA = mesh->data.faces.normalIndices[faceIndex * 3 + 0];
				uint32_t normalIndexB = mesh->data.faces.normalIndices[faceIndex * 3 + 1];
				uint32_t normalIndexC = mesh->data.faces.normalIndices[faceIndex * 3 + 2];
				cv3 normalA, normalB, normalC;
				memcpy(&normalA, mesh->data.normals.data + normalIndexA * 3, sizeof(cv3));
				memcpy(&normalB, mesh->data.normals.data + normalIndexB * 3, sizeof(cv3));
				memcpy(&normalC, mesh->data.normals.data + normalIndexC * 3, sizeof(cv3));

				closestHitInfo.normal = cv3_add(cv3_add(cv3_mulf(normalA, u), cv3_mulf(normalB, v)), cv3_mulf(normalC, w));
			}
		}
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
				.viewDir = rayD
			});
		context->depth--;
		return (ray_hit_t)
		{
			.light = light,
			.surface = intersectionPoint
		};
	}

	cv3 skybox = cv3_lerp((cv3) {1.0f, 1.0f, 1.0f}, (cv3) {0.5f, 0.7f, 1.0f}, rayD.z * 0.5f + 0.5f);
	//uint64_t skyboxStartTime = cranpl_timestamp_micro();
	//cv3 skybox = sample_hdr(rayD, background, backgroundWidth, backgroundHeight, backgroundStride);
	//context->renderStats.skyboxTime += cranpl_timestamp_micro() - skyboxStartTime;

	context->depth--;
	return (ray_hit_t)
	{
		.light = cv3_mulf(skybox, 10000.0f),
		.surface = cv3_add(rayO, cv3_mulf(rayD, 1000.0f))
	};
}

// TODO: This recast bias is simply to avoid re-intersecting with our object when casting.
// Do we want to handle this some other way?
const float ReCastBias = 0.0001f;
static cv3 shader_lambert(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	material_lambert_t lambertData = ((const material_lambert_t* cran_restrict)materialData)[materialIndex];

	cv3 castDir = cv3_add(inputs.normal, sphere_random(&context->randomSeed));
	// TODO: Consider iteration instead of recursion
	ray_hit_t result = cast_scene(context, scene, cv3_add(inputs.surface, cv3_mulf(inputs.normal, ReCastBias)), castDir);

	float lambertCosine = fmaxf(0.0f, cv3_dot(cv3_normalized(castDir), inputs.normal));
	cv3 sceneCast = cv3_mulf(cv3_mulf(result.light, lambertCosine), light_attenuation(result.surface, inputs.surface));

	return cv3_mul(sceneCast, cv3_mulf(lambertData.color, cran_rpi));
}

static cv3 shader_mirror(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t const* scene, shader_inputs_t inputs)
{
	material_mirror_t mirrorData = ((const material_mirror_t* cran_restrict)materialData)[materialIndex];

	cv3 castDir = cv3_reflect(inputs.viewDir, inputs.normal);
	ray_hit_t result = cast_scene(context, scene, cv3_add(inputs.surface, cv3_mulf(inputs.normal, ReCastBias)), castDir);

	float lambertCosine = fmaxf(0.0f, cv3_dot(cv3_normalized(castDir), inputs.normal));
	cv3 sceneCast = cv3_mulf(cv3_mulf(result.light, lambertCosine), light_attenuation(result.surface, inputs.surface));

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
					// TODO: Do we want to scale for average in the loop or outside the loop?
					// With too many SPP, the sceneColor might get too significant.

					ray_hit_t hit = cast_scene(renderContext, &renderData->scene, renderData->origin, rayDir);
					sceneColor = cv3_add(sceneColor, cv3_mulf(hit.light, light_attenuation(hit.surface, renderData->origin)));
				}
				sceneColor = cv3_mulf(sceneColor, cf_rcp((float)renderConfig.samplesPerPixel));

				int32_t imgIdx = ((y + renderData->halfImgHeight) * renderData->imgWidth + (x + renderData->halfImgWidth)) * renderData->imgStride;
				threadContext->hdrOutput[imgIdx + 0] = sceneColor.x;
				threadContext->hdrOutput[imgIdx + 1] = sceneColor.y;
				threadContext->hdrOutput[imgIdx + 2] = sceneColor.z;
				threadContext->hdrOutput[imgIdx + 3] = 1.0f;
			}
		}
	}
}

int main()
{
	renderConfig = (render_config_t)
	{
		.maxDepth = 99,
		.samplesPerPixel = 1,
		.renderWidth = 1024,
		.renderHeight = 768
	};

	static render_data_t mainRenderData;
	uint64_t startTime = cranpl_timestamp_micro();

	background = stbi_loadf("background_4k.hdr", &backgroundWidth, &backgroundHeight, &backgroundStride, 0);

	render_context_t mainRenderContext =
	{
		.randomSeed = 143324
	};
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

	mainRenderData.origin = (cv3){ 0.0f, -2.0f, 0.0f };
	mainRenderData.forward = (cv3){ .x = 0.0f,.y = 1.0f,.z = 0.0f };
	mainRenderData.right = (cv3){ .x = 1.0f,.y = 0.0f,.z = 0.0f };
	mainRenderData.up = (cv3){ .x = 0.0f,.y = 0.0f,.z = 1.0f };

	float* cran_restrict hdrImage = malloc(mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride * sizeof(float));

	uint64_t renderStartTime = cranpl_timestamp_micro();

	static render_queue_t mainRenderQueue = { 0 };
	{
		mainRenderQueue.chunks = malloc(sizeof(render_chunk_t) * mainRenderData.imgHeight);
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

	thread_context_t* threadContexts = malloc(sizeof(thread_context_t) * cranpl_get_core_count());
	void** threadHandles = malloc(sizeof(void*) * cranpl_get_core_count() - 1);
	for (uint32_t i = 0; i < cranpl_get_core_count() - 1; i++)
	{
		threadContexts[i] = (thread_context_t)
		{
			.renderData = &mainRenderData,
			.renderQueue = &mainRenderQueue,
			.context = 
			{
				.randomSeed = i + 321
			},
			.hdrOutput = hdrImage
		};

		threadHandles[i] = cranpl_create_thread(&render_scene_async, &threadContexts[i]);
	}

	// Start a render on our main thread as well.
	threadContexts[cranpl_get_core_count() - 1] = (thread_context_t)
	{
		.renderData = &mainRenderData,
		.renderQueue = &mainRenderQueue,
		.context = 
		{
			.randomSeed = cranpl_get_core_count() - 1
		},
		.hdrOutput = hdrImage
	};
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
		uint8_t* cran_restrict bitmap = malloc(mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride);
		for (int32_t i = 0; i < mainRenderData.imgWidth * mainRenderData.imgHeight * mainRenderData.imgStride; i+=mainRenderData.imgStride)
		{
			bitmap[i + 0] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 2]));
			bitmap[i + 1] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 1]));
			bitmap[i + 2] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 0]));
			bitmap[i + 3] = (uint8_t)(255.99f * hdrImage[i + 3]);
		}

		cranpl_write_bmp("render.bmp", bitmap, mainRenderData.imgWidth, mainRenderData.imgHeight);
		system("render.bmp");
		free(bitmap);
	}

	free(threadContexts);
	free(threadHandles);
	free(mainRenderQueue.chunks);
	free(hdrImage);
	stbi_image_free(background);

	for (uint32_t i = 0; i < cranpl_get_core_count(); i++)
	{
		merge_render_stats(&mainRenderContext.renderStats, &threadContexts[i].context.renderStats);
	}

	// Print stats
	{
		system("cls");
		printf("Total Time: %f\n", micro_to_seconds(mainRenderContext.renderStats.totalTime));
		printf("\tScene Generation Time: %f [%f%%]\n", micro_to_seconds(mainRenderContext.renderStats.sceneGenerationTime), (float)mainRenderContext.renderStats.sceneGenerationTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
		printf("\tRender Time: %f [%f%%]\n", micro_to_seconds(mainRenderContext.renderStats.renderTime), (float)mainRenderContext.renderStats.renderTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
		printf("----------\n");
		printf("Accumulated Threading Data\n");
		printf("\t\tIntersection Time: %f [%f%%]\n", micro_to_seconds(mainRenderContext.renderStats.intersectionTime), (float)mainRenderContext.renderStats.intersectionTime / (float)mainRenderContext.renderStats.renderTime * 100.0f);
		printf("\t\t\tBVH Traversal Time: %f [%f%%]\n", micro_to_seconds(mainRenderContext.renderStats.bvhTraversalTime), (float)mainRenderContext.renderStats.bvhTraversalTime / (float)mainRenderContext.renderStats.intersectionTime * 100.0f);
		printf("\t\t\t\tBVH Tests: %" PRIu64 "\n", mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount);
		printf("\t\t\t\t\tBVH Hits: %" PRIu64 "[%f%%]\n", mainRenderContext.renderStats.bvhHitCount, (float)mainRenderContext.renderStats.bvhHitCount/(float)(mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount) * 100.0f);
		printf("\t\t\t\t\t\tBVH Leaf Hits: %" PRIu64 "[%f%%]\n", mainRenderContext.renderStats.bvhLeafHitCount, (float)mainRenderContext.renderStats.bvhLeafHitCount/(float)mainRenderContext.renderStats.bvhHitCount * 100.0f);
		printf("\t\t\t\t\tBVH Misses: %" PRIu64 "[%f%%]\n", mainRenderContext.renderStats.bvhMissCount, (float)mainRenderContext.renderStats.bvhMissCount/(float)(mainRenderContext.renderStats.bvhHitCount + mainRenderContext.renderStats.bvhMissCount) * 100.0f);
		printf("\t\tSkybox Time: %f [%f%%]\n", micro_to_seconds(mainRenderContext.renderStats.skyboxTime), (float)mainRenderContext.renderStats.skyboxTime / (float)mainRenderContext.renderStats.renderTime * 100.0f);
		printf("----------\n");
		printf("\tImage Space Time: %f [%f%%]\n", micro_to_seconds(mainRenderContext.renderStats.imageSpaceTime), (float)mainRenderContext.renderStats.imageSpaceTime / (float)mainRenderContext.renderStats.totalTime * 100.0f);
		printf("\n");
		printf("MRays/seconds: %f\n", (float)mainRenderContext.renderStats.rayCount / micro_to_seconds(mainRenderContext.renderStats.renderTime) / 1000000.0f);
		printf("Rays Fired: %" PRIu64 "\n", mainRenderContext.renderStats.rayCount);
		printf("\tCamera Rays Fired: %" PRIu64 " [%f%%]\n", mainRenderContext.renderStats.primaryRayCount, (float)mainRenderContext.renderStats.primaryRayCount / (float)mainRenderContext.renderStats.rayCount * 100.0f);
		printf("\tBounce Rays Fired: %" PRIu64 " [%f%%]\n", mainRenderContext.renderStats.rayCount - mainRenderContext.renderStats.primaryRayCount, (float)(mainRenderContext.renderStats.rayCount - mainRenderContext.renderStats.primaryRayCount) / (float)mainRenderContext.renderStats.rayCount * 100.0f);
		printf("\n");
		printf("BVH\n");
		printf("\tBVH Node Count: %" PRIu64 "\n", mainRenderContext.renderStats.bvhNodeCount);

		system("pause");
	}
	return 0;
}
