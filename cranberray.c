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
#include <immintrin.h>

#include "stb_image.h"
#include "cranberry_platform.h"

struct
{
	uint64_t rayCount; // TODO: only reference through atomics
	uint64_t primaryRayCount;
	uint64_t totalTime;
	uint64_t sceneGenerationTime;
	uint64_t renderTime;
	uint64_t intersectionTime;
	uint64_t bvhTraversalTime;
	uint64_t skyboxTime;
	uint64_t imageSpaceTime;
} renderStats;

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
	uint32_t randomSeed;
} render_context_t;

static float micro_to_seconds(uint64_t time)
{
	return (float)time / 1000000.0f;
}

// Forward decls
void* memset(void* dst, int val, size_t size);
void* memcpy(void* dst, void const* src, size_t size);

#define _PI_VAL 3.14159265358979323846264338327f
const float PI = _PI_VAL;
const float TAO = _PI_VAL * 2.0f;
const float RPI = 1.0f / _PI_VAL;
const float RTAO = 1.0f / (_PI_VAL * 2.0f);

// float math
static float rcp(float f)
{
	// return 1.0f / f;

	union
	{
		__m128 sse;
		float f[4];
	} conv;
	conv.sse = _mm_rcp_ss(_mm_set_ss(f));
	return conv.f[0];
}

static float rsqrt(float f)
{
	// return 1.0f / sqrtf(f);

	union
	{
		__m128 sse;
		float f[4];
	} conv;
	conv.sse = _mm_rsqrt_ss(_mm_set_ss(f));
	return conv.f[0];
}

static bool quadratic(float a, float b, float c, float* restrict out1, float* restrict out2)
{
	assert(out1 != out2);

	// TODO: Replace with more numerically robust version.
	float determinant = b * b - 4.0f * a * c;
	if (determinant < 0.0f)
	{
		return false;
	}

	float d = sqrtf(determinant);
	float e = rcp(2.0f * a);

	*out1 = (-b - d) * e;
	*out2 = (-b + d) * e;
	return true;
}

float rand01f( uint32_t *seed )
{
	// http://www.iquilezles.org/www/articles/sfrand/sfrand.htm
	assert(seed[0] != 0);
	union
	{
		float f;
		uint32_t i;
	} res;

	seed[0] *= 16807;
	res.i = ((((unsigned int)seed[0])>>9 ) | 0x3f800000);
	return res.f - 1.0f;
}

static float rgb_to_luminance(float r, float g, float b)
{
	return (0.2126f*r + 0.7152f*g + 0.0722f*b);
}

// vector math
typedef struct
{
	float x, y, z;
} vec3;

static vec3 vec3_mulf(vec3 l, float r)
{
	return (vec3) { .x = l.x * r, .y = l.y * r, .z = l.z * r };
}

static vec3 vec3_add(vec3 l, vec3 r)
{
	return (vec3) {.x = l.x + r.x, .y = l.y + r.y, .z = l.z + r.z};
}

static vec3 vec3_addf(vec3 l, float r)
{
	return (vec3) {.x = l.x + r, .y = l.y + r, .z = l.z + r};
}

static vec3 vec3_sub(vec3 l, vec3 r)
{
	return (vec3) {.x = l.x - r.x, .y = l.y - r.y, .z = l.z - r.z};
}

static vec3 vec3_subf(vec3 l, float r)
{
	return (vec3) {.x = l.x - r, .y = l.y - r, .z = l.z - r};
}

static vec3 vec3_mul(vec3 l, vec3 r)
{
	return (vec3) {.x = l.x * r.x, .y = l.y * r.y, .z = l.z * r.z};
}

static float vec3_dot(vec3 l, vec3 r)
{
	return l.x * r.x + l.y * r.y + l.z * r.z;
}

static vec3 vec3_cross(vec3 l, vec3 r)
{
	return (vec3)
	{
		.x = l.y*l.z - l.z*r.y,
		.y = l.z*r.x - l.x*r.z,
		.z = l.x*r.y - l.y*r.x
	};
}

static vec3 vec3_lerp(vec3 l, vec3 r, float t)
{
	return vec3_add(vec3_mulf(l, 1.0f - t), vec3_mulf(r, t));
}

static float vec3_length(vec3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static float vec3_rlength(vec3 v)
{
	return rsqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static vec3 vec3_normalized(vec3 v)
{
	return vec3_mulf(v, vec3_rlength(v));
}

static vec3 vec3_min(vec3 v, vec3 m)
{
	return (vec3){fminf(v.x, m.x), fminf(v.y, m.y), fminf(v.z, m.z)};
}

static vec3 vec3_max(vec3 v, vec3 m)
{
	return (vec3){fmaxf(v.x, m.x), fmaxf(v.y, m.y), fmaxf(v.z, m.z)};
}

static vec3 vec3_rcp(vec3 v)
{
	// TODO: consider SSE
	return (vec3) { rcp(v.x), rcp(v.y), rcp(v.z) };
}

typedef struct
{
	vec3 i, j, k;
} mat3;

static mat3 mat3_from_basis(vec3 i, vec3 j, vec3 k)
{
	assert(vec3_length(i) < 1.01f && vec3_length(i) > 0.99f);
	assert(vec3_length(j) < 1.01f && vec3_length(j) > 0.99f);
	assert(vec3_length(k) < 1.01f && vec3_length(k) > 0.99f);
	return (mat3)
	{
		.i = i,
		.j = j,
		.k = k
	};
}

static mat3 mat3_basis_from_normal(vec3 n)
{
	// Frisvad ONB from https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
	// revised from Pixar https://graphics.pixar.com/library/OrthonormalB/paper.pdf#page=2&zoom=auto,-233,561
	float sign = copysignf(1.0f, n.z);
	float a = -rcp(sign + n.z);
	float b = -n.x*n.y*a;
	vec3 i = (vec3) { 1.0f + sign * n.x*n.x*a, sign * b, -sign * n.x };
	vec3 j = (vec3) { b, sign + n.y*n.y*a, -n.y };

	return mat3_from_basis(i, j, n);
}

static vec3 mat3_mul_vec3(mat3 m, vec3 v)
{
	vec3 vx = (vec3) { v.x, v.x, v.x };
	vec3 vy = (vec3) { v.y, v.y, v.y };
	vec3 vz = (vec3) { v.z, v.z, v.z };

	vec3 rx = vec3_mul(vx, m.i);
	vec3 ry = vec3_mul(vy, m.j);
	vec3 rz = vec3_mul(vz, m.k);

	return vec3_add(vec3_add(rx, ry), rz);
}

static vec3 mat3_rotate_vec3(mat3 m, vec3 v)
{
	return mat3_mul_vec3(m, v);
}

static bool sphere_does_ray_intersect(vec3 rayO, vec3 rayD, vec3 sphereO, float sphereR)
{
	vec3 sphereRaySpace = vec3_sub(sphereO, rayO);
	float projectedDistance = vec3_dot(sphereRaySpace, rayD);
	float distanceToRaySqr = vec3_dot(sphereRaySpace,sphereRaySpace) - projectedDistance * projectedDistance;
	return (distanceToRaySqr < sphereR * sphereR);
}

static float sphere_ray_intersection(vec3 rayO, vec3 rayD, float rayMin, float rayMax, vec3 sphereO, float sphereR)
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
	vec3 raySphereSpace = vec3_sub(rayO, sphereO);
	float a = rayD.x * rayD.x + rayD.y * rayD.y + rayD.z * rayD.z;
	float b = 2.0f * rayD.x * raySphereSpace.x + 2.0f * rayD.y * raySphereSpace.y + 2.0f * rayD.z * raySphereSpace.z;
	float c = raySphereSpace.x * raySphereSpace.x + raySphereSpace.y * raySphereSpace.y + raySphereSpace.z * raySphereSpace.z - sphereR * sphereR;

	float d1, d2;
	if (!quadratic(a, b, c, &d1, &d2))
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

static vec3 sphere_random(uint32_t* seed)
{
	vec3 p;
	do
	{
		p = vec3_mulf((vec3) { rand01f(seed)-0.5f, rand01f(seed)-0.5f, rand01f(seed)-0.5f }, 2.0f);
	} while (vec3_dot(p, p) <= 1.0f);

	return p;
}

static vec3 vec3_reflect(vec3 i, vec3 n)
{
	return vec3_sub(i, vec3_mulf(n, 2.0f * vec3_dot(i, n)));
}

// a is between 0 and 2 PI
// t is between 0 and PI (0 being the bottom, PI being the top)
static void vec3_to_spherical(vec3 v, float* restrict a, float* restrict t)
{
	assert(a != t);

	float rlenght = vec3_rlength(v);
	float azimuth = atan2f(v.y, v.x);
	*a = (azimuth < 0.0f ? TAO + azimuth : azimuth);
	*t = acosf(v.z * rlenght);
}

// theta is between 0 and 2PI (horizontal plane)
// phi is between 0 and PI (vertical plane)
static vec3 vec3_from_spherical(float theta, float phi, float radius)
{
	return (vec3) { cosf(theta) * sinf(phi) * radius, sinf(theta) * sinf(phi) * radius, radius * cosf(phi) };
}

static vec3 sample_hdr(vec3 v, float* image, int32_t imgWidth, int32_t imgHeight, int32_t imgStride)
{
	float azimuth, theta;
	vec3_to_spherical(v, &azimuth, &theta);

	int32_t readY = (int32_t)(fminf(theta * RPI, 0.999f) * (float)imgHeight);
	int32_t readX = (int32_t)(fminf(azimuth * RTAO, 0.999f) * (float)imgWidth);
	int32_t readIndex = (readY * imgWidth + readX) * imgStride;

	// TODO: Don't just clamp to 1, remap our image later on. For now this is fine.
	vec3 color;
	color.x = image[readIndex + 0];
	color.y = image[readIndex + 1];
	color.z = image[readIndex + 2];
	return color;
}

typedef struct
{
	vec3 min;
	vec3 max;
} aabb;

static bool aabb_does_ray_intersect_sse(vec3 rayO, vec3 rayD, float rayMin, float rayMax, vec3 aabbMin, vec3 aabbMax)
{
	// Wasting 25% of vector space, do we want to do lanes? This is currently just easier.
	__m128 vrayO = _mm_loadu_ps(&rayO.x);
	__m128 vrayD = _mm_loadu_ps(&rayD.x);
	__m128 vmin = _mm_loadu_ps(&aabbMin.x);
	__m128 vmax = _mm_loadu_ps(&aabbMax.x);
	__m128 vrayMax = _mm_set_ps1(rayMax);
	__m128 vrayMin = _mm_set_ps1(rayMin);

	__m128 invD = _mm_rcp_ps(vrayD);
	__m128 t0s = _mm_mul_ps(_mm_sub_ps(vmin, vrayO), invD);
	__m128 t1s = _mm_mul_ps(_mm_sub_ps(vmax, vrayO), invD);

	__m128 tsmaller = _mm_min_ps(t0s, t1s);
	// Our fourth element is bad, we need to overwrite it
	tsmaller = _mm_shuffle_ps(tsmaller, tsmaller, _MM_SHUFFLE(2, 2, 1, 0));

	__m128 tbigger = _mm_max_ps(t0s, t1s);
	tbigger = _mm_shuffle_ps(tbigger, tbigger, _MM_SHUFFLE(2, 2, 1, 1));

	tsmaller = _mm_max_ps(tsmaller, _mm_shuffle_ps(tsmaller, tsmaller, _MM_SHUFFLE(2, 1, 0, 3)));
	tsmaller = _mm_max_ps(tsmaller, _mm_shuffle_ps(tsmaller, tsmaller, _MM_SHUFFLE(1, 0, 3, 2)));
	vrayMin = _mm_max_ps(vrayMin, tsmaller);

	tbigger = _mm_min_ps(tbigger, _mm_shuffle_ps(tbigger, tbigger, _MM_SHUFFLE(2, 1, 0, 3)));
	tbigger = _mm_min_ps(tbigger, _mm_shuffle_ps(tbigger, tbigger, _MM_SHUFFLE(1, 0, 3, 2)));
	vrayMax = _mm_min_ps(vrayMax, tbigger);

	return _mm_movemask_ps(_mm_cmplt_ps(vrayMin, vrayMax));
}

// Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
static bool aabb_does_ray_intersect(vec3 rayO, vec3 rayD, float rayMin, float rayMax, vec3 aabbMin, vec3 aabbMax)
{
	return aabb_does_ray_intersect_sse(rayO, rayD, rayMin, rayMax, aabbMin, aabbMax);

	/*vec3 invD = vec3_rcp(rayD);
	vec3 t0s = vec3_mul(vec3_sub(aabbMin, rayO), invD);
	vec3 t1s = vec3_mul(vec3_sub(aabbMax, rayO), invD);

	vec3 tsmaller = vec3_min(t0s, t1s);
	vec3 tbigger  = vec3_max(t0s, t1s);
 
	float tmin = fmaxf(rayMin, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
	float tmax = fminf(rayMax, fminf(tbigger.x, fminf(tbigger.y, tbigger.z)));
	return (tmin < tmax);*/
}

typedef enum
{
	material_lambert,
	material_metal,
	material_count
} material_type_e;

typedef struct
{
	vec3 color;
} material_lambert_t;

typedef struct
{
	uint8_t tempData; // TODO: Fill with real metalic data
} material_metal_t;

typedef struct
{
	uint16_t dataIndex;
	material_type_e typeIndex;
} material_index_t;
static_assert(material_count < 255, "Only 255 renderable types are supported.");

typedef enum
{
	renderable_sphere,
	renderable_count
} renderable_type_e;

typedef struct
{
	uint16_t dataIndex;
	renderable_type_e typeIndex;
} renderable_index_t;
static_assert(renderable_count < 255, "Only 255 renderable types are supported.");

typedef struct
{
	float rad;
} sphere_t;

typedef struct
{
	vec3 pos;
	material_index_t materialIndex;
	renderable_index_t renderableIndex;
} instance_t;

// TODO: Refine our scene description
typedef struct
{
	aabb bound;

	union
	{
		struct
		{
			uint32_t left;
			uint32_t right;
		} jumps;

		uint32_t instance;
	} indices;
	bool isLeaf;
} bvh_node_t;

typedef struct
{
	bvh_node_t* nodes;
} bvh_t;

typedef struct
{
	void* restrict materials[material_count];
	void* restrict renderables[renderable_count];

	struct
	{
		instance_t* data;
		uint32_t count;
	} instances;

	bvh_t bvh; // TODO: BVH!
} ray_scene_t;

typedef struct
{
	aabb bound;
	uint32_t instanceIndex;
} instance_aabb_pair_t;

static int instance_aabb_sort_min_x(const void* l, const void* r)
{
	const instance_aabb_pair_t* left = (const instance_aabb_pair_t*)l;
	const instance_aabb_pair_t* right = (const instance_aabb_pair_t*)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)(left->bound.min.x - right->bound.min.x);
}

static int instance_aabb_sort_min_y(const void* l, const void* r)
{
	const instance_aabb_pair_t* left = (const instance_aabb_pair_t*)l;
	const instance_aabb_pair_t* right = (const instance_aabb_pair_t*)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)(left->bound.min.y - right->bound.min.y);
}

static int instance_aabb_sort_min_z(const void* l, const void* r)
{
	const instance_aabb_pair_t* left = (const instance_aabb_pair_t*)l;
	const instance_aabb_pair_t* right = (const instance_aabb_pair_t*)r;

	// If left is greater than right, result is > 0 - left goes after right
	// If right is greater than left, result is < 0 - right goes after left
	// If equal, well they're equivalent
	return (int)(left->bound.min.z - right->bound.min.z);
}

static void build_bvh(render_context_t* context, ray_scene_t* scene)
{
	uint32_t leafCount = scene->instances.count;
	instance_aabb_pair_t* leafs = malloc(sizeof(instance_aabb_pair_t) * leafCount);
	for (uint32_t i = 0; i < leafCount; i++)
	{
		vec3 pos = scene->instances.data[i].pos;
		renderable_index_t renderableIndex = scene->instances.data[i].renderableIndex;

		leafs[i].instanceIndex = i;
		switch (renderableIndex.typeIndex)
		{
		case renderable_sphere:
		{
			float rad = ((sphere_t*)scene->renderables[renderableIndex.dataIndex])->rad;
			leafs[i].bound.min = vec3_subf(pos, rad);
			leafs[i].bound.max = vec3_addf(pos, rad);
		}
		break;
		default:
			assert(false);
			break;
		}
	}

	int(*sortFuncs[3])(const void* l, const void* r) = { instance_aabb_sort_min_x, instance_aabb_sort_min_y, instance_aabb_sort_min_z };

	struct
	{
		instance_aabb_pair_t* start;
		uint32_t count;
		uint32_t* parentIndex;
	} bvhWorkgroup[100];
	uint32_t workgroupQueueEnd = 1;

	bvhWorkgroup[0].start = leafs;
	bvhWorkgroup[0].count = leafCount;
	bvhWorkgroup[0].parentIndex = NULL;

	uint32_t bvhWriteIter = 0;
	static bvh_node_t builtBVH[1000]; // Alot-ish, just for now
	for (uint32_t workgroupIter = 0; workgroupIter != workgroupQueueEnd; workgroupIter = (workgroupIter + 1) % 100) // TODO: constant for workgroup size
	{
		instance_aabb_pair_t* start = bvhWorkgroup[workgroupIter].start;
		uint32_t count = bvhWorkgroup[workgroupIter].count;

		if (bvhWorkgroup[workgroupIter].parentIndex != NULL)
		{
			*(bvhWorkgroup[workgroupIter].parentIndex) = bvhWriteIter;
		}

		aabb bounds = start[0].bound;
		for (uint32_t i = 1; i < count; i++)
		{
			bounds.min = vec3_min(start[i].bound.min, bounds.min);
			bounds.max = vec3_max(start[i].bound.max, bounds.max);
		}

		bool isLeaf = (count == 1);
		builtBVH[bvhWriteIter] = (bvh_node_t)
		{
			.bound = bounds,
			.indices.instance = start[0].instanceIndex,
			.isLeaf = isLeaf,
		};

		if (!isLeaf)
		{
			// TODO: Since we're doing all the iteration work in the sort, maybe we could also do the partitioning in the sort?
			uint32_t axis = (uint32_t)(rand01f(&context->randomSeed)*3.0f);
			qsort(start, count, sizeof(instance_aabb_pair_t), sortFuncs[axis]);

			bvhWorkgroup[workgroupQueueEnd].start = start;
			bvhWorkgroup[workgroupQueueEnd].count = count / 2;
			bvhWorkgroup[workgroupQueueEnd].parentIndex = &builtBVH[bvhWriteIter].indices.jumps.left;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % 100; // TODO: Constant for workgroup size
			assert(workgroupQueueEnd != workgroupIter);

			bvhWorkgroup[workgroupQueueEnd].start = start + count / 2;
			bvhWorkgroup[workgroupQueueEnd].count = count - count / 2;
			bvhWorkgroup[workgroupQueueEnd].parentIndex = &builtBVH[bvhWriteIter].indices.jumps.right;
			workgroupQueueEnd = (workgroupQueueEnd + 1) % 100; // TODO: Constant for workgroup size
			assert(workgroupQueueEnd != workgroupIter);
		}

		bvhWriteIter++;
	}

	free(leafs);

	scene->bvh = (bvh_t)
	{
		.nodes = builtBVH
	};
}

static uint32_t traverse_bvh(bvh_t* bvh, vec3 rayO, vec3 rayD, float rayMin, float rayMax, uint32_t* instanceCandidates, uint32_t maxInstances)
{
	uint64_t traversalStartTime = cranpl_timestamp_micro();

	uint32_t instanceIter = 0;

	uint32_t testQueue[1000]; // TODO: Currently we just allow 1000 stack pushes. Fix this!
	uint32_t testQueueSize = 1;

	testQueue[0] = 0;
	for (uint32_t testQueueIter = 0; (testQueueIter - testQueueSize) > 0; testQueueIter++)
	{
		uint32_t nodeIndex = testQueue[testQueueIter];

		if (aabb_does_ray_intersect(rayO, rayD, rayMin, rayMax, bvh->nodes[nodeIndex].bound.min, bvh->nodes[nodeIndex].bound.max))
		{
			// All our leaves are packed at the end of the 
			bool isLeaf = bvh->nodes[nodeIndex].isLeaf;
			if (isLeaf)
			{
				instanceCandidates[instanceIter] = bvh->nodes[nodeIndex].indices.instance;
				instanceIter++;
				assert(instanceIter <= maxInstances);
			}
			else
			{
				testQueue[testQueueSize] = bvh->nodes[nodeIndex].indices.jumps.left;
				testQueueSize++;
				testQueue[testQueueSize] = bvh->nodes[nodeIndex].indices.jumps.right;
				testQueueSize++;
			}
		}
	}

	renderStats.bvhTraversalTime += cranpl_timestamp_micro() - traversalStartTime;
	return instanceIter;
}

static void generate_scene(render_context_t* context, ray_scene_t* scene)
{
	uint64_t startTime = cranpl_timestamp_micro();

	static instance_t instances[1];
	static sphere_t baseSphere = { .rad = 0.2f };

	static material_lambert_t lambert = { .color = { 0.5f, 0.8f, 1.0f } };
	static material_metal_t metal = { 0 };

	for (uint32_t i = 0; i < 1; i++)
	{
		instances[i] = (instance_t)
		{
			.pos = { 0 },
			.materialIndex = { .dataIndex = 0,.typeIndex = 0 },
			.renderableIndex = { .dataIndex = 0, .typeIndex = renderable_sphere, }
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
		.renderables[renderable_sphere] = &baseSphere,
		.materials =
		{
			[material_lambert] = &lambert,
			[material_metal] = &metal
		}
	};

	build_bvh(context, scene);
	renderStats.sceneGenerationTime = cranpl_timestamp_micro() - startTime;
}

int backgroundWidth, backgroundHeight, backgroundStride;
float* restrict background;
static vec3 cast_scene(render_context_t* context, ray_scene_t* scene, vec3 rayO, vec3 rayD, uint32_t depth)
{
	if (depth >= renderConfig.maxDepth)
	{
		return (vec3) { 0 };
	}

	renderStats.rayCount++;

	const float NoRayIntersection = FLT_MAX;

	// TODO: This recast bias is simply to avoid re-intersecting with our object when casting.
	// Do we want to handle this some other way?
	const float ReCastBias = 0.01f;

	uint32_t closestInstanceIndex = 0;
	float closestDistance = NoRayIntersection;

	// TODO: Traverse our BVH instead - Done
	// TODO: Gather list of candidate BVHs - Done
	// TODO: Sort list of instances by type
	// TODO: find closest intersection in candidates
	// TODO: ?!?!?
	uint64_t intersectionStartTime = cranpl_timestamp_micro();

	uint32_t candidates[100]; // TODO: Max candidates of 100?
	uint32_t candidateCount = traverse_bvh(&scene->bvh, rayO, rayD, ReCastBias, NoRayIntersection, candidates, 100);
	for (uint32_t i = 0; i < candidateCount; i++)
	{
		// TODO: Add support for other shapes
		uint32_t candidateIndex = candidates[i];

		vec3 instancePos = scene->instances.data[candidateIndex].pos;
		renderable_index_t renderableIndex = scene->instances.data[candidateIndex].renderableIndex;
		assert(renderableIndex.typeIndex == renderable_sphere);
		sphere_t sphere = ((sphere_t*)scene->renderables[renderable_sphere])[renderableIndex.dataIndex];

		if (sphere_does_ray_intersect(rayO, rayD, instancePos, sphere.rad))
		{
			float intersectionDistance = sphere_ray_intersection(rayO, rayD, ReCastBias, NoRayIntersection, instancePos, sphere.rad);
			// TODO: Do we want to handle tie breakers somehow?
			if (intersectionDistance < closestDistance)
			{
				closestInstanceIndex = candidateIndex;
				closestDistance = intersectionDistance;
			}
		}
	}
	renderStats.intersectionTime += cranpl_timestamp_micro() - intersectionStartTime;

	if (closestDistance != NoRayIntersection)
	{
		vec3 instancePos = scene->instances.data[closestInstanceIndex].pos;
		material_index_t materialIndex = scene->instances.data[closestInstanceIndex].materialIndex;
		renderable_index_t renderableIndex = scene->instances.data[closestInstanceIndex].renderableIndex;
		// TODO: Still only handling spheres
		sphere_t sphere = ((sphere_t*)scene->renderables[renderable_sphere])[renderableIndex.dataIndex];

		vec3 intersectionPoint = vec3_add(rayO, vec3_mulf(rayD, closestDistance));
		vec3 surfacePoint = vec3_sub(intersectionPoint, instancePos);
		vec3 normal = vec3_mulf(surfacePoint, rcp(sphere.rad));

		// TODO: real materials please
		switch (materialIndex.typeIndex)
		{
		case material_lambert:
		{
			material_lambert_t lambertData = ((material_lambert_t*)scene->materials[material_lambert])[materialIndex.dataIndex];

			vec3 spherePoint = vec3_add(normal, sphere_random(&context->randomSeed));
			vec3 result = cast_scene(context, scene, intersectionPoint, spherePoint, depth+1);

			float lambertCosine = fmaxf(0.0f, vec3_dot(vec3_normalized(spherePoint), normal));
			assert(lambertCosine <= 1.05f);
			vec3 sceneCast = vec3_mulf(result, lambertCosine);

			return vec3_mul(sceneCast, vec3_mulf(lambertData.color, RPI));
		}
		case material_metal:
		{
			vec3 castDir = vec3_reflect(rayD, normal);
			vec3 sceneCast = cast_scene(context, scene, intersectionPoint, castDir, depth+1);

			return sceneCast;
		}
		default:
			assert(false); // missing type!
			break;
		}

		// TODO: lighting be like
		return (vec3) { 0 };
	}

	uint64_t skyboxStartTime = cranpl_timestamp_micro();
	vec3 skybox = sample_hdr(rayD, background, backgroundWidth, backgroundHeight, backgroundStride);
	renderStats.skyboxTime += cranpl_timestamp_micro() - skyboxStartTime;

	return skybox;
}

int main()
{
	renderConfig = (render_config_t)
	{
		.maxDepth = UINT32_MAX,
		.samplesPerPixel = 100,
		.renderWidth = 400,
		.renderHeight = 200
	};

	render_context_t renderContext =
	{
		.randomSeed = 12,
	};

	uint64_t startTime = cranpl_timestamp_micro();

	background = stbi_loadf("background_4k.hdr", &backgroundWidth, &backgroundHeight, &backgroundStride, 0);

	ray_scene_t scene;
	generate_scene(&renderContext, &scene);

	int32_t imgWidth = renderConfig.renderWidth, imgHeight = renderConfig.renderHeight, imgStride = 4;
	int32_t halfImgWidth = imgWidth / 2, halfImgHeight = imgHeight / 2;

	// TODO: How do we want to express our camera?
	// Currently simply using the near triangle.
	float near = 1.0f, nearHeight = 1.0f, nearWidth = nearHeight * (float)imgWidth / (float)imgHeight;

	vec3 origin = { 0.0f, -1.0f, 0.0f };
	vec3 forward = { .x = 0.0f,.y = 1.0f,.z = 0.0f }, right = { .x = 1.0f,.y = 0.0f,.z = 0.0f }, up = { .x = 0.0f, .y = 0.0f, .z = 1.0f };

	float* restrict hdrImage = malloc(imgWidth * imgHeight * imgStride * sizeof(float));

	uint64_t renderStartTime = cranpl_timestamp_micro();

	uint64_t totalIterationTime = 0;
	// Sample our scene for every pixel in the bitmap. (Could be upsampled if we wanted to)
	float xStep = nearWidth / (float)imgWidth, yStep = nearHeight / (float)imgHeight;
	for (int32_t y = -halfImgHeight; y < halfImgHeight; y++)
	{
		// Progress data
		uint64_t iterationStartTime = cranpl_timestamp_micro();
		{
			system("cls");
			printf("Completed: %.2f%%\n", ((float)(y + halfImgHeight) / (float)imgHeight) * 100.0f);
			if (totalIterationTime != 0)
			{
				// Use doubles, our value can be quite large until we divide.
				float timeStep = micro_to_seconds(totalIterationTime) / (float)(y + halfImgHeight);
				printf("Remaining Time: %.2f\n\n", timeStep * (imgHeight - (y + halfImgHeight)));
			}
		}

		float yOff = yStep * (float)y;
		for (int32_t x = -halfImgWidth; x < halfImgWidth; x++)
		{
			float xOff = xStep * (float)x;

			vec3 sceneColor = { 0 };
			for (uint32_t i = 0; i < renderConfig.samplesPerPixel; i++)
			{
				renderStats.primaryRayCount++;

				float randX = xOff + xStep * (rand01f(&renderContext.randomSeed) * 0.5f - 0.5f);
				float randY = yOff + yStep * (rand01f(&renderContext.randomSeed) * 0.5f - 0.5f);

				// Construct our ray as a vector going from our origin to our near plane
				// V = F*n + R*ix*worldWidth/imgWidth + U*iy*worldHeight/imgHeight
				vec3 rayDir = vec3_add(vec3_mulf(forward, near), vec3_add(vec3_mulf(right, randX), vec3_mulf(up, randY)));
				sceneColor = vec3_add(sceneColor, vec3_mulf(cast_scene(&renderContext, &scene, origin, rayDir, 0), rcp((float)renderConfig.samplesPerPixel)));
			}

			int32_t imgIdx = ((y + halfImgHeight) * imgWidth + (x + halfImgWidth)) * imgStride;
			hdrImage[imgIdx + 0] = sceneColor.x;
			hdrImage[imgIdx + 1] = sceneColor.y;
			hdrImage[imgIdx + 2] = sceneColor.z;
			hdrImage[imgIdx + 3] = 1.0f;
		}

		// Progress data
		{
			totalIterationTime += cranpl_timestamp_micro() - iterationStartTime;
		}
	}

	renderStats.renderTime = (cranpl_timestamp_micro() - renderStartTime);

	// Image Space Effects
	{
		uint64_t imageSpaceStartTime = cranpl_timestamp_micro();
		// reinhard tonemapping
		for (int32_t y = 0; y < imgHeight; y++)
		{
			for (int32_t x = 0; x < imgWidth; x++)
			{
				int32_t readIndex = (y * imgWidth + x) * imgStride;

				hdrImage[readIndex + 0] = hdrImage[readIndex + 0] / (hdrImage[readIndex + 0] + 1.0f);
				hdrImage[readIndex + 1] = hdrImage[readIndex + 1] / (hdrImage[readIndex + 1] + 1.0f);
				hdrImage[readIndex + 2] = hdrImage[readIndex + 2] / (hdrImage[readIndex + 2] + 1.0f);
			}
		}
		renderStats.imageSpaceTime = cranpl_timestamp_micro() - imageSpaceStartTime;
	}

	// Convert HDR to 8 bit bitmap
	{
		uint8_t* restrict bitmap = malloc(imgWidth * imgHeight * imgStride);
		for (int32_t i = 0; i < imgWidth * imgHeight * imgStride; i+=imgStride)
		{
			bitmap[i + 0] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 2]));
			bitmap[i + 1] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 1]));
			bitmap[i + 2] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 0]));
			bitmap[i + 3] = (uint8_t)(255.99f * hdrImage[i + 3]);
		}

		cranpl_write_bmp("render.bmp", bitmap, imgWidth, imgHeight);
		system("render.bmp");
		free(bitmap);
	}

	free(hdrImage);
	stbi_image_free(background);

	renderStats.totalTime = cranpl_timestamp_micro() - startTime;

	// Print stats
	{
		system("cls");
		printf("Total Time: %f\n", micro_to_seconds(renderStats.totalTime));
		printf("\tScene Generation Time: %f [%f%%]\n", micro_to_seconds(renderStats.sceneGenerationTime), (float)renderStats.sceneGenerationTime / (float)renderStats.totalTime);
		printf("\tRender Time: %f [%f%%]\n", micro_to_seconds(renderStats.renderTime), (float)renderStats.renderTime / (float)renderStats.totalTime);
		printf("\t\tIntersection Time: %f [%f%%]\n", micro_to_seconds(renderStats.intersectionTime), (float)renderStats.intersectionTime / (float)renderStats.renderTime);
		printf("\t\t\tBVH Traversal Time: %f [%f%%]\n", micro_to_seconds(renderStats.bvhTraversalTime), (float)renderStats.bvhTraversalTime / (float)renderStats.intersectionTime);
		printf("\t\tSkybox Time: %f [%f%%]\n", micro_to_seconds(renderStats.skyboxTime), (float)renderStats.skyboxTime / (float)renderStats.renderTime);
		printf("\tImage Space Time: %f [%f%%]\n", micro_to_seconds(renderStats.imageSpaceTime), (float)renderStats.imageSpaceTime / (float)renderStats.totalTime);
		printf("\n");
		printf("MRays/seconds: %f\n", (float)renderStats.rayCount / micro_to_seconds(renderStats.renderTime) / 1000000.0f);
		printf("Rays Fired: %" PRIu64 "\n", renderStats.rayCount);
		printf("\tCamera Rays Fired: %" PRIu64 " [%f%%]\n", renderStats.primaryRayCount, (float)renderStats.primaryRayCount / (float)renderStats.rayCount);
		printf("\tBounce Rays Fired: %" PRIu64 " [%f%%]\n", renderStats.rayCount - renderStats.primaryRayCount, (float)(renderStats.rayCount - renderStats.primaryRayCount) / (float)renderStats.rayCount);

		system("pause");
	}
	return 0;
}
