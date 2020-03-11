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
#include <immintrin.h>

#include "stb_image.h"
#include "cranberry_platform.h"
#include "cranberry_loader.h"

struct
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
	int32_t randomSeed;
	uint32_t depth;
} render_context_t;

static float micro_to_seconds(uint64_t time)
{
	return (float)time / 1000000.0f;
}

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

static bool quadratic(float a, float b, float c, float* cran_restrict out1, float* cran_restrict out2)
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

#define lane_count 4
__declspec(align(16)) typedef union
{
	float f[lane_count];
	__m128 sse;
} lane_t;

static lane_t lane_replicate(float f)
{
	return (lane_t) { .sse = _mm_set_ps1(f) };
}

static lane_t lane_max(lane_t l, lane_t r)
{
	return (lane_t) { .sse = _mm_max_ps(l.sse, r.sse) };
}

static lane_t lane_min(lane_t l, lane_t r)
{
	return (lane_t) { .sse = _mm_min_ps(l.sse, r.sse) };
}

static lane_t lane_less(lane_t l, lane_t r)
{
	return (lane_t) { .sse = _mm_cmplt_ps(l.sse, r.sse) };
}

static lane_t lane_add(lane_t l, lane_t r)
{
	return (lane_t) { .sse = _mm_add_ps(l.sse, r.sse) };
}

static lane_t lane_sub(lane_t l, lane_t r)
{
	return (lane_t) { .sse = _mm_sub_ps(l.sse, r.sse) };
}

static lane_t lane_mul(lane_t l, lane_t r)
{
	return (lane_t) { .sse = _mm_mul_ps(l.sse, r.sse) };
}

static uint32_t lane_mask(lane_t v)
{
	return _mm_movemask_ps(v.sse);
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
		.x = l.y*r.z - l.z*r.y,
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
	union
	{
		__m128 sse;
		float f[4];
	} conv;

	conv.sse = _mm_rcp_ps(_mm_loadu_ps(&v.x));
	return (vec3) { conv.f[0], conv.f[1], conv.f[2] };

	// return (vec3) { rcp(v.x), rcp(v.y), rcp(v.z) };
}

typedef struct
{
	lane_t x;
	lane_t y;
	lane_t z;
} vec3_lanes_t;

static vec3_lanes_t vec3_lanes_replicate(vec3 v)
{
	return (vec3_lanes_t)
	{
		.x = lane_replicate(v.x),
		.y = lane_replicate(v.y),
		.z = lane_replicate(v.z)
	};
}

static void vec3_lanes_set(vec3_lanes_t* lanes, vec3 v, uint32_t i)
{
	lanes->x.f[i] = v.x;
	lanes->y.f[i] = v.y;
	lanes->z.f[i] = v.z;
}

static vec3_lanes_t vec3_lanes_add(vec3_lanes_t l, vec3_lanes_t r)
{
	return (vec3_lanes_t)
	{
		.x = lane_add(l.x, r.x),
		.y = lane_add(l.y, r.y),
		.z = lane_add(l.z, r.z)
	};
}

static vec3_lanes_t vec3_lanes_sub(vec3_lanes_t l, vec3_lanes_t r)
{
	return (vec3_lanes_t)
	{
		.x = lane_sub(l.x, r.x),
		.y = lane_sub(l.y, r.y),
		.z = lane_sub(l.z, r.z)
	};
}

static vec3_lanes_t vec3_lanes_mul(vec3_lanes_t l, vec3_lanes_t r)
{
	return (vec3_lanes_t)
	{
		.x = lane_mul(l.x, r.x),
		.y = lane_mul(l.y, r.y),
		.z = lane_mul(l.z, r.z)
	};
}

static vec3_lanes_t vec3_lanes_min(vec3_lanes_t l, vec3_lanes_t r)
{
	return (vec3_lanes_t)
	{
		.x = lane_min(l.x, r.x),
		.y = lane_min(l.y, r.y),
		.z = lane_min(l.z, r.z)
	};
}

static vec3_lanes_t vec3_lanes_max(vec3_lanes_t l, vec3_lanes_t r)
{
	return (vec3_lanes_t)
	{
		.x = lane_max(l.x, r.x),
		.y = lane_max(l.y, r.y),
		.z = lane_max(l.z, r.z)
	};
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

static bool sphere_does_ray_intersect(vec3 rayO, vec3 rayD, float sphereR)
{
	float projectedDistance = -vec3_dot(rayO, rayD);
	float distanceToRaySqr = vec3_dot(rayO,rayO) - projectedDistance * projectedDistance;
	return (distanceToRaySqr < sphereR * sphereR);
}

static float sphere_ray_intersection(vec3 rayO, vec3 rayD, float rayMin, float rayMax, float sphereR)
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
	vec3 raySphereSpace = rayO;
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

static float triangle_ray_intersection(vec3 rayO, vec3 rayD, float rayMin, float rayMax, vec3 A, vec3 B, vec3 C, float* out_u, float* out_v, float* out_w)
{
	// Source: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
	vec3 e1 = vec3_sub(B, A);
	vec3 e2 = vec3_sub(C, A);
	vec3 p = vec3_cross(rayD, e2);
	float d = vec3_dot(e1, p);
	if (d < FLT_EPSILON)
	{
		return rayMax;
	}

	float invD = rcp(d);
	vec3 tv = vec3_sub(rayO, A);
	float v = vec3_dot(tv, p) * invD;
	if (v < 0.0f || v > 1.0f)
	{
		return rayMax;
	}

	vec3 q = vec3_cross(tv, e1);
	float w = vec3_dot(rayD, q) * invD;
	if (w < 0.0f || v + w > 1.0f)
	{
		return rayMax;
	}

	float t = vec3_dot(e2, q) * invD;
	if (t > rayMin && t < rayMax)
	{
		*out_u = 1.0f - v - w;
		*out_v = v;
		*out_w = w;
		return t;
	}

	return rayMax;
}

static vec3 sphere_random(int32_t* seed)
{
	vec3 p;
	do
	{
		p = vec3_mulf((vec3) { random01f(seed)-0.5f, random01f(seed)-0.5f, random01f(seed)-0.5f }, 2.0f);
	} while (vec3_dot(p, p) <= 1.0f);

	return p;
}

static vec3 vec3_reflect(vec3 i, vec3 n)
{
	return vec3_sub(i, vec3_mulf(n, 2.0f * vec3_dot(i, n)));
}

// a is between 0 and 2 PI
// t is between 0 and PI (0 being the bottom, PI being the top)
static void vec3_to_spherical(vec3 v, float* cran_restrict a, float* cran_restrict t)
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
	tbigger = _mm_shuffle_ps(tbigger, tbigger, _MM_SHUFFLE(2, 2, 1, 0));

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

static uint32_t aabb_does_ray_intersect_lanes(vec3 rayO, vec3 rayD, float rayMin, float rayMax, vec3_lanes_t aabbMin, vec3_lanes_t aabbMax)
{
	vec3_lanes_t rayOLanes = vec3_lanes_replicate(rayO);
	vec3_lanes_t invD = vec3_lanes_replicate(vec3_rcp(rayD));
	vec3_lanes_t t0s = vec3_lanes_mul(vec3_lanes_sub(aabbMin, rayOLanes), invD);
	vec3_lanes_t t1s = vec3_lanes_mul(vec3_lanes_sub(aabbMax, rayOLanes), invD);

	vec3_lanes_t tsmaller = vec3_lanes_min(t0s, t1s);
	vec3_lanes_t tbigger  = vec3_lanes_max(t0s, t1s);
 
	lane_t rayMinLane = lane_replicate(rayMin);
	lane_t rayMaxLane = lane_replicate(rayMax);
	lane_t tmin = lane_max(rayMinLane, lane_max(tsmaller.x, lane_max(tsmaller.y, tsmaller.z)));
	lane_t tmax = lane_min(rayMaxLane, lane_min(tbigger.x, lane_min(tbigger.y, tbigger.z)));
	lane_t result = lane_less(tmin, tmax);
	return lane_mask(result);
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
	aabb* bounds;
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
	vec3 color;
} material_lambert_t;

typedef struct
{
	vec3 color;
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
	vec3 pos;
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
	aabb bound;
	uint32_t index;
} index_aabb_pair_t;

typedef struct
{
	vec3 surface;
	vec3 normal;
	vec3 viewDir;
} shader_inputs_t;

typedef vec3(material_shader_t)(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t* scene, shader_inputs_t inputs);

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
		.bounds = malloc(sizeof(aabb) * 100000), // TODO: Actually get an accurate size?/Release this memory.
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

		aabb bounds = start[0].bound;
		for (uint32_t i = 1; i < count; i++)
		{
			bounds.min = vec3_min(start[i].bound.min, bounds.min);
			bounds.max = vec3_max(start[i].bound.max, bounds.max);
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

	renderStats.bvhNodeCount = bvhWriteIter;
	return builtBVH;
}

static uint32_t traverse_bvh(bvh_t* bvh, vec3 rayO, vec3 rayD, float rayMin, float rayMax, uint32_t* candidates, uint32_t maxInstances)
{
	uint64_t traversalStartTime = cranpl_timestamp_micro();

	uint32_t iter = 0;

	uint32_t testQueue[10000]; // TODO: Currently we just allow 1000 stack pushes. Fix this!
	uint32_t testQueueSize = 1;
	uint32_t testQueueIter = 0;

	testQueue[0] = 0;
	while ((int32_t)testQueueSize - (int32_t)testQueueIter > 0)
	{
		vec3_lanes_t boundMins = { 0 };
		vec3_lanes_t boundMaxs = { 0 };

		uint32_t activeLaneCount = min(testQueueSize - testQueueIter, lane_count);
		for (uint32_t i = 0; i < activeLaneCount; i++)
		{
			uint32_t nodeIndex = testQueue[testQueueIter + i];
			vec3_lanes_set(&boundMins, bvh->bounds[nodeIndex].min, i);
			vec3_lanes_set(&boundMaxs, bvh->bounds[nodeIndex].max, i);
		}

		uint32_t intersections = aabb_does_ray_intersect_lanes(rayO, rayD, rayMin, rayMax, boundMins, boundMaxs);
		if (intersections > 0)
		{
			for (uint32_t i = 0; i < activeLaneCount; i++)
			{
				if (intersections & (1 << i))
				{
					uint32_t nodeIndex = testQueue[testQueueIter + i];

					renderStats.bvhHitCount++;

					// All our leaves are packed at the end of the 
					bool isLeaf = bvh->jumps[nodeIndex].isLeaf;
					if (isLeaf)
					{
						renderStats.bvhLeafHitCount++;

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
					renderStats.bvhMissCount++;
				}
			}
		}
		else
		{
			renderStats.bvhMissCount += activeLaneCount;
		}


		testQueueIter += activeLaneCount;
	}

	renderStats.bvhTraversalTime += cranpl_timestamp_micro() - traversalStartTime;
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

			vec3 vertA, vertB, vertC;
			memcpy(&vertA, mesh.data.vertices.data + vertIndexA * 3, sizeof(vec3));
			memcpy(&vertB, mesh.data.vertices.data + vertIndexB * 3, sizeof(vec3));
			memcpy(&vertC, mesh.data.vertices.data + vertIndexC * 3, sizeof(vec3));

			leafs[i].bound.min = vec3_min(vec3_min(vertA, vertB), vertC);
			leafs[i].bound.max = vec3_max(vec3_max(vertA, vertB), vertC);

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
			vec3 pos = scene->instances.data[i].pos;
			uint32_t renderableIndex = scene->instances.data[i].renderableIndex;

			leafs[i].index = i;

			mesh_t* meshData = &scene->renderables[renderableIndex];
			for (uint32_t vert = 0; vert < meshData->data.vertices.count; vert++)
			{
				// TODO: type pun here
				vec3 vertex;
				memcpy(&vertex, meshData->data.vertices.data + vert * 3, sizeof(vec3));

				leafs[i].bound.min = vec3_min(leafs[i].bound.min, vec3_add(vertex, pos));
				leafs[i].bound.max = vec3_max(leafs[i].bound.max, vec3_add(vertex, pos));
			}
		}

		scene->bvh = build_bvh(context, leafs, leafCount);
		free(leafs);
	}

	renderStats.sceneGenerationTime = cranpl_timestamp_micro() - startTime;
}

int backgroundWidth, backgroundHeight, backgroundStride;
float* cran_restrict background;
static vec3 cast_scene(render_context_t* context, ray_scene_t* scene, vec3 rayO, vec3 rayD)
{
	context->depth++;
	if (context->depth >= renderConfig.maxDepth)
	{
		context->depth--;
		return (vec3) { 0 };
	}

	renderStats.rayCount++;

	const float NoRayIntersection = FLT_MAX;

	struct
	{
		float distance;
		vec3 normal;
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
	uint32_t candidateCount = traverse_bvh(&scene->bvh, rayO, rayD, 0.0f, NoRayIntersection, candidates, 1000);
	for (uint32_t i = 0; i < candidateCount; i++)
	{
		uint32_t candidateIndex = candidates[i];

		vec3 instancePos = scene->instances.data[candidateIndex].pos;
		uint32_t renderableIndex = scene->instances.data[candidateIndex].renderableIndex;

		vec3 rayInstanceO = vec3_sub(rayO, instancePos);

		float intersectionDistance = 0.0f;

		mesh_t* mesh = &scene->renderables[renderableIndex];
		material_index_t* materialIndices = mesh->materialIndices;

		uint32_t meshCandidates[1000]; // TODO:
		uint32_t meshCandidateCount = traverse_bvh(&mesh->bvh, rayO, rayD, 0.0f, NoRayIntersection, meshCandidates, 1000);
		for (uint32_t faceCandidate = 0; faceCandidate < meshCandidateCount; faceCandidate++)
		{
			uint32_t faceIndex = meshCandidates[faceCandidate];

			uint32_t vertIndexA = mesh->data.faces.vertexIndices[faceIndex * 3 + 0];
			uint32_t vertIndexB = mesh->data.faces.vertexIndices[faceIndex * 3 + 1];
			uint32_t vertIndexC = mesh->data.faces.vertexIndices[faceIndex * 3 + 2];

			vec3 vertA, vertB, vertC;
			memcpy(&vertA, mesh->data.vertices.data + vertIndexA * 3, sizeof(vec3));
			memcpy(&vertB, mesh->data.vertices.data + vertIndexB * 3, sizeof(vec3));
			memcpy(&vertC, mesh->data.vertices.data + vertIndexC * 3, sizeof(vec3));

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
				vec3 normalA, normalB, normalC;
				memcpy(&normalA, mesh->data.normals.data + normalIndexA * 3, sizeof(vec3));
				memcpy(&normalB, mesh->data.normals.data + normalIndexB * 3, sizeof(vec3));
				memcpy(&normalC, mesh->data.normals.data + normalIndexC * 3, sizeof(vec3));

				closestHitInfo.normal = vec3_add(vec3_add(vec3_mulf(normalA, u), vec3_mulf(normalB, v)), vec3_mulf(normalC, w));
			}
		}
	}
	renderStats.intersectionTime += cranpl_timestamp_micro() - intersectionStartTime;

	if (closestHitInfo.distance != NoRayIntersection)
	{
		material_index_t materialIndex = closestHitInfo.materialIndex;
		vec3 intersectionPoint = vec3_add(rayO, vec3_mulf(rayD, closestHitInfo.distance));

		vec3 light = shaders[materialIndex.typeIndex](scene->materials[materialIndex.typeIndex], materialIndex.dataIndex, context, scene,
			(shader_inputs_t)
			{
				.surface = intersectionPoint,
				.normal = closestHitInfo.normal,
				.viewDir = rayD
			});
		context->depth--;
		return light;
	}

	vec3 skybox = vec3_lerp((vec3) {1.0f, 1.0f, 1.0f}, (vec3) {0.5f, 0.7f, 1.0f}, rayD.z * 0.5f + 0.5f);
	//uint64_t skyboxStartTime = cranpl_timestamp_micro();
	//vec3 skybox = sample_hdr(rayD, background, backgroundWidth, backgroundHeight, backgroundStride);
	//renderStats.skyboxTime += cranpl_timestamp_micro() - skyboxStartTime;

	context->depth--;
	return skybox;
}

// TODO: This recast bias is simply to avoid re-intersecting with our object when casting.
// Do we want to handle this some other way?
const float ReCastBias = 0.001f;
static vec3 shader_lambert(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t* scene, shader_inputs_t inputs)
{
	material_lambert_t lambertData = ((const material_lambert_t* cran_restrict)materialData)[materialIndex];

	vec3 randomDir = vec3_add(inputs.normal, sphere_random(&context->randomSeed));
	// TODO: Consider iteration instead of recursion
	vec3 result = cast_scene(context, scene, vec3_add(inputs.surface, vec3_mulf(inputs.normal, ReCastBias)), randomDir);

	float lambertCosine = fmaxf(0.0f, vec3_dot(vec3_normalized(randomDir), inputs.normal));
	vec3 sceneCast = vec3_mulf(result, lambertCosine);

	return vec3_mul(sceneCast, vec3_mulf(lambertData.color, RPI));
}

static vec3 shader_mirror(const void* cran_restrict materialData, uint32_t materialIndex, render_context_t* context, ray_scene_t* scene, shader_inputs_t inputs)
{
	material_mirror_t mirrorData = ((const material_mirror_t* cran_restrict)materialData)[materialIndex];

	vec3 castDir = vec3_reflect(inputs.viewDir, inputs.normal);
	vec3 sceneCast = cast_scene(context, scene, vec3_add(inputs.surface, vec3_mulf(inputs.normal, ReCastBias)), castDir);

	return vec3_mul(sceneCast, mirrorData.color);
}

int main()
{
	renderConfig = (render_config_t)
	{
		.maxDepth = 99,
		.samplesPerPixel = 4,
		.renderWidth = 1024,
		.renderHeight = 768
	};

	render_context_t renderContext =
	{
		.randomSeed = 10346425,
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

	vec3 origin = { 0.0f, -2.0f, 0.0f };
	vec3 forward = { .x = 0.0f,.y = 1.0f,.z = 0.0f }, right = { .x = 1.0f,.y = 0.0f,.z = 0.0f }, up = { .x = 0.0f, .y = 0.0f, .z = 1.0f };

	float* cran_restrict hdrImage = malloc(imgWidth * imgHeight * imgStride * sizeof(float));

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

				float randX = xOff + xStep * (random01f(&renderContext.randomSeed) * 0.5f - 0.5f);
				float randY = yOff + yStep * (random01f(&renderContext.randomSeed) * 0.5f - 0.5f);

				// Construct our ray as a vector going from our origin to our near plane
				// V = F*n + R*ix*worldWidth/imgWidth + U*iy*worldHeight/imgHeight
				vec3 rayDir = vec3_add(vec3_mulf(forward, near), vec3_add(vec3_mulf(right, randX), vec3_mulf(up, randY)));
				// TODO: Do we want to scale for average in the loop or outside the loop?
				// With too many SPP, the sceneColor might get too significant.
				sceneColor = vec3_add(sceneColor, cast_scene(&renderContext, &scene, origin, rayDir));
			}
			sceneColor = vec3_mulf(sceneColor, rcp((float)renderConfig.samplesPerPixel));

			int32_t imgIdx = ((y + halfImgHeight) * imgWidth + (x + halfImgWidth)) * imgStride;
			hdrImage[imgIdx + 0] = sceneColor.x;
			hdrImage[imgIdx + 1] = sceneColor.y;
			hdrImage[imgIdx + 2] = sceneColor.z;
			hdrImage[imgIdx + 3] = 1.0f;
		}

		totalIterationTime += cranpl_timestamp_micro() - iterationStartTime;
	}

	renderStats.renderTime = (cranpl_timestamp_micro() - renderStartTime);

	// Image Space Effects
	bool enableImageSpace = false;
	if(enableImageSpace)
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

	renderStats.totalTime = cranpl_timestamp_micro() - startTime;

	// Convert HDR to 8 bit bitmap
	{
		uint8_t* cran_restrict bitmap = malloc(imgWidth * imgHeight * imgStride);
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

	// Print stats
	{
		system("cls");
		printf("Total Time: %f\n", micro_to_seconds(renderStats.totalTime));
		printf("\tScene Generation Time: %f [%f%%]\n", micro_to_seconds(renderStats.sceneGenerationTime), (float)renderStats.sceneGenerationTime / (float)renderStats.totalTime * 100.0f);
		printf("\tRender Time: %f [%f%%]\n", micro_to_seconds(renderStats.renderTime), (float)renderStats.renderTime / (float)renderStats.totalTime * 100.0f);
		printf("\t\tIntersection Time: %f [%f%%]\n", micro_to_seconds(renderStats.intersectionTime), (float)renderStats.intersectionTime / (float)renderStats.renderTime * 100.0f);
		printf("\t\t\tBVH Traversal Time: %f [%f%%]\n", micro_to_seconds(renderStats.bvhTraversalTime), (float)renderStats.bvhTraversalTime / (float)renderStats.intersectionTime * 100.0f);
		printf("\t\t\t\tBVH Tests: %" PRIu64 "\n", renderStats.bvhHitCount + renderStats.bvhMissCount);
		printf("\t\t\t\t\tBVH Hits: %" PRIu64 "[%f%%]\n", renderStats.bvhHitCount, (float)renderStats.bvhHitCount/(float)(renderStats.bvhHitCount + renderStats.bvhMissCount) * 100.0f);
		printf("\t\t\t\t\t\tBVH Leaf Hits: %" PRIu64 "[%f%%]\n", renderStats.bvhLeafHitCount, (float)renderStats.bvhLeafHitCount/(float)renderStats.bvhHitCount * 100.0f);
		printf("\t\t\t\t\tBVH Misses: %" PRIu64 "[%f%%]\n", renderStats.bvhMissCount, (float)renderStats.bvhMissCount/(float)(renderStats.bvhHitCount + renderStats.bvhMissCount) * 100.0f);
		printf("\t\tSkybox Time: %f [%f%%]\n", micro_to_seconds(renderStats.skyboxTime), (float)renderStats.skyboxTime / (float)renderStats.renderTime * 100.0f);
		printf("\tImage Space Time: %f [%f%%]\n", micro_to_seconds(renderStats.imageSpaceTime), (float)renderStats.imageSpaceTime / (float)renderStats.totalTime * 100.0f);
		printf("\n");
		printf("MRays/seconds: %f\n", (float)renderStats.rayCount / micro_to_seconds(renderStats.renderTime) / 1000000.0f);
		printf("Rays Fired: %" PRIu64 "\n", renderStats.rayCount);
		printf("\tCamera Rays Fired: %" PRIu64 " [%f%%]\n", renderStats.primaryRayCount, (float)renderStats.primaryRayCount / (float)renderStats.rayCount * 100.0f);
		printf("\tBounce Rays Fired: %" PRIu64 " [%f%%]\n", renderStats.rayCount - renderStats.primaryRayCount, (float)(renderStats.rayCount - renderStats.primaryRayCount) / (float)renderStats.rayCount * 100.0f);
		printf("\n");
		printf("BVH\n");
		printf("\tBVH Node Count: %" PRIu64 "\n", renderStats.bvhNodeCount);

		system("pause");
	}
	return 0;
}
