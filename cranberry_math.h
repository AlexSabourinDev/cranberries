#pragma once

#include <math.h>
#include <immintrin.h>

#define cran_inline inline
#define cran_forceinline __forceinline
#define cran_align(a) __declspec(align(16))

#ifdef _MSC_BUILD
#define cran_restrict __restrict
#else
#define cran_restrict restrict
#endif

#define cran_pi_val 3.14159265358979323846264338327f
static const float cran_pi = cran_pi_val;
static const float cran_tao = cran_pi_val * 2.0f;
static const float cran_rpi = 1.0f / cran_pi_val;
static const float cran_rtao = 1.0f / (cran_pi_val * 2.0f);

#define cran_lane_count 4
cran_align(16) typedef union
{
	float f[cran_lane_count];
	__m128 sse;
} cfl;

typedef struct
{
	float x, y, z;
} cv3;

typedef struct
{
	cfl x;
	cfl y;
	cfl z;
} cv3l;

typedef struct
{
	cv3 i, j, k;
} cm3;

typedef struct
{
	cv3 min;
	cv3 max;
} caabb;

// Single API
cran_forceinline float cf_rcp(float f);
cran_forceinline float cf_fast_rcp(float f);
cran_forceinline float cf_rsqrt(float f);
cran_forceinline float cf_fast_rsqrt(float f);
cran_forceinline bool cf_quadratic(float a, float b, float c, float* cran_restrict out1, float* cran_restrict out2);

// Lane API
cran_forceinline cfl cfl_replicate(float f);
cran_forceinline cfl cfl_load(float* v);
cran_forceinline cfl cfl_max(cfl l, cfl r);
cran_forceinline cfl cfl_min(cfl l, cfl r);
cran_forceinline cfl cfl_less(cfl l, cfl r);
cran_forceinline cfl cfl_add(cfl l, cfl r);
cran_forceinline cfl cfl_sub(cfl l, cfl r);
cran_forceinline cfl cfl_mul(cfl l, cfl r);
cran_forceinline int cfl_mask(cfl v);
cran_forceinline cfl cfl_rcp(cfl v);
cran_forceinline cfl cfl_lt(cfl l, cfl r);

// V3 API
cran_forceinline cv3 cv3_mulf(cv3 l, float r);
cran_forceinline cv3 cv3_add(cv3 l, cv3 r);
cran_forceinline cv3 cv3_addf(cv3 l, float r);
cran_forceinline cv3 cv3_sub(cv3 l, cv3 r);
cran_forceinline cv3 cv3_subf(cv3 l, float r);
cran_forceinline cv3 cv3_mul(cv3 l, cv3 r);
cran_forceinline float cv3_dot(cv3 l, cv3 r);
cran_forceinline cv3 cv3_cross(cv3 l, cv3 r);
cran_forceinline cv3 cv3_lerp(cv3 l, cv3 r, float t);
cran_forceinline float cv3_length(cv3 v);
cran_forceinline float cv3_rlength(cv3 v);
cran_forceinline cv3 cv3_normalized(cv3 v);
cran_forceinline cv3 cv3_min(cv3 v, cv3 m);
cran_forceinline cv3 cv3_max(cv3 v, cv3 m);
cran_forceinline cv3 cv3_rcp(cv3 v);
cran_forceinline cv3 cv3_fast_rcp(cv3 v);
cran_forceinline cv3 cv3_reflect(cv3 i, cv3 n);
// a is between 0 and 2 PI
// t is between 0 and PI (0 being the bottom, PI being the top)
cran_forceinline void cv3_to_spherical(cv3 v, float* cran_restrict a, float* cran_restrict t);
// theta is between 0 and 2PI (horizontal plane)
// phi is between 0 and PI (vertical plane)
cran_forceinline cv3 cv3_from_spherical(float theta, float phi, float radius);

// V3 Lane API
cran_forceinline cv3l cv3l_replicate(cv3 v);
cran_forceinline void cv3l_set(cv3l* lanes, cv3 v, uint32_t i);
cran_forceinline cv3l cv3l_add(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_sub(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_mul(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_min(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_max(cv3l l, cv3l r);

// Matrix API
cran_forceinline cm3 cm3_from_basis(cv3 i, cv3 j, cv3 k);
cran_forceinline cm3 cm3_basis_from_normal(cv3 n);
cran_forceinline cv3 cm3_mul_cv3(cm3 m, cv3 v);
cran_forceinline cv3 cm3_rotate_cv3(cm3 m, cv3 v);

// AABB API
cran_forceinline bool caabb_does_ray_intersect(cv3 rayO, cv3 rayD, float rayMin, float rayMax, caabb aabb);
cran_forceinline uint32_t caabb_does_ray_intersect_lanes(cv3 rayO, cv3 rayD, float rayMin, float rayMax, cv3l aabbMin, cv3l aabbMax);

// Single Implementation
cran_forceinline float cf_rcp(float f)
{
	return 1.0f / f;
}

cran_forceinline float cf_fast_rcp(float f)
{
	union
	{
		__m128 sse;
		float f[4];
	} conv;
	conv.sse = _mm_rcp_ss(_mm_set_ss(f));
	return conv.f[0];
}

cran_forceinline float cf_rsqrt(float f)
{
	return 1.0f / sqrtf(f);
}

cran_forceinline float cf_fast_rsqrt(float f)
{
	union
	{
		__m128 sse;
		float f[4];
	} conv;
	conv.sse = _mm_rsqrt_ss(_mm_set_ss(f));
	return conv.f[0];
}

cran_forceinline bool cf_quadratic(float a, float b, float c, float* cran_restrict out1, float* cran_restrict out2)
{
	// TODO: Replace with more numerically robust version.
	float determinant = b * b - 4.0f * a * c;
	if (determinant < 0.0f)
	{
		return false;
	}

	float d = sqrtf(determinant);
	float e = cf_rcp(2.0f * a);

	*out1 = (-b - d) * e;
	*out2 = (-b + d) * e;
	return true;
}

// Lane Implementation
cran_forceinline cfl cfl_replicate(float f)
{
	return (cfl) { .sse = _mm_set_ps1(f) };
}

cran_forceinline cfl cfl_load(float* f)
{
	return (cfl) { .sse = _mm_loadu_ps(f) };
}

cran_forceinline cfl cfl_max(cfl l, cfl r)
{
	return (cfl) { .sse = _mm_max_ps(l.sse, r.sse) };
}

cran_forceinline cfl cfl_min(cfl l, cfl r)
{
	return (cfl) { .sse = _mm_min_ps(l.sse, r.sse) };
}

cran_forceinline cfl cfl_less(cfl l, cfl r)
{
	return (cfl) { .sse = _mm_cmplt_ps(l.sse, r.sse) };
}

cran_forceinline cfl cfl_add(cfl l, cfl r)
{
	return (cfl) { .sse = _mm_add_ps(l.sse, r.sse) };
}

cran_forceinline cfl cfl_sub(cfl l, cfl r)
{
	return (cfl) { .sse = _mm_sub_ps(l.sse, r.sse) };
}

cran_forceinline cfl cfl_mul(cfl l, cfl r)
{
	return (cfl) { .sse = _mm_mul_ps(l.sse, r.sse) };
}

cran_forceinline int cfl_mask(cfl v)
{
	return _mm_movemask_ps(v.sse);
}

cran_forceinline cfl cfl_rcp(cfl v)
{
	return (cfl) { .sse = _mm_rcp_ps(v.sse) };
}

cran_forceinline cfl cfl_lt(cfl l, cfl r)
{
	return  (cfl) { .sse = _mm_cmplt_ps(l.sse, r.sse) };
}

// V3 Implementation
cran_forceinline cv3 cv3_mulf(cv3 l, float r)
{
	return (cv3) { .x = l.x * r, .y = l.y * r, .z = l.z * r };
}

cran_forceinline cv3 cv3_add(cv3 l, cv3 r)
{
	return (cv3) {.x = l.x + r.x, .y = l.y + r.y, .z = l.z + r.z};
}

cran_forceinline cv3 cv3_addf(cv3 l, float r)
{
	return (cv3) {.x = l.x + r, .y = l.y + r, .z = l.z + r};
}

cran_forceinline cv3 cv3_sub(cv3 l, cv3 r)
{
	return (cv3) {.x = l.x - r.x, .y = l.y - r.y, .z = l.z - r.z};
}

cran_forceinline cv3 cv3_subf(cv3 l, float r)
{
	return (cv3) {.x = l.x - r, .y = l.y - r, .z = l.z - r};
}

cran_forceinline cv3 cv3_mul(cv3 l, cv3 r)
{
	return (cv3) {.x = l.x * r.x, .y = l.y * r.y, .z = l.z * r.z};
}

cran_forceinline float cv3_dot(cv3 l, cv3 r)
{
	return l.x * r.x + l.y * r.y + l.z * r.z;
}

cran_forceinline cv3 cv3_cross(cv3 l, cv3 r)
{
	return (cv3)
	{
		.x = l.y*r.z - l.z*r.y,
		.y = l.z*r.x - l.x*r.z,
		.z = l.x*r.y - l.y*r.x
	};
}

cran_forceinline cv3 cv3_lerp(cv3 l, cv3 r, float t)
{
	return cv3_add(cv3_mulf(l, 1.0f - t), cv3_mulf(r, t));
}

cran_forceinline float cv3_length(cv3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

cran_forceinline float cv3_rlength(cv3 v)
{
	return cf_rsqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

cran_forceinline cv3 cv3_normalized(cv3 v)
{
	return cv3_mulf(v, cv3_rlength(v));
}

cran_forceinline cv3 cv3_min(cv3 v, cv3 m)
{
	return (cv3){fminf(v.x, m.x), fminf(v.y, m.y), fminf(v.z, m.z)};
}

cran_forceinline cv3 cv3_max(cv3 v, cv3 m)
{
	return (cv3){fmaxf(v.x, m.x), fmaxf(v.y, m.y), fmaxf(v.z, m.z)};
}

cran_forceinline cv3 cv3_rcp(cv3 v)
{
	return (cv3) { cf_rcp(v.x), cf_rcp(v.y), cf_rcp(v.z) };
}

cran_forceinline cv3 cv3_fast_rcp(cv3 v)
{
	union
	{
		__m128 sse;
		float f[4];
	} conv;

	conv.sse = _mm_rcp_ps(_mm_loadu_ps(&v.x));
	return (cv3) { conv.f[0], conv.f[1], conv.f[2] };
}

cran_forceinline cv3 cv3_reflect(cv3 i, cv3 n)
{
	return cv3_sub(i, cv3_mulf(n, 2.0f * cv3_dot(i, n)));
}

cran_forceinline void cv3_to_spherical(cv3 v, float* cran_restrict a, float* cran_restrict t)
{
	float rlenght = cv3_rlength(v);
	float azimuth = atan2f(v.y, v.x);
	*a = (azimuth < 0.0f ? cran_tao + azimuth : azimuth);
	*t = acosf(v.z * rlenght);
}

cran_forceinline cv3 cv3_from_spherical(float theta, float phi, float radius)
{
	return (cv3) { cosf(theta) * sinf(phi) * radius, sinf(theta) * sinf(phi) * radius, radius * cosf(phi) };
}

// V3 Lane Implementation
cran_forceinline cv3l cv3l_replicate(cv3 v)
{
	return (cv3l)
	{
		.x = cfl_replicate(v.x),
		.y = cfl_replicate(v.y),
		.z = cfl_replicate(v.z)
	};
}

cran_forceinline void cv3l_set(cv3l* lanes, cv3 v, uint32_t i)
{
	lanes->x.f[i] = v.x;
	lanes->y.f[i] = v.y;
	lanes->z.f[i] = v.z;
}

cran_forceinline cv3l cv3l_add(cv3l l, cv3l r)
{
	return (cv3l)
	{
		.x = cfl_add(l.x, r.x),
		.y = cfl_add(l.y, r.y),
		.z = cfl_add(l.z, r.z)
	};
}

cran_forceinline cv3l cv3l_sub(cv3l l, cv3l r)
{
	return (cv3l)
	{
		.x = cfl_sub(l.x, r.x),
		.y = cfl_sub(l.y, r.y),
		.z = cfl_sub(l.z, r.z)
	};
}

cran_forceinline cv3l cv3l_mul(cv3l l, cv3l r)
{
	return (cv3l)
	{
		.x = cfl_mul(l.x, r.x),
		.y = cfl_mul(l.y, r.y),
		.z = cfl_mul(l.z, r.z)
	};
}

cran_forceinline cv3l cv3l_min(cv3l l, cv3l r)
{
	return (cv3l)
	{
		.x = cfl_min(l.x, r.x),
		.y = cfl_min(l.y, r.y),
		.z = cfl_min(l.z, r.z)
	};
}

cran_forceinline cv3l cv3l_max(cv3l l, cv3l r)
{
	return (cv3l)
	{
		.x = cfl_max(l.x, r.x),
		.y = cfl_max(l.y, r.y),
		.z = cfl_max(l.z, r.z)
	};
}

// Matrix Implementation
cran_forceinline cm3 cm3_from_basis(cv3 i, cv3 j, cv3 k)
{
	assert(cv3_length(i) < 1.01f && cv3_length(i) > 0.99f);
	assert(cv3_length(j) < 1.01f && cv3_length(j) > 0.99f);
	assert(cv3_length(k) < 1.01f && cv3_length(k) > 0.99f);
	return (cm3)
	{
		.i = i,
		.j = j,
		.k = k
	};
}

cran_forceinline cm3 cm3_basis_from_normal(cv3 n)
{
	// Frisvad ONB from https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
	// revised from Pixar https://graphics.pixar.com/library/OrthonormalB/paper.pdf#page=2&zoom=auto,-233,561
	float sign = copysignf(1.0f, n.z);
	float a = -cf_rcp(sign + n.z);
	float b = -n.x*n.y*a;
	cv3 i = (cv3) { 1.0f + sign * n.x*n.x*a, sign * b, -sign * n.x };
	cv3 j = (cv3) { b, sign + n.y*n.y*a, -n.y };

	return cm3_from_basis(i, j, n);
}

cran_forceinline cv3 cm3_mul_cv3(cm3 m, cv3 v)
{
	cv3 vx = (cv3) { v.x, v.x, v.x };
	cv3 vy = (cv3) { v.y, v.y, v.y };
	cv3 vz = (cv3) { v.z, v.z, v.z };

	cv3 rx = cv3_mul(vx, m.i);
	cv3 ry = cv3_mul(vy, m.j);
	cv3 rz = cv3_mul(vz, m.k);

	return cv3_add(cv3_add(rx, ry), rz);
}

cran_forceinline cv3 cm3_rotate_cv3(cm3 m, cv3 v)
{
	return cm3_mul_cv3(m, v);
}

// AABB Implementation
cran_forceinline bool caabb_does_ray_intersect(cv3 rayO, cv3 rayD, float rayMin, float rayMax, caabb aabb)
{
	// Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525

	/*cv3 invD = cv3_rcp(rayD);
	cv3 t0s = cv3_mul(cv3_sub(aabbMin, rayO), invD);
	cv3 t1s = cv3_mul(cv3_sub(aabbMax, rayO), invD);

	cv3 tsmaller = cv3_min(t0s, t1s);
	cv3 tbigger  = cv3_max(t0s, t1s);
 
	float tmin = fmaxf(rayMin, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
	float tmax = fminf(rayMax, fminf(tbigger.x, fminf(tbigger.y, tbigger.z)));
	return (tmin < tmax);*/

	cfl vrayO = cfl_load(&rayO.x);
	cfl vrayD = cfl_load(&rayD.x);
	cfl vmin = cfl_load(&aabb.min.x);
	cfl vmax = cfl_load(&aabb.max.x);
	cfl vrayMax = cfl_replicate(rayMax);
	cfl vrayMin = cfl_replicate(rayMin);

	cfl invD = cfl_rcp(vrayD);
	cfl t0s = cfl_mul(cfl_sub(vmin, vrayO), invD);
	cfl t1s = cfl_mul(cfl_sub(vmax, vrayO), invD);

	cfl tsmaller = cfl_min(t0s, t1s);
	// Our fourth element is bad, we need to overwrite it
	tsmaller.sse = _mm_shuffle_ps(tsmaller.sse, tsmaller.sse, _MM_SHUFFLE(2, 2, 1, 0));

	cfl tbigger = cfl_max(t0s, t1s);
	tbigger.sse = _mm_shuffle_ps(tbigger.sse, tbigger.sse, _MM_SHUFFLE(2, 2, 1, 0));

	tsmaller = cfl_max(tsmaller, (cfl) { .sse = _mm_shuffle_ps(tsmaller.sse, tsmaller.sse, _MM_SHUFFLE(2, 1, 0, 3)) });
	tsmaller = cfl_max(tsmaller, (cfl) { .sse = _mm_shuffle_ps(tsmaller.sse, tsmaller.sse, _MM_SHUFFLE(1, 0, 3, 2)) });
	vrayMin = cfl_max(vrayMin, tsmaller);

	tbigger = cfl_min(tbigger, (cfl) { .sse = _mm_shuffle_ps(tbigger.sse, tbigger.sse, _MM_SHUFFLE(2, 1, 0, 3)) });
	tbigger = cfl_min(tbigger, (cfl) { .sse = _mm_shuffle_ps(tbigger.sse, tbigger.sse, _MM_SHUFFLE(1, 0, 3, 2)) });
	vrayMax = cfl_min(vrayMax, tbigger);

	return cfl_mask(cfl_lt(vrayMin, vrayMax));
}

cran_forceinline uint32_t caabb_does_ray_intersect_lanes(cv3 rayO, cv3 rayD, float rayMin, float rayMax, cv3l aabbMin, cv3l aabbMax)
{
	cv3l rayOLanes = cv3l_replicate(rayO);
	cv3l invD = cv3l_replicate(cv3_rcp(rayD));
	cv3l t0s = cv3l_mul(cv3l_sub(aabbMin, rayOLanes), invD);
	cv3l t1s = cv3l_mul(cv3l_sub(aabbMax, rayOLanes), invD);

	cv3l tsmaller = cv3l_min(t0s, t1s);
	cv3l tbigger  = cv3l_max(t0s, t1s);
 
	cfl rayMinLane = cfl_replicate(rayMin);
	cfl rayMaxLane = cfl_replicate(rayMax);
	cfl tmin = cfl_max(rayMinLane, cfl_max(tsmaller.x, cfl_max(tsmaller.y, tsmaller.z)));
	cfl tmax = cfl_min(rayMaxLane, cfl_min(tbigger.x, cfl_min(tbigger.y, tbigger.z)));
	cfl result = cfl_less(tmin, tmax);
	return cfl_mask(result);
}
