#pragma once

// Convention
// cf = floating point
// cfl = floating point lanes
// cv# = vector math (i.e. cv2, cv3)
// cvl = vector lane math (i.e. cv3l)
// cm# = matrix math
// cmi = miscellaneous

#include <math.h>
#include <immintrin.h>

#define cran_inline inline
#define cran_forceinline __forceinline
#define cran_align(a) __declspec(align(a))

#ifdef _MSC_BUILD
#pragma warning(disable : 4201)
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
	float x, y;
} cv2;

typedef union
{
	struct
	{
		float x, y, z;
	};

	struct
	{
		float r, g, b;
	};
} cv3;

typedef union
{
	struct
	{
		float x, y, z, w;
	};

	struct
	{
		float r, g, b, a;
	};
} cv4;

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
cran_forceinline float cf_bilinear(float topLeft, float topRight, float bottomLeft, float bottomRight, float tx, float ty);
cran_forceinline float cf_lerp(float a, float b, float t);
cran_forceinline float cf_sign(float a);
// Guarantees to return either -1 or 1
cran_forceinline float cf_sign_no_zero(float a);
cran_forceinline bool cf_finite(float a);
cran_forceinline float cf_frac(float a);

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

// V2 API
cran_forceinline cv2 cv2_mulf(cv2 l, float r);
cran_forceinline cv2 cv2_add(cv2 l, cv2 r);

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
cran_forceinline float cv3_sqrlength(cv3 v);
cran_forceinline float cv3_sqrdistance(cv3 l, cv3 r);
cran_forceinline cv3 cv3_normalize(cv3 v);
cran_forceinline cv3 cv3_min(cv3 v, cv3 m);
cran_forceinline cv3 cv3_max(cv3 v, cv3 m);
cran_forceinline cv3 cv3_rcp(cv3 v);
cran_forceinline cv3 cv3_fast_rcp(cv3 v);
// Expecting i to be incident. (i.n < 0)
cran_forceinline cv3 cv3_reflect(cv3 i, cv3 n);
cran_forceinline cv3 cv3_inverse(cv3 i);
// a is between 0 and 2 PI
// t is between 0 and PI (0 being the bottom, PI being the top)
cran_forceinline void cv3_to_spherical(cv3 v, float* cran_restrict a, float* cran_restrict t);
// theta is between 0 and PI (vertical plane)
// phi is between 0 and 2PI (horizontal plane)
cran_forceinline cv3 cv3_from_spherical(float theta, float phi, float radius);
cran_forceinline cv3 cv3_barycentric(cv3 a, cv3 b, cv3 c, cv3 uvw);

// V3 Lane API
cran_forceinline cv3l cv3l_replicate(cv3 v);
cran_forceinline void cv3l_set(cv3l* lanes, cv3 v, uint32_t i);
// Stride (in bytes) is stride to next vector
// Offset (in bytes) is offset from strided element to vector
cran_forceinline cv3l cv3l_indexed_load(void const* vectors, uint32_t stride, uint32_t offset, uint32_t* indices, uint32_t indexCount);
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
cran_forceinline bool caabb_does_line_intersect(cv3 a, cv3 b, caabb aabb);
cran_forceinline uint32_t caabb_does_ray_intersect_lanes(cv3 rayO, cv3 rayD, float rayMin, float rayMax, cv3l aabbMin, cv3l aabbMax);

enum
{
	caabb_x = 0,
	caabb_y,
	caabb_z
};
cran_forceinline cv3 caabb_center(caabb l);
cran_forceinline float caabb_centroid(caabb l, uint32_t axis);
cran_forceinline float caabb_side(caabb l, uint32_t axis);
cran_forceinline caabb caabb_merge(caabb l, caabb r);
cran_forceinline float caabb_surface_area(caabb l);
cran_forceinline void caabb_split_8(caabb parent, caabb children[8]);
cran_forceinline caabb caabb_consume(caabb parent, cv3 point);

// Miscellaneous API
// Expecting i to be exitant (i.n > 0)
cran_forceinline cv3 cmi_fresnel_schlick_r0(cv3 r0, cv3 n, cv3 i);
// r1 = exiting refractive index (usually air)
// r2 = entering refactive index
// Expecting i to be exitant (i.n > 0)
cran_forceinline float cmi_fresnel_schlick(float r1, float r2, cv3 n, cv3 i);

// Single Implementation
cran_forceinline float cf_rcp(float f)
{
	return 1.0f / f;
}

cran_forceinline float cf_fast_rcp(float f)
{
	__m128 sse = _mm_rcp_ss(_mm_load_ss(&f));
	_mm_store_ss(&f, sse);
	return f;
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

cran_forceinline float cf_bilinear(float topLeft, float topRight, float bottomLeft, float bottomRight, float tx, float ty)
{
	float top = tx*topRight + (1.0f-tx)*topLeft;
	float bottom = tx * bottomRight + (1.0f - tx)*bottomLeft;
	return ty * top + (1.0f - ty)*bottom;
}

cran_forceinline float cf_lerp(float a, float b, float t)
{
	return t * b + (1.0f - t)*a;
}

cran_forceinline float cf_sign(float a)
{
	// Don't handle NaN, inf
	/*union
	{
		uint32_t u;
		float f;
	} conv;
	conv.f = a;
	conv.u = (conv.u & 0x80000000 | 0x3F800000) & ((int32_t)(-conv.u ^ conv.u) >> 31);
	return conv.f;*/

	__m128 f = _mm_load_ss(&a);
	__m128 c0 = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	__m128 c1 = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
	__m128i c2 = _mm_setzero_si128();

	__m128 s = _mm_or_ps(_mm_and_ps(f, c0), c1);

	__m128i u = _mm_castps_si128(f);
	u = _mm_srai_epi32(_mm_xor_si128(_mm_sub_epi32(c2, u), u), 31);

	_mm_store_ss(&a, _mm_and_ps(s, _mm_castsi128_ps(u)));
	return a;
}

cran_forceinline float cf_sign_no_zero(float a)
{
	// Don't handle NaN, inf or 0

	__m128 f = _mm_load_ss(&a);
	__m128 c0 = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	__m128 c1 = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));

	_mm_store_ss(&a, _mm_or_ps(_mm_and_ps(f, c0), c1));
	return a;
}

cran_forceinline bool cf_finite(float a)
{
	union
	{
		uint32_t u;
		float f;
	} conv;
	conv.f = a;
	return (conv.u & 0x7F800000) != 0x7F800000;
}

cran_forceinline float cf_frac(float a)
{
	return a - truncf(a);
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

// V2 Implementation
cran_forceinline cv2 cv2_mulf(cv2 l, float r)
{
	return (cv2) { .x = l.x * r, .y = l.y * r };
}

cran_forceinline cv2 cv2_add(cv2 l, cv2 r)
{
	return (cv2) {.x = l.x + r.x, .y = l.y + r.y};
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

cran_forceinline float cv3_sqrlength(cv3 v)
{
	return (v.x * v.x + v.y * v.y + v.z * v.z);
}

cran_forceinline float cv3_sqrdistance(cv3 l, cv3 r)
{
	return cv3_sqrlength(cv3_sub(l, r));
}

cran_forceinline cv3 cv3_normalize(cv3 v)
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

cran_forceinline cv3 cv3_inverse(cv3 i)
{
	return (cv3) { -i.x, -i.y, -i.z };
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
	return (cv3) { cosf(phi) * sinf(theta) * radius, sinf(phi) * sinf(theta) * radius, radius * cosf(theta) };
}

cran_forceinline cv3 cv3_barycentric(cv3 a, cv3 b, cv3 c, cv3 uvw)
{
	return cv3_add(cv3_add(cv3_mulf(a, uvw.x), cv3_mulf(b, uvw.y)), cv3_mulf(c, uvw.z));
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

cran_forceinline cv3l cv3l_indexed_load(void const* vectors, uint32_t stride, uint32_t offset, uint32_t* indices, uint32_t indexCount)
{
	__m128 loadedVectors[cran_lane_count];
	for (uint32_t i = 0; i < indexCount; i++)
	{
		uint8_t const* vectorData = (uint8_t*)vectors;
		loadedVectors[i] = _mm_load_ps((float const*)(vectorData + indices[i] * stride + offset));
	}

	__m128 XY0 = _mm_shuffle_ps(loadedVectors[0], loadedVectors[1], _MM_SHUFFLE(1, 0, 1, 0));
	__m128 XY1 = _mm_shuffle_ps(loadedVectors[2], loadedVectors[3], _MM_SHUFFLE(1, 0, 1, 0));
	__m128 Z0 = _mm_shuffle_ps(loadedVectors[0], loadedVectors[1], _MM_SHUFFLE(3, 2, 3, 2));
	__m128 Z1 = _mm_shuffle_ps(loadedVectors[2], loadedVectors[3], _MM_SHUFFLE(3, 2, 3, 2));

	return (cv3l)
	{
		.x = {.sse = _mm_shuffle_ps(XY0, XY1, _MM_SHUFFLE(2, 0, 2, 0))},
		.y = {.sse = _mm_shuffle_ps(XY0, XY1, _MM_SHUFFLE(3, 1, 3, 1))},
		.z = {.sse = _mm_shuffle_ps(Z0, Z1, _MM_SHUFFLE(2, 0, 2, 0))}
	};
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
	float sign = cf_sign_no_zero(n.z);
	float a = -cf_rcp(sign + n.z);
	float b = n.x*n.y*a;
	cv3 i = (cv3) { 1.0f + sign * n.x*n.x*a, sign * b, -sign * n.x };
	cv3 j = (cv3) { b, sign + n.y*n.y*a, -n.y };

	return cm3_from_basis(i, j, n);
}

cran_forceinline cv3 cm3_mul_cv3(cm3 m, cv3 v)
{
	cv3 rx = cv3_mulf(m.i, v.x);
	cv3 ry = cv3_mulf(m.j, v.y);
	cv3 rz = cv3_mulf(m.k, v.z);

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

cran_forceinline bool caabb_does_line_intersect(cv3 a, cv3 b, caabb aabb)
{
	// TODO: Can we specialize this intersection?
	return caabb_does_ray_intersect(a, cv3_sub(b, a), 0.0f, 1.0f, aabb);
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

cran_forceinline cv3 caabb_center(caabb l)
{
	return (cv3) { (l.max.x + l.min.x)*0.5f, (l.max.y + l.min.y)*0.5f, (l.max.z + l.min.z)*0.5f };
}

cran_forceinline float caabb_centroid(caabb l, uint32_t axis)
{
	return ((&l.max.x)[axis] + (&l.min.x)[axis]) * 0.5f;
}

cran_forceinline float caabb_side(caabb l, uint32_t axis)
{
	return ((&l.max.x)[axis] - (&l.min.x)[axis]);
}

cran_forceinline caabb caabb_merge(caabb l, caabb r)
{
	return (caabb) { .max = cv3_max(l.max, r.max), .min = cv3_min(l.min, r.min) };
}

cran_forceinline float caabb_surface_area(caabb l)
{
	return ((caabb_side(l,caabb_x)*caabb_side(l,caabb_y))+(caabb_side(l,caabb_y)*caabb_side(l,caabb_z))+(caabb_side(l,caabb_x)*caabb_side(l,caabb_z)))*2.0f;
}

cran_forceinline void caabb_split_8(caabb parent, caabb children[8])
{
	cv3 center = cv3_mulf(cv3_sub(parent.max, parent.min), 0.5f);
	cv3 childSize = cv3_mulf(cv3_sub(parent.max, parent.min), 0.5f);

	children[0] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
	childSize.x = -childSize.x;
	children[1] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
	childSize.y = -childSize.y;
	children[2] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
	childSize.x = -childSize.x;
	children[3] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
	childSize.z = -childSize.z;
	children[4] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
	childSize.y = -childSize.y;
	children[5] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
	childSize.x = -childSize.x;
	children[6] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
	childSize.y = -childSize.y;
	children[7] = (caabb) {.min = center, .max = cv3_add(center, childSize) };
}

cran_forceinline caabb caabb_consume(caabb parent, cv3 point)
{
	return (caabb) { .min = cv3_min(parent.min, point), .max = cv3_max(parent.max, point) };
}

// Miscellaneous Implementation
cran_forceinline cv3 cmi_fresnel_schlick_r0(cv3 r0, cv3 n, cv3 i)
{
	float a = fminf(1.0f - cv3_dot(n, i), 1.0f);
	return cv3_add(r0, cv3_mulf(cv3_sub((cv3) { 1.0f, 1.0f, 1.0f }, r0), a*a*a*a*a));
}

// r1 = exiting refractive index (usually air)
// r2 = entering refactive index
cran_forceinline float cmi_fresnel_schlick(float r1, float r2, cv3 n, cv3 i)
{
	float r0 = (r1 - r2) / (r1 + r2);
	r0 *= r0;
	float a = fminf(1.0f - cv3_dot(n, i), 1.0f);
	return r0 + (1.0f - r0)*a*a*a*a*a;
}
