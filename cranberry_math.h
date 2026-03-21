// Copyright (C) 2024 Alexandre Sabourin
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

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
#include <stdint.h>
#include <stdbool.h>

#define cran_inline inline
#define cran_forceinline __forceinline
#define cran_align(a) __declspec(align(a))

#ifdef _MSC_BUILD
#pragma warning(disable : 4201)
#define cran_restrict __restrict
#else
#define cran_restrict restrict
#endif

#define cran_pi 3.14159265358979323846264338327f
#define cran_tao (cran_pi * 2.0f)
#define cran_rpi (1.0f / cran_pi)
#define cran_rtao (1.0f / (cran_pi * 2.0f))
#define cran_deg_to_rad (cran_pi / 180.0f)
#define cran_rad_to_deg (180.0f / cran_pi)
#define cran_f32_mantissa_bits 23
#define cran_f32_exp_bits 8
#define cran_f32_exp_bias 127

typedef uint32_t cu32;
typedef int32_t ci32;
typedef uint16_t cu16;
typedef uint16_t half;
typedef float cf;

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

typedef struct 
{
	uint32_t x, y;
} cu2;

typedef struct
{
	uint16_t x, y;
} ch2;

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

	struct
	{
		float f[3];
	};
} cv3;

typedef struct
{
	uint16_t x, y, z;
} ch3;

typedef struct 
{
	uint32_t x, y, z;
} cu3;


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

	struct
	{
		float f[4];
	};
} cv4;

typedef struct
{
	uint16_t x, y, z, w;
} ch4;

typedef struct
{
	cfl x;
	cfl y;
	cfl z;
} cv3l;

typedef union
{
	cv3 v[3];
	struct
	{
		float f[9];
	};
} cm3;

typedef union
{
	cv4 v[3];
	struct 
	{
		float f[12];
	};
} cm3x4;

typedef union
{
	cv4 v[4];
	struct 
	{
		float f[16];
	};
} cm4;

static cm4 const cm4_identity = 
{
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

typedef struct
{
	cv3 min;
	cv3 max;
} caabb;

// Quaternion
typedef struct 
{
	float x, y, z, w;
} cq;

static cv3 const cv3_forward = { 0.0f, 1.0f, 0.0f };
static cv3 const cv3_right = { 1.0f, 0.0f, 0.0f };
static cv3 const cv3_up = { 0.0f, 0.0f, 1.0f };
static cq const cq_identity = { 0.0f, 0.0f, 0.0f, 1.0f };

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
cran_forceinline cu16 cf_f32_to_f16(float f32);
cran_forceinline float cf_f16_to_f32(cu16 f16);
cran_forceinline float cf_mad(float a, float b, float c); 
cran_forceinline bool cf_close_enough(float a, float b, float t);
cran_forceinline float cf_saturate(float f);
cran_forceinline cu32 cf_compress_unorm(float f, uint32_t bitCount);
cran_forceinline float cf_decompress_unorm(cu32 u, uint32_t bitCount);
cran_forceinline uint32_t cf_compress_float_unsigned(float f, uint32_t exponentBitCount, uint32_t mantissaBitCount);
cran_forceinline float cf_decompress_float_unsigned(uint32_t u, uint32_t exponentBitCount, uint32_t mantissaBitCount);
cran_forceinline uint32_t cf_f32_bits_to_u32(float v);
cran_forceinline float cf_u32_bits_to_f32(uint32_t u);

// Uint32
cran_forceinline cu32 cu_div_ceil(cu32 left, cu32 right);

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
cran_forceinline cv2 cv2_rcp(cv2 v);
cran_forceinline cv2 cv2_madf(cv2 v, float m, float a);
cran_forceinline ch2 cv2_f32_to_f16(cv2 f32);
cran_forceinline cv2 cv2_f16_to_f32(ch2 f16);

cran_forceinline cv2 cu2_to_cv2(cu2 v);

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
cran_forceinline float cv3_fast_rlength(cv3 v);
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
cran_forceinline void cv3_to_spherical(cv3 v, float* cran_restrict theta, float* cran_restrict phi);
// x is between 0 and 2 PI
// y is between 0 and PI (0 being the bottom, PI being the top)
cran_forceinline cv2 cv3_to_spherical_cv2(cv3 v);
// theta is between 0 and PI (vertical plane)
// phi is between 0 and 2PI (horizontal plane)
cran_forceinline cv3 cv3_from_spherical(float phi, float theta, float radius);
cran_forceinline cv3 cv3_from_spherical_cv2(cv2 angles, float radius) { return cv3_from_spherical(angles.x, angles.y, radius); }
cran_forceinline cv3 cv3_barycentric(cv3 a, cv3 b, cv3 c, cv3 uvw);
cran_forceinline cv2 cv3_to_octahedral(cv3 dir);
cran_forceinline cv3 cv3_from_octahedral(cv2 oct);
cran_forceinline cv3 cv3_madf(cv3 v, float m, float a);

cran_forceinline ch3 cv3_f32_to_f16(cv3 f32);
cran_forceinline cv3 cv3_f16_to_f32(ch3 f16);

// V3 Lane API
cran_forceinline cv3l cv3l_replicate(cv3 v);
cran_forceinline void cv3l_set(cv3l* lanes, cv3 v, uint32_t i);
// Stride (in bytes) is stride to next vector
// Offset (in bytes) is offset from strided element to vector
// indceCount must be <= cran_lane count. If it isn't, loaded vectors will be limited to cran_lane_count
cran_forceinline cv3l cv3l_indexed_load(void const* vectors, uint32_t stride, uint32_t offset, uint32_t* indices, uint32_t indexCount);
cran_forceinline cv3l cv3l_add(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_sub(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_mul(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_min(cv3l l, cv3l r);
cran_forceinline cv3l cv3l_max(cv3l l, cv3l r);

// V4 API
cran_forceinline cv4 cv4_add(cv4 l, cv4 r);
cran_forceinline cv4 cv4_mulf(cv4 l, float r);
cran_forceinline float cv4_dot(cv4 l, cv4 r);

cran_forceinline ch4 cv4_f32_to_f16(cv4 f32);
cran_forceinline cv4 cv4_f16_to_f32(ch4 f16);

// Matrix API
cran_forceinline cm3 cm3_rotate_xy(float theta);
cran_forceinline cm3 cm3_from_basis(cv3 i, cv3 j, cv3 k);
cran_forceinline cm3 cm3_basis_from_normal(cv3 n);
cran_forceinline cv3 cm3_mul_cv3(cm3 m, cv3 v);
cran_forceinline cv3 cm3_rotate_cv3(cm3 m, cv3 v);
cran_forceinline cm3x4 cm3_to_cm3x4(cm3 m);
cran_forceinline cm4 cm3_to_cm4(cm3 m);
cran_forceinline cm3 cm3_transpose(cm3 m);
cran_forceinline cm3 cm3_inverse_orthonormal(cm3 m);

cran_forceinline cm4 cm3x4_to_cm4(cm3x4 m);

cran_forceinline cm4 cm4_rotate_xy(float theta);
cran_forceinline cm4 cm4_translate(cv3 pos);
cran_forceinline cm4 cm4_mul(cm4 l, cm4 r);
cran_forceinline cm4 cm4_perspective_projection(float fov, float nearPlane, float farPlane, float aspectRatio);
cran_forceinline cm4 cm4_inv_perspective_projection(float fov, float nearPlane, float farPlane, float aspectRatio);
cran_forceinline cm4 cm4_ortho_projection(cv2 extent, float nearPlane, float farPlane);

// AABB API
cran_forceinline bool caabb_does_ray_intersect(cv3 rayO, cv3 rayD, float rayMin, float rayMax, caabb aabb);
cran_forceinline bool caabb_does_line_intersect(cv3 a, cv3 b, caabb aabb);
cran_forceinline uint32_t caabb_does_ray_intersect_lanes(cv3 rayO, cv3 rayD, float rayMin, float rayMax, cv3l aabbMin, cv3l aabbMax);

// Quaternion API
cran_forceinline cm3 cq_to_cm3(cq q);
cran_forceinline cq cq_conjugate(cq q);
cran_forceinline cq cq_inverse(cq q);
cran_forceinline cq cq_axis_angle(cv3 axis, float angleRad);
cran_forceinline cq cq_mul(cq left, cq right);
cran_forceinline cv3 cq_rotate(cq rotator, cv3 point);
cran_forceinline bool cq_is_normalized(cq q);
cran_forceinline cq cq_euler(float yaw, float pitch, float roll);

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
	__m128 sse = _mm_rcp_ss(_mm_set_ss(f));
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

cran_forceinline cu16 cf_f32_to_f16(float f32)
{
	return (cu16)_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f32), _MM_FROUND_TO_NEAREST_INT));
}

cran_forceinline  float cf_f16_to_f32(cu16 f16)
{
	return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)f16)));
}

cran_forceinline float cf_mad(float a, float b, float c)
{
	__m128 sseA = _mm_set_ss(a);
	__m128 sseB = _mm_set_ss(b);
	__m128 sseC = _mm_set_ss(c);
	__m128 sseResult = _mm_fmadd_ps(sseA, sseB, sseC);
	float result;
	_mm_store_ss(&result, sseResult);
	return result;
}

cran_forceinline bool cf_close_enough(float a, float b, float t)
{
	return fabsf(a - b) < t;
}

cran_forceinline float cf_saturate(float f)
{
	return fminf(fmaxf(f, 0.0f), 1.0f);
}

cran_forceinline cu32 cf_compress_unorm(float f, uint32_t bitCount)
{
	float maxVal = (float)((1u << bitCount) - 1);
	return (cu32)roundf(f * maxVal);
}

cran_forceinline float cf_decompress_unorm(cu32 u, uint32_t bitCount)
{
	float invMaxVal = cf_rcp((float)((1u << bitCount) - 1));
	return (float)u * invMaxVal;
}

cran_forceinline uint32_t cf_compress_float_unsigned(float f, uint32_t exponentBitCount, uint32_t mantissaBitCount)
{
	if (!cf_finite(f) || exponentBitCount > cran_f32_exp_bits || mantissaBitCount > cran_f32_mantissa_bits || f < 0.0f)
	{
		return UINT32_MAX; // Panic and return max int.
	}

	uint32_t u = cf_f32_bits_to_u32(f);

	uint32_t nextMantissaBit = 1u << (cran_f32_mantissa_bits - mantissaBitCount - 1);
	u += (mantissaBitCount < cran_f32_mantissa_bits ? nextMantissaBit : 0); // Round by adding. We're unsigned so this is fine.

	int32_t exponent = (int32_t)((u >> cran_f32_mantissa_bits) & 0xFF) - cran_f32_exp_bias;
	
	int32_t exponentRange = 1 << exponentBitCount;
	if (abs(exponent) >= exponentRange / 2)
	{
		return UINT32_MAX; // Panic and return max int.
	}

	uint32_t mantissaMask = (1u << mantissaBitCount) - 1;
	uint32_t mantissa = (u >> (cran_f32_mantissa_bits - mantissaBitCount)) & mantissaMask;

	int32_t exponentBias = (exponentRange / 2) - 1;
	return ((uint32_t)(exponent + exponentBias) << mantissaBitCount) | mantissa;
}

cran_forceinline float cf_decompress_float_unsigned(uint32_t u, uint32_t exponentBitCount, uint32_t mantissaBitCount)
{
	int32_t biasedExponent = (int32_t)(u >> mantissaBitCount); // Assuming upper bits are zeroed out
	int32_t exponentRange = 1 << exponentBitCount;
	if (biasedExponent >= exponentRange)
	{
		return NAN;
	}

	int32_t exponentBias = (int32_t)(exponentRange / 2) - 1;
	int32_t exponent = (int32_t)biasedExponent - exponentBias;

	uint32_t mantissaMask = (1u << mantissaBitCount) - 1;
	uint32_t mantissa = u & mantissaMask;
	
	// Combine our exponent and mantissa
	return cf_u32_bits_to_f32(((uint32_t)(exponent + cran_f32_exp_bias) << cran_f32_mantissa_bits)
		| (mantissa << (cran_f32_mantissa_bits - mantissaBitCount)));
}

cran_forceinline uint32_t cf_f32_bits_to_u32(float v)
{
	union
	{
		uint32_t u;
		float f;
	} conv;
	conv.f = v;
	return conv.u;
}

cran_forceinline float cf_u32_bits_to_f32(uint32_t u)
{
	union
	{
		uint32_t u;
		float f;
	} conv;
	conv.u = u;
	return conv.f;
}

cran_forceinline cu32 cu_div_ceil(cu32 left, cu32 right)
{
	return (left + right - 1) / right;
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

cran_forceinline cv2 cv2_rcp(cv2 v)
{
	return (cv2) { cf_rcp(v.x), cf_rcp(v.y) };
}

cran_forceinline cv2 cv2_madf(cv2 v, float m, float a)
{
	return (cv2)
	{
		cf_mad(v.x, m, a),
		cf_mad(v.y, m, a)
	};
}

cran_forceinline ch2 cv2_f32_to_f16(cv2 f32)
{
	return (ch2){cf_f32_to_f16(f32.x), cf_f32_to_f16(f32.y)};
}

cran_forceinline cv2 cv2_f16_to_f32(ch2 f16)
{
	return (cv2){cf_f16_to_f32(f16.x), cf_f16_to_f32(f16.y)};
}

cran_forceinline cv2 cu2_to_cv2(cu2 v)
{
	return (cv2){ (float)v.x, (float)v.y };
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

cran_forceinline float cv3_fast_rlength(cv3 v)
{
	return cf_fast_rsqrt(v.x * v.x + v.y * v.y + v.z * v.z);
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
	float lengthSq = cv3_sqrlength(v);
	if (lengthSq < 0.0001f)
	{
		return (cv3) { 0 };
	}
	return cv3_mulf(v, cf_rsqrt(lengthSq));
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

cran_forceinline void cv3_to_spherical(cv3 v, float* cran_restrict phi, float* cran_restrict theta)
{
	float rlength = cv3_rlength(v);
	float azimuth = atan2f(v.y, v.x);
	*phi = (azimuth < 0.0f ? cran_tao + azimuth : azimuth);
	*theta = acosf(v.z * rlength);
}

cran_forceinline cv2 cv3_to_spherical_cv2(cv3 v)
{
	cv2 r;
	cv3_to_spherical(v, &r.x, &r.y);
	return r;
}

cran_forceinline cv3 cv3_from_spherical(float phi, float theta, float radius)
{
	return (cv3) { cosf(phi) * sinf(theta) * radius, sinf(phi) * sinf(theta) * radius, radius * cosf(theta) };
}

cran_forceinline cv3 cv3_barycentric(cv3 a, cv3 b, cv3 c, cv3 uvw)
{
	return cv3_add(cv3_add(cv3_mulf(a, uvw.x), cv3_mulf(b, uvw.y)), cv3_mulf(c, uvw.z));
}

// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
static cv2 octahedral_wrap( cv2 v )
{
	cv2 result;
	result.x = (1.0f - fabsf(v.y)) * (v.x >= 0.0f ? 1.0f : -1.0f);
	result.y = (1.0f - fabsf(v.x)) * (v.y >= 0.0f ? 1.0f : -1.0f);
	return result;
}

cran_forceinline cv2 cv3_to_octahedral(cv3 dir)
{
	dir = cv3_mulf(dir, cf_rcp( fabsf( dir.x ) + fabsf( dir.y ) + fabsf( dir.z ) ));
	cv2 results = dir.z >= 0.0f ? (cv2){dir.x, dir.y} : octahedral_wrap( (cv2){dir.x, dir.y} );
	results = cv2_madf(results, 0.5f, 0.5f);
	return results;
}

cran_forceinline cv3 cv3_from_octahedral(cv2 oct)
{
	oct = cv2_madf(oct, 2.0f, -1.0f);
 
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	cv3 n = (cv3){oct.x, oct.y, 1.0f - fabsf(oct.x) - fabsf(oct.y)};
	float t = cf_saturate( -n.z );
	n.x += n.x >= 0.0f ? -t : t;
	n.y += n.y >= 0.0f ? -t : t;
	return cv3_normalize( n );
}

cran_forceinline cv3 cv3_madf(cv3 v, float m, float a)
{
	return (cv3)
	{
		cf_mad(v.x, m, a),
		cf_mad(v.y, m, a),
		cf_mad(v.z, m, a),
	};
}

cran_forceinline ch3 cv3_f32_to_f16(cv3 f32)
{
	return (ch3){cf_f32_to_f16(f32.x), cf_f32_to_f16(f32.y), cf_f32_to_f16(f32.z)};
}

cran_forceinline cv3 cv3_f16_to_f32(ch3 f16)
{
	return (cv3){cf_f16_to_f32(f16.x), cf_f16_to_f32(f16.y), cf_f16_to_f32(f16.z)};
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
	for (uint32_t i = 0; i < indexCount && i < cran_lane_count; i++)
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

// V4 Implementation
cran_forceinline cv4 cv4_add(cv4 l, cv4 r)
{
	return (cv4){l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w};
}

cran_forceinline cv4 cv4_mulf(cv4 l, float r)
{
	return (cv4){l.x * r, l.y * r, l.z * r, l.w * r};
}

cran_forceinline float cv4_dot(cv4 l, cv4 r)
{
	return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
}

cran_forceinline ch4 cv4_f32_to_f16(cv4 f32)
{
	return (ch4)
	{
		cf_f32_to_f16(f32.x),
		cf_f32_to_f16(f32.y),
		cf_f32_to_f16(f32.z),
		cf_f32_to_f16(f32.w),
	};
}

cran_forceinline cv4 cv4_f16_to_f32(ch4 f16)
{
	return (cv4)
	{
		cf_f16_to_f32(f16.x),
		cf_f16_to_f32(f16.y),
		cf_f16_to_f32(f16.z),
		cf_f16_to_f32(f16.w),
	};
}

// Matrix Implementation
cran_forceinline cm3 cm3_rotate_xy(float theta)
{
	return (cm3)
	{
		cosf(theta), -sinf(theta), 0.0f,
		sinf(theta), cosf(theta), 0.0f,
		0.0f, 0.0f, 1.0f
	};
}

cran_forceinline cm3 cm3_from_basis(cv3 i, cv3 j, cv3 k)
{
	return (cm3)
	{
		i.x, j.x, k.x,
		i.y, j.y, k.y,
		i.z, j.z, k.z
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
	float rx = cv3_dot(m.v[0], v);
	float ry = cv3_dot(m.v[1], v);
	float rz = cv3_dot(m.v[2], v);

	return (cv3) { rx, ry, rz };
}

cran_forceinline cv3 cm3_rotate_cv3(cm3 m, cv3 v)
{
	return cm3_mul_cv3(m, v);
}

cran_forceinline cm3x4 cm3_to_cm3x4(cm3 m)
{
	return (cm3x4)
	{
		m.f[0], m.f[1], m.f[2], 0.0f,
		m.f[3], m.f[4], m.f[5], 0.0f,
		m.f[6], m.f[7], m.f[8], 0.0f,
	};
}

cran_forceinline cm4 cm3_to_cm4(cm3 m)
{
	return (cm4)
	{
		m.f[0], m.f[1], m.f[2], 0.0f,
		m.f[3], m.f[4], m.f[5], 0.0f,
		m.f[6], m.f[7], m.f[8], 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
	};
}

cran_forceinline cm3 cm3_transpose(cm3 m)
{
	return (cm3)
	{
		m.f[0], m.f[3], m.f[6],
		m.f[1], m.f[4], m.f[7],
		m.f[2], m.f[5], m.f[8],
	};
}

cran_forceinline cm3 cm3_inverse_orthonormal(cm3 m)
{
	return cm3_transpose(m);
}

cran_forceinline cm4 cm3x4_to_cm4(cm3x4 m)
{
	return (cm4)
	{
		m.f[0], m.f[1], m.f[2],  m.f[3],
		m.f[4], m.f[5], m.f[6],  m.f[7],
		m.f[8], m.f[9], m.f[10], m.f[11],
		0.0f, 0.0f, 0.0f, 1.0f,
	};
}

cran_forceinline cm4 cm4_rotate_xy(float theta)
{
	return (cm4)
	{
		cosf(theta), -sinf(theta), 0.0f, 0.0f,
		sinf(theta), cosf(theta), 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};
}

cran_forceinline cm4 cm4_translate(cv3 pos)
{
	cm4 m = cm4_identity;
	m.v[0].w = pos.x;
	m.v[1].w = pos.y;
	m.v[2].w = pos.z;
	return m;
}

cran_forceinline cm4 cm4_mul(cm4 l, cm4 r)
{
	cm4 result;
	for(uint32_t row = 0; row < 4; row++)
	{
		cv4 r0 = cv4_mulf(r.v[0], l.v[row].x);
		cv4 r1 = cv4_mulf(r.v[1], l.v[row].y);
		cv4 r2 = cv4_mulf(r.v[2], l.v[row].z);
		cv4 r3 = cv4_mulf(r.v[3], l.v[row].w);

		result.v[row] = cv4_add(cv4_add(r0, r1), cv4_add(r2, r3));
	}
	
	return result;
}

cran_forceinline cm4 cm4_perspective_projection(float fov, float nearPlane, float farPlane, float aspectRatio)
{
	return (cm4){ fov / aspectRatio, 0.0f, 0.0f, 0.0f,
                0.0f, fov, 0.0f, 0.0f,
                0.0f, 0.0f, farPlane / (farPlane - nearPlane), -nearPlane * farPlane / (farPlane - nearPlane),
                0.0f, 0.0f, 1.0, 0.0f};
}

cran_forceinline cm4 cm4_inv_perspective_projection(float fov, float nearPlane, float farPlane, float aspectRatio)
{
	return (cm4){ aspectRatio / fov, 0.0f, 0.0f, 0.0f,
                0.0f, -1.0f / fov, 0.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f,
                0.0f, 0.0f, (farPlane - nearPlane) / (-nearPlane * farPlane), farPlane / (nearPlane * farPlane)};
}

inline cran_forceinline cm4 cm4_ortho_projection(cv2 extent, float nearPlane, float farPlane)
{
	float depthRange = farPlane-nearPlane;
	return (cm4){1.0f / extent.x, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f / extent.y, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f/depthRange, -nearPlane/depthRange,
				0.0f, 0.0f, 0.0, 1.0f};
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
	return (l.max.f[axis] + l.min.f[axis]) * 0.5f;
}

cran_forceinline float caabb_side(caabb l, uint32_t axis)
{
	return l.max.f[axis] - l.min.f[axis];
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

// Quaternion Implementation
cran_forceinline cm3 cq_to_cm3(cq q)
{
	cv3 right = cq_rotate(q, (cv3){ 1.0f, 0.0f, 0.0f });
    cv3 forward = cq_rotate(q, (cv3){ 0.0f, 1.0f, 0.0f });
    cv3 up = cq_rotate(q, (cv3){ 0.0f, 0.0f, 1.0f });
    return (cm3)
    {
        right.x, forward.x, up.x,
        right.y, forward.y, up.y,
        right.z, forward.z, up.z,
    };
}

cran_forceinline cq cq_conjugate(cq q)
{
	return (cq){ -q.x, -q.y, -q.z, q.w };
}

cran_forceinline cq cq_inverse(cq q)
{
	return cq_conjugate(q);
}

cran_forceinline cq cq_axis_angle(cv3 axis, float angleRad)
{
    cv3 real = cv3_mulf(axis, sinf(angleRad * 0.5f));
    return (cq){ real.x, real.y, real.z, cosf(angleRad * 0.5f) };
}

cran_forceinline cq cq_axis_angle_cosAngle(cv3 axis, float cosAngle)
{
	float sinHalfAngle = sqrtf(cf_mad(cosAngle, -0.5f, 0.5f));
	float cosHalfAngle = sqrtf(cf_mad(cosAngle, 0.5f, 0.5f));
    cv3 real = cv3_mulf(axis, sinHalfAngle);
    return (cq){ real.x, real.y, real.z, cosHalfAngle };
}

cran_forceinline cq cq_mul(cq left, cq right)
{
	return (cq)
    {
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
    };
}

cran_forceinline cv3 cq_rotate(cq rotator, cv3 point)
{
    cq rotatedPoint = cq_mul(cq_mul(rotator, (cq){ point.x, point.y, point.z, 0.0f }), cq_inverse(rotator));
    return (cv3){ rotatedPoint.x, rotatedPoint.y, rotatedPoint.z };
}

static float cq_norm_squared(cq q)
{
	return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
}

static cq cq_mulf(cq q, float f)
{
	return (cq){q.x * f, q.y * f, q.z * f, q.w * f};
}

cran_forceinline bool cq_is_normalized(cq q)
{
	float squaredNorm = cq_norm_squared(q);
	return cf_close_enough(squaredNorm, 1.0f, 1.0f / 65536.0f);
}

cran_forceinline cq cq_normalize(cq q)
{
	float squaredNorm = cq_norm_squared(q);
	if(cf_close_enough(squaredNorm, 0.0f, 1.0f / 65536.0f))
	{
		return cq_identity;
	}
	return cq_mulf(q, cf_fast_rsqrt(squaredNorm));
}

cran_forceinline cq cq_euler(float yaw, float pitch, float roll)
{
	cq const yawQuaternion = cq_axis_angle(cv3_up, yaw);
	cq const pitchQuaternion = cq_axis_angle(cv3_right, pitch);
	cq const rollQuaternion = cq_axis_angle(cv3_forward, roll);

	return cq_mul(cq_mul(yawQuaternion, pitchQuaternion), rollQuaternion);
}