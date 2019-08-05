#ifndef __CRANBERRY_MATH_H
#define __CRANBERRY_MATH_H

#include <math.h>

#ifdef CRANM_SSE
#include <immintrin.h>
#include <emmintrin.h>

#define cranm_shuffle_sse(a, b) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(a), b))

#endif // CRANM_SSE

#ifdef CRANM_DEBUG_SLOW
#include <assert.h>
#endif // CRANM_DEBUG_SLOW

#define cranm_pi 3.14159265358979323846f

// Types

typedef struct
{
	float x, y, z, w;
} cranm_vec_t;

typedef struct
{
	float x, y, z, w;
} cranm_quat_t;

typedef struct
{
	cranm_quat_t rot;
	cranm_vec_t pos;
	float scale;
} cranm_transform_t;

typedef struct
{
	float m[16];
} cranm_mat4x4_t;

// API

inline cranm_vec_t cranm_add3(cranm_vec_t l, cranm_vec_t r);
inline cranm_vec_t cranm_sub3(cranm_vec_t l, cranm_vec_t r);
inline cranm_vec_t cranm_scale(cranm_vec_t l, float s);
inline cranm_vec_t cranm_scale3(cranm_vec_t l, cranm_vec_t r);
inline cranm_vec_t cranm_cross(cranm_vec_t l, cranm_vec_t r);
inline cranm_vec_t cranm_normalize3(cranm_vec_t v);
inline cranm_vec_t cranm_recriprocal3(cranm_vec_t v);

inline cranm_vec_t cranm_quat_t_xyz(cranm_quat_t q);
inline cranm_quat_t cranm_axis_angleq(cranm_vec_t axis, float angle);
inline cranm_quat_t cranm_mulq(cranm_quat_t l, cranm_quat_t r);
inline cranm_quat_t cranm_inverse_mulq(cranm_quat_t l, cranm_quat_t r);
inline cranm_quat_t cranm_inverseq(cranm_quat_t q);
inline cranm_vec_t cranm_rot3(cranm_vec_t v, cranm_quat_t r);
inline cranm_vec_t cranm_inverse_rot3(cranm_vec_t v, cranm_quat_t r);

inline cranm_mat4x4_t cranm_identity4x4();
inline cranm_mat4x4_t cranm_mul4x4(cranm_mat4x4_t l, cranm_mat4x4_t r);
inline cranm_mat4x4_t cranm_perspective(float near, float far, float fov);

inline cranm_transform_t cranm_transform(cranm_transform_t t, cranm_transform_t by);
inline cranm_transform_t cranm_inverse_transform(cranm_transform_t t, cranm_transform_t by);

// IMPL

inline cranm_vec_t cranm_add3(cranm_vec_t l, cranm_vec_t r)
{
#ifdef CRANM_SSE
	__m128 lv = _mm_load_ps((float*)&l);
	__m128 rv = _mm_load_ps((float*)&r);

	cranm_vec_t result;
	_mm_store_ps((float*)&result, _mm_add_ps(lv, rv));
	return result;
#else
	return (cranm_vec_t) { .x = l.x + r.x, l.y + r.y, l.z + r.z };
#endif // CRANM_SSE
}

inline cranm_vec_t cranm_sub3(cranm_vec_t l, cranm_vec_t r)
{
#ifdef CRANM_SSE
	__m128 lv = _mm_load_ps((float*)&l);
	__m128 rv = _mm_load_ps((float*)&r);

	cranm_vec_t result;
	_mm_store_ps((float*)&result, _mm_sub_ps(lv, rv));
	return result;
#else
	return (cranm_vec_t) { .x = l.x - r.x, l.y - r.y, l.z - r.z };
#endif // CRANM_SSE
}

inline cranm_vec_t cranm_scale(cranm_vec_t l, float s)
{
#ifdef CRANM_SSE
	__m128 sv = _mm_set1_ps(s);
	__m128 lv = _mm_load_ps((float*)&l);

	cranm_vec_t result;
	_mm_store_ps((float*)&result, _mm_mul_ps(sv, lv));
	return result;
#else
	return (cranm_vec_t) { .x = l.x * s, .y = l.y * s, .z = l.z * s };
#endif // CRANM_SSE
}

inline cranm_vec_t cranm_scale3(cranm_vec_t l, cranm_vec_t r)
{
#ifdef CRANM_SSE
	__m128 lv = _mm_load_ps((float*)&l);
	__m128 rv = _mm_load_ps((float*)&r);

	cranm_vec_t result;
	_mm_store_ps((float*)&result, _mm_mul_ps(lv, rv));
	return result;
#else
	return (cranm_vec_t) { .x = l.x * r.x, .y = l.y * r.y, .z = l.z * r.z };
#endif // CRANM_SSE
}

inline cranm_vec_t cranm_cross(cranm_vec_t l, cranm_vec_t r)
{
#ifdef CRANM_SSE
	__m128 lv = _mm_load_ps((float*)&l);
	__m128 rv = _mm_load_ps((float*)&r);

	__m128 l1 = cranm_shuffle_sse(lv, _MM_SHUFFLE(0, 0, 2, 1));
	__m128 l2 = cranm_shuffle_sse(lv, _MM_SHUFFLE(0, 1, 0, 2));

	__m128 r1 = cranm_shuffle_sse(rv, _MM_SHUFFLE(0, 1, 0, 2));
	__m128 r2 = cranm_shuffle_sse(rv, _MM_SHUFFLE(0, 0, 2, 1));
	
	__m128 lm = _mm_mul_ps(l1, r1);
	__m128 rm = _mm_mul_ps(l2, r2);
	cranm_vec_t result;
	_mm_store_ps((float*)&result, _mm_sub_ps(lm, rm));

	return result;
#else
	return (cranm_vec_t) 
	{ 
		.x = l.y * r.z - l.z * r.y,
		.y = l.z * r.x - l.x * r.z,
		.z = l.x * r.y - l.y * r.x
	};
#endif // CRANM_SSE
}

inline cranm_vec_t cranm_normalize3(cranm_vec_t v)
{
	float rm = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return (cranm_vec_t) { .x = v.x * rm, .y = v.y * rm, .z = v.z * rm };
}

inline cranm_vec_t cranm_recriprocal3(cranm_vec_t v)
{
#ifdef CRANM_SSE
	__m128 lv = _mm_load_ps((float*)&v);

	cranm_vec_t result;
	_mm_store_ps((float*)&result, _mm_rcp_ps(lv));
	return result;
#else
	return (cranm_vec_t) { .x = 1.0f / v.x, .y = 1.0f / v.y, .z = 1.0f / v.z };
#endif // CRANM_SSE
}

inline cranm_vec_t cranm_quat_t_xyz(cranm_quat_t q)
{
	return (cranm_vec_t) { .x = q.x, .y = q.y, .z = q.z, .w = 0.0f };
}

inline cranm_quat_t cranm_mulq(cranm_quat_t l, cranm_quat_t r)
{
#ifdef CRANM_SSE
	__m128 q = _mm_load_ps((float*)&r);
	__m128 s = _mm_load_ps((float*)&l);

	__m128 w = cranm_shuffle_sse(s, _MM_SHUFFLE(3, 3, 3, 3));
	__m128 x = cranm_shuffle_sse(s, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 y = cranm_shuffle_sse(s, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 z = cranm_shuffle_sse(s, _MM_SHUFFLE(2, 2, 2, 2));

	__m128 rw = _mm_mul_ps(w, q);
	__m128 rx = _mm_mul_ps(x, cranm_shuffle_sse(q, _MM_SHUFFLE(0, 1, 2, 3)));
	__m128 ry = _mm_mul_ps(y, cranm_shuffle_sse(q, _MM_SHUFFLE(1, 0, 3, 2)));
	__m128 rz = _mm_mul_ps(z, cranm_shuffle_sse(q, _MM_SHUFFLE(2, 3, 0, 1)));

	__m128 f = _mm_add_ps(rw, _mm_xor_ps(rx, _mm_set_ps(-0.0f, -0.0f, 0.0f, 0.0f)));
	f = _mm_add_ps(f, _mm_xor_ps(ry, _mm_set_ps(-0.0f, 0.0f, 0.0f, -0.0f)));
	f = _mm_add_ps(f, _mm_xor_ps(rz, _mm_set_ps(-0.0f, 0.0f, -0.0f, 0.0f)));

	cranm_quat_t result;
	_mm_store_ps((float*)&result, f);

#ifdef CRANM_DEBUG_SLOW
	cranm_quat_t test = 
	{
		.x = l.w * r.x + l.x * r.w - l.y * r.z + l.z * r.y,
		.y = l.w * r.y + l.x * r.z + l.y * r.w - l.z * r.x,
		.z = l.w * r.z - l.x * r.y + l.y * r.x + l.z * r.w,
		.w = l.w * r.w - l.x * r.x - l.y * r.y - l.z * r.z
	};

	assert(result.x == test.x && result.y == test.y && result.z == test.z && result.w == test.w);
#endif // CRANM_DEBUG_SLOW

	return result;
#else
	return (cranm_quat_t)
	{
		.x = l.w * r.x + l.x * r.w - l.y * r.z + l.z * r.y,
		.y = l.w * r.y + l.x * r.z + l.y * r.w - l.z * r.x,
		.z = l.w * r.z - l.x * r.y + l.y * r.x + l.z * r.w,
		.w = l.w * r.w - l.x * r.x - l.y * r.y - l.z * r.z
	};
#endif // CRANM_SSE
}

inline cranm_quat_t cranm_inverse_mulq(cranm_quat_t l, cranm_quat_t r)
{
#ifdef CRANM_SSE
	__m128 q = _mm_load_ps((float*)&r);
	__m128 s = _mm_load_ps((float*)&l);

	__m128 w = cranm_shuffle_sse(s, _MM_SHUFFLE(3, 3, 3, 3));
	w = _mm_xor_ps(w, _mm_set_ps(0.0f, -0.0f, -0.0f, -0.0f));

	__m128 x = cranm_shuffle_sse(s, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 y = cranm_shuffle_sse(s, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 z = cranm_shuffle_sse(s, _MM_SHUFFLE(2, 2, 2, 2));

	__m128 rw = _mm_mul_ps(w, q);
	__m128 rx = _mm_mul_ps(x, cranm_shuffle_sse(q, _MM_SHUFFLE(0, 1, 2, 3)));
	__m128 ry = _mm_mul_ps(y, cranm_shuffle_sse(q, _MM_SHUFFLE(1, 0, 3, 2)));
	__m128 rz = _mm_mul_ps(z, cranm_shuffle_sse(q, _MM_SHUFFLE(2, 3, 0, 1)));

	__m128 f = _mm_add_ps(rw, _mm_xor_ps(rx, _mm_set_ps(0.0f, 0.0f, -0.0f, 0.0f)));
	f = _mm_add_ps(f, _mm_xor_ps(ry, _mm_set_ps(0.0f, -0.0f, 0.0f, 0.0f)));
	f = _mm_add_ps(f, _mm_xor_ps(rz, _mm_set_ps(0.0f, 0.0f, 0.0f, -0.0f)));

	cranm_quat_t result;
	_mm_store_ps((float*)&result, f);

#ifdef CRANM_DEBUG_SLOW
	cranm_quat_t test =
	{
		.x = -l.w * r.x + l.x * r.w + l.y * r.z - l.z * r.y,
		.y = -l.w * r.y - l.x * r.z + l.y * r.w + l.z * r.x,
		.z = -l.w * r.z + l.x * r.y - l.y * r.x + l.z * r.w,
		.w =  l.w * r.w + l.x * r.x + l.y * r.y + l.z * r.z
	};

	assert(result.x == test.x && result.y == test.y && result.z == test.z && result.w == test.w);
#endif // CRANM_DEBUG_SLOW

	return result;
#else
	return (cranm_quat_t)
	{
		.x = -l.w * r.x + l.x * r.w + l.y * r.z - l.z * r.y,
		.y = -l.w * r.y - l.x * r.z + l.y * r.w + l.z * r.x,
		.z = -l.w * r.z + l.x * r.y - l.y * r.x + l.z * r.w,
		.w =  l.w * r.w + l.x * r.x + l.y * r.y + l.z * r.z
	};
#endif // CRANM_SSE
}

inline cranm_quat_t cranm_axis_angleq(cranm_vec_t axis, float angle)
{
	float cr = cosf(angle * 0.5f);
	float sr = sinf(angle * 0.5f);
	return (cranm_quat_t) { .w = cr, .x = axis.x * sr, .y = axis.y * sr, .z = axis.z * sr };
}

inline cranm_quat_t cranm_inverseq(cranm_quat_t q)
{
	return (cranm_quat_t) { .x = -q.x, .y = -q.y, .z = -q.z, .w = q.w };
}

inline cranm_vec_t cranm_rot3(cranm_vec_t v, cranm_quat_t r)
{
	cranm_vec_t t = cranm_scale(cranm_quat_t_xyz(r), 2.0f);
	t = cranm_cross(t, v);

	cranm_vec_t res = cranm_add3(v, cranm_scale(t, r.w));
	return cranm_add3(res, cranm_cross(cranm_quat_t_xyz(r), t));
}

inline cranm_vec_t cranm_inverse_rot3(cranm_vec_t v, cranm_quat_t r)
{
	cranm_vec_t t = cranm_scale(cranm_quat_t_xyz(r), -2.0f);
	t = cranm_cross(t, v);

	cranm_vec_t res = cranm_add3(v, cranm_scale(t, r.w));
	return cranm_add3(res, cranm_cross(cranm_scale(cranm_quat_t_xyz(r), -1.0f), t));
}

inline cranm_mat4x4_t cranm_identity4x4()
{
	cranm_mat4x4_t mat;
	mat.m[0] = mat.m[5] = mat.m[10] = mat.m[15] = 1.0f;
	return mat;
}

inline cranm_mat4x4_t cranm_mul4x4(cranm_mat4x4_t l, cranm_mat4x4_t r)
{
	cranm_mat4x4_t mat;

	for (unsigned int i = 0; i < 4; ++i)
	{
		float l0 = l.m[i * 4 + 0];
		float l1 = l.m[i * 4 + 1];
		float l2 = l.m[i * 4 + 2];
		float l3 = l.m[i * 4 + 3];

		mat.m[i * 4 + 0] = l0 * r.m[0] + l1 * r.m[4] + l2 * r.m[8] + l3 * r.m[12];
		mat.m[i * 4 + 1] = l0 * r.m[1] + l1 * r.m[5] + l2 * r.m[9] + l3 * r.m[13];
		mat.m[i * 4 + 2] = l0 * r.m[2] + l1 * r.m[6] + l2 * r.m[10] + l3 * r.m[14];
		mat.m[i * 4 + 3] = l0 * r.m[3] + l1 * r.m[7] + l2 * r.m[11] + l3 * r.m[15];
	}

	return mat;
}

inline cranm_mat4x4_t cranm_perspective(float near, float far, float fov)
{
	float s = 1.0f / tanf(fov * 3.14159265358979f / 360.0f);

	cranm_mat4x4_t mat = { 0 };
	mat.m[0] = s;
	mat.m[5] = s;
	mat.m[10] = far / (far - near);
	mat.m[11] = 1.0f;
	mat.m[14] = -far * near / (far - near);

	return mat;
}

inline cranm_transform_t cranm_transform(cranm_transform_t t, cranm_transform_t by)
{
	return (cranm_transform_t)
	{
		.rot = cranm_mulq(t.rot, by.rot),
		.pos = cranm_add3(cranm_rot3(cranm_scale(t.pos, by.scale), by.rot), by.pos),
		.scale = t.scale * by.scale
	};
}

inline cranm_transform_t cranm_inverse_transform(cranm_transform_t t, cranm_transform_t by)
{
	float inverseScale = 1.0f / by.scale;

	return (cranm_transform_t)
	{
		.rot = cranm_inverse_mulq(t.rot, by.rot),
		.pos = cranm_scale(cranm_inverse_rot3(cranm_sub3(t.pos, by.pos), by.rot), inverseScale),
		.scale = t.scale * inverseScale
	};
}

#endif // __CRANBERRY_MATH_H
