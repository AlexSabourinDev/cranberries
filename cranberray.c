// Links and things:
// https://patapom.com/blog/BRDF/BRDF%20Models/

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

#include "cranberry_platform.h"

// Forward decls
void* memset(void* dst, int val, size_t size);

const float PI = 3.14159265358979323846264338327f;

// float math
static float rcp(float f)
{
	return 1.0f / f;
}

static float rsqrt(float f)
{
	return 1.0f / sqrtf(f);
}

static void quadratic(float a, float b, float c, float* out1, float* out2)
{
	// TODO: Replace with more numerically robust version.
	float d = sqrtf(b*b - 4.0f * a * c);
	float e = rcp(2.0f * a);

	*out1 = (-b - d) * e;
	*out2 = (-b + d) * e;
}

static float randf01()
{
	return (float)rand() / (float)RAND_MAX;
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

static vec3 vec3_sub(vec3 l, vec3 r)
{
	return (vec3) {.x = l.x - r.x, .y = l.y - r.y, .z = l.z - r.z};
}

static vec3 vec3_mul(vec3 l, vec3 r)
{
	return (vec3) {.x = l.x * r.x, .y = l.y * r.y, .z = l.z * r.z};
}

static float vec3_dot(vec3 l, vec3 r)
{
	return l.x * r.x + l.y * r.y + l.z * r.z;
}

static vec3 vec3_lerp(vec3 l, vec3 r, float t)
{
	return vec3_add(vec3_mulf(l, 1.0f - t), vec3_mulf(r, t));
}

static vec3 vec3_normalized(vec3 v)
{
	return vec3_mulf(v, rsqrt(v.x * v.x + v.y * v.y + v.z * v.z));
}

static bool doesSphereRayIntersect(vec3 rayO, vec3 rayD, vec3 sphereO, float sphereR)
{
	vec3 sphereRaySpace = vec3_sub(sphereO, rayO);
	float projectedDistance = vec3_dot(sphereRaySpace, rayD);
	float distanceToRaySqr = vec3_dot(sphereRaySpace,sphereRaySpace) - projectedDistance * projectedDistance;
	return (distanceToRaySqr < sphereR * sphereR);
}

static float sphereRayIntersection(vec3 rayO, vec3 rayD, float rayMin, float rayMax, vec3 sphereO, float sphereR)
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
	quadratic(a, b, c, &d1, &d2);

	// Get our closest point
	float d = d1 < d2 ? d1 : d2;
	if (d > rayMin && d < rayMax)
	{
		return d;
	}

	return rayMax;
}

static vec3 randomInUnitSphere()
{
	vec3 p;
	do
	{
		p = vec3_mulf((vec3) { randf01()-0.5f, randf01()-0.5f, randf01()-0.5f }, 2.0f);
	} while (vec3_dot(p, p) <= 1.0f);

	return p;
}

vec3 castScene(vec3 rayO, vec3 rayD)
{
	const float NoRayIntersection = FLT_MAX;

	// TODO: This recast bias is simply to avoid re-intersecting with our object when casting.
	// Do we want to handle this some other way?
	const float ReCastBias = 0.0001f;

	vec3 sceneColor = { 0 };

	// TODO: Refine our scene description
	#define circleCount 2
	vec3 circleOrigins[circleCount] = { {.x = 0.0f,.y = 0.0f,.z = 2.0f }, {.x = 0.0f,.y = -10.5f,.z = 2.0f} };
	float circleRads[circleCount] = { 0.5f, 10.0f };

	float closestDistance = NoRayIntersection;
	float closestRadius = 0.0f;
	vec3 closestCircle = { 0 };
	for (uint32_t i = 0; i < circleCount; i++)
	{
		if (doesSphereRayIntersect(rayO, rayD, circleOrigins[i], circleRads[i]))
		{
			float intersectionDistance = sphereRayIntersection(rayO, rayD, ReCastBias, NoRayIntersection, circleOrigins[i], circleRads[i]);
			// TODO: Do we want to handle tie breakers somehow?
			if (intersectionDistance < closestDistance)
			{
				closestDistance = intersectionDistance;
				closestRadius = circleRads[i];
				closestCircle = circleOrigins[i];
			}
		}
	}

	if (closestDistance != NoRayIntersection)
	{
		vec3 intersectionPoint = vec3_add(rayO, vec3_mulf(rayD, closestDistance));
		vec3 surfacePoint = vec3_sub(intersectionPoint, closestCircle);
		vec3 normal = vec3_mulf(surfacePoint, rcp(closestRadius));

		vec3 spherePoint = vec3_add(intersectionPoint, normal);
		spherePoint = vec3_add(normal, randomInUnitSphere());

		vec3 sceneCast = castScene(intersectionPoint, spherePoint);

		// TODO: lighting be like
		return vec3_add(vec3_mulf(sceneCast, 0.5f), sceneColor);
	}

	rayD = vec3_normalized(rayD);
	return vec3_lerp((vec3){0.5f, 0.7f, 1.0f}, (vec3){ 1.0f, 1.0f, 1.0f }, rayD.y * 0.5f + 0.5f);
}

int main()
{
	int32_t imgWidth = 2048, imgHeight = 1024, imgStride = 4;

	// TODO: How do we want to express our camera?
	// Currently simply using the near triangle.
	float near = 1.0f, nearHeight = 2.0f, nearWidth = nearHeight * (float)imgWidth / (float)imgHeight;

	vec3 origin = { 0 };
	vec3 forward = { .x = 0.0f,.y = 0.0f,.z = 1.0f }, right = { .x = 1.0f,.y = 0.0f,.z = 0.0f }, up = { .x = 0.0f, .y = 1.0f, .z = 0.0f };

	float* hdrImage = malloc(imgWidth * imgHeight * imgStride * sizeof(float));

	// Sample our scene for every pixel in the bitmap. (Could be upsampled if we wanted to)
	float xStep = nearWidth / (float)imgWidth, yStep = nearHeight / (float)imgHeight;
	for (int32_t y = -imgHeight / 2; y < imgHeight / 2; y++)
	{
		float yOff = yStep * (float)y;
		for (int32_t x = -imgWidth / 2; x < imgWidth / 2; x++)
		{
			float xOff = xStep * (float)x;
			// Construct our ray as a vector going from our origin to our near plane
			// V = F*n + R*ix*worldWidth/imgWidth + U*iy*worldHeight/imgHeight
			vec3 rayDir = vec3_add(vec3_mulf(forward, near), vec3_add(vec3_mulf(right, xOff), vec3_mulf(up, yOff)));

			// The length of our vector (L) is:
			// L = sqrt(B^2 + yOff^2)
			// B = the length of the hypothenuse of the base triangle formed by forward + right.
			// B = sqrt(near^2 + xOff^2)
			// Substitute
			// L = sqrt(sqrt(near^2 + xOff^2)^2 + yOff^2)
			// Simplify
			// L = sqrt(near^2 + xOff^2 + yOff^2)
			rayDir = vec3_mulf(rayDir, rsqrt(near * near + xOff * xOff + yOff * yOff));

			vec3 sceneColor = castScene(origin, rayDir);

			int32_t imgIdx = ((y + imgHeight / 2) * imgWidth + (x + imgWidth / 2)) * imgStride;
			hdrImage[imgIdx + 0] = sceneColor.x;
			hdrImage[imgIdx + 1] = sceneColor.y;
			hdrImage[imgIdx + 2] = sceneColor.z;
			hdrImage[imgIdx + 3] = 1.0f;
		}
	}

	// Convert HDR to 8 bit bitmap
	{
		uint8_t* bitmap = malloc(imgWidth * imgHeight * imgStride);
		for (int32_t i = 0; i < imgWidth * imgHeight * imgStride; i+=4)
		{
			bitmap[i + 0] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 2]));
			bitmap[i + 1] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 1]));
			bitmap[i + 2] = (uint8_t)(255.99f * sqrtf(hdrImage[i + 0]));
			bitmap[i + 3] = (uint8_t)(255.99f * hdrImage[i + 3]);
		}

		cranpl_write_bmp("render.bmp", bitmap, imgWidth, imgHeight);
		free(bitmap);
	}

	free(hdrImage);
	return 0;
}
