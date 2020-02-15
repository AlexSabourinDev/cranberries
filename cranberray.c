#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "cranberry_platform.h"

// Forward decls
void* memset(void* dst, int val, size_t size);

const float PI = 3.14159265358979323846264338327f;

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

static float vec3_dot(vec3 l, vec3 r)
{
	return l.x * r.x + l.y * r.y + l.z * r.z;
}

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
	float e = 1.0f / (2.0f * a);

	*out1 = (-b - d) * e;
	*out2 = (-b + d) * e;
}

int main()
{
	int32_t imgWidth = 2048, imgHeight = 1024, imgStride = 4;

	// TODO: How do we want to express our camera?
	// Currently simply using the near triangle.
	float near = 1.0f, nearHeight = 1.0f, nearWidth = nearHeight * (float)imgWidth / (float)imgHeight;

	vec3 origin = { 0 };
	vec3 forward = { .x = 0.0f,.y = 0.0f,.z = 1.0f }, right = { .x = 1.0f,.y = 0.0f,.z = 0.0f }, up = { .x = 0.0f, .y = 1.0f, .z = 0.0f };

	uint8_t* bitmap = malloc(imgWidth * imgHeight * imgStride);

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

			// TODO: Refine our scene description
			vec3 circleOrigin = { .x = 0.0f,.y = 0.0f,.z = 2.0f };
			float circleRad = 0.5f;

			// TODO: We'll definitely need intersection information when we're shading.
			vec3 circleRaySpace = vec3_sub(circleOrigin, origin);
			float distanceToRaySqr = vec3_dot(circleRaySpace,circleRaySpace) - vec3_dot(circleRaySpace, rayDir) * vec3_dot(circleRaySpace, rayDir);

			int32_t imgIdx = ((y + imgHeight / 2) * imgWidth + (x + imgWidth / 2)) * imgStride;
			if (distanceToRaySqr < circleRad * circleRad)
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
				vec3 rayCircleSpace = vec3_sub(origin, circleOrigin);
				float a = rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z;
				float b = 2.0f * rayDir.x * rayCircleSpace.x + 2.0f * rayDir.y * rayCircleSpace.y + 2.0f * rayDir.z * rayCircleSpace.z;
				float c = rayCircleSpace.x * rayCircleSpace.x + rayCircleSpace.y * rayCircleSpace.y + rayCircleSpace.z * rayCircleSpace.z - circleRad * circleRad;

				float d1, d2;
				quadratic(a, b, c, &d1, &d2);

				// Get our closest point
				float d = d1 < d2 ? d1 : d2;
				vec3 intersectionPoint = vec3_add(vec3_mulf(rayDir, d), origin);

				// Render the absolute normals of the sphere for now.
				vec3 normal = vec3_mulf(vec3_sub(intersectionPoint, circleOrigin), rcp(circleRad));
				bitmap[imgIdx + 0] = (uint8_t)fabs(255.0f * normal.x);
				bitmap[imgIdx + 1] = (uint8_t)fabs(255.0f * normal.y);
				bitmap[imgIdx + 2] = (uint8_t)fabs(255.0f * normal.z);
				bitmap[imgIdx + 3] = 0xFF;
			}
			else
			{
				memset(bitmap + imgIdx, 0x00, imgStride);
			}
		}
	}

	cranpl_write_bmp("render.bmp", bitmap, imgWidth, imgHeight);

	return 0;
}
