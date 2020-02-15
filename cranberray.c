#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "cranberry_platform.h"

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
			vec3 h = vec3_sub(circleOrigin, origin);
			float distanceToRaySqr = vec3_dot(h,h) - vec3_dot(h, rayDir) * vec3_dot(h, rayDir);

			int32_t imgIdx = ((y + imgHeight / 2) * imgWidth + (x + imgWidth / 2)) * imgStride;
			bitmap[imgIdx + 0] = (distanceToRaySqr < circleRad * circleRad) ? 0xFF : 0x00;
			bitmap[imgIdx + 1] = (distanceToRaySqr < circleRad * circleRad) ? 0xFF : 0x00;
			bitmap[imgIdx + 2] = (distanceToRaySqr < circleRad * circleRad) ? 0xFF : 0x00;
			bitmap[imgIdx + 3] = 0xFF;
		}
	}

	cranpl_write_bmp("render.bmp", bitmap, imgWidth, imgHeight);

	return 0;
}
