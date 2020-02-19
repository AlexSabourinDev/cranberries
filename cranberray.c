// Links and things:
// https://patapom.com/blog/BRDF/BRDF%20Models/
// https://www.realtimerendering.com/raytracing/Ray%20Tracing%20in%20a%20Weekend.pdf

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

static bool quadratic(float a, float b, float c, float* __restrict out1, float* __restrict out2)
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

static float rand01f()
{
	return (float)rand() / (float)RAND_MAX;
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

static vec3 randomInUnitSphere()
{
	vec3 p;
	do
	{
		p = vec3_mulf((vec3) { rand01f()-0.5f, rand01f()-0.5f, rand01f()-0.5f }, 2.0f);
	} while (vec3_dot(p, p) <= 1.0f);

	return p;
}

static vec3 vec3_reflect(vec3 i, vec3 n)
{
	return vec3_sub(i, vec3_mulf(n, 2.0f * vec3_dot(i, n)));
}

// a is between 0 and 2 PI
// t is between 0 and PI (0 being the bottom, PI being the top)
static void vec3_to_spherical(vec3 v, float* __restrict a, float* __restrict t)
{
	assert(a != t);

	float rlenght = vec3_rlength(v);
	float azimuth = atan2f(v.z, v.x);
	*a = (azimuth < 0.0f ? TAO + azimuth : azimuth);
	*t = acosf(v.y * rlenght);
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
	void* __restrict materials[material_count];
	void* __restrict renderables[renderable_count];

	struct
	{
		instance_t* data;
		uint32_t count;
	} instances;

	void* __restrict bvh; // TODO: BVH!
} ray_scene_t;

int backgroundWidth, backgroundHeight, backgroundStride;
float* __restrict background;
static vec3 cast_scene(ray_scene_t* scene, vec3 rayO, vec3 rayD, uint32_t depth)
{
	if (depth >= renderConfig.maxDepth)
	{
		return (vec3) { 0 };
	}

	renderStats.rayCount++;

	const float NoRayIntersection = FLT_MAX;

	// TODO: This recast bias is simply to avoid re-intersecting with our object when casting.
	// Do we want to handle this some other way?
	const float ReCastBias = 0.0001f;

	uint32_t closestInstanceIndex = 0;
	float closestDistance = NoRayIntersection;

	// TODO: Traverse our BVH instead
	// TODO: Gather list of candidate BVHs
	// TODO: Sort list of instances by type
	// TODO: find closest intersection in candidates
	// TODO: ?!?!?
	uint64_t intersectionStartTime = cranpl_timestamp_micro();
	for (uint32_t i = 0; i < scene->instances.count; i++)
	{
		// TODO: Add support for other shapes
		vec3 instancePos = scene->instances.data[i].pos;
		renderable_index_t renderableIndex = scene->instances.data[i].renderableIndex;
		assert(renderableIndex.typeIndex == renderable_sphere);
		sphere_t sphere = ((sphere_t*)scene->renderables[renderable_sphere])[renderableIndex.dataIndex];

		if (doesSphereRayIntersect(rayO, rayD, instancePos, sphere.rad))
		{
			float intersectionDistance = sphereRayIntersection(rayO, rayD, ReCastBias, NoRayIntersection, instancePos, sphere.rad);
			// TODO: Do we want to handle tie breakers somehow?
			if (intersectionDistance < closestDistance)
			{
				closestInstanceIndex = i;
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

			vec3 spherePoint = vec3_add(intersectionPoint, normal);
			spherePoint = vec3_add(normal, randomInUnitSphere());
			vec3 sceneCast = cast_scene(scene, intersectionPoint, spherePoint, depth+1);

			sceneCast = vec3_mulf(sceneCast, vec3_dot(spherePoint, normal));
			return vec3_mul(sceneCast, vec3_mulf(lambertData.color, RPI));
		}
		case material_metal:
		{
			vec3 castDir = vec3_reflect(rayD, normal);
			vec3 sceneCast = cast_scene(scene, intersectionPoint, castDir, depth+1);

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

void generate_scene(ray_scene_t* scene)
{
	uint64_t startTime = cranpl_timestamp_micro();

	static instance_t instances[90];
	static sphere_t baseSphere = { .rad = 0.05f };

	static material_lambert_t lambert = { .color = { 0.8f, 0.5f, 0.9f } };
	static material_metal_t metal = { 0 };

	for (uint32_t i = 0; i < 48; i++)
	{
		float t = (float)i / (float)48;
		instances[i] = (instance_t)
		{
			.pos = { .x = cosf(t * TAO), .y = sinf(t * TAO), .z = 2.0f },
			.materialIndex = { .dataIndex = 0,.typeIndex = i % 2 },
			.renderableIndex = { .dataIndex = 0, .typeIndex = renderable_sphere, }
		};
	}

	for (uint32_t i = 0; i < 24; i++)
	{
		float t = (float)i / (float)24;
		instances[i + 48] = (instance_t)
		{
			.pos = { .x = cosf(t * TAO) * 0.75f, .y = sinf(t * TAO) * 0.75f, .z = 2.0f },
			.materialIndex = { .dataIndex = 0,.typeIndex = (i + 1) % 2 },
			.renderableIndex = { .dataIndex = 0, .typeIndex = renderable_sphere, }
		};
	}

	for (uint32_t i = 0; i < 12; i++)
	{
		float t = (float)i / (float)12;
		instances[i + 72] = (instance_t)
		{
			.pos = { .x = cosf(t * TAO) * 0.5f, .y = sinf(t * TAO) * 0.5f, .z = 2.0f },
			.materialIndex = { .dataIndex = 0,.typeIndex = i % 2 },
			.renderableIndex = { .dataIndex = 0, .typeIndex = renderable_sphere, }
		};
	}

	for (uint32_t i = 0; i < 6; i++)
	{
		float t = (float)i / (float)6;
		instances[i + 84] = (instance_t)
		{
			.pos = { .x = cosf(t * TAO) * 0.25f, .y = sinf(t * TAO) * 0.25f, .z = 2.0f },
			.materialIndex = { .dataIndex = 0,.typeIndex = (i + 1) % 2 },
			.renderableIndex = { .dataIndex = 0, .typeIndex = renderable_sphere, }
		};
	}


	// Output our scene
	*scene = (ray_scene_t)
	{
		.instances =
		{
			.data = instances,
			.count = 90
		},
		.renderables[renderable_sphere] = &baseSphere,
		.materials =
		{
			[material_lambert] = &lambert,
			[material_metal] = &metal
		}
	};

	renderStats.sceneGenerationTime = cranpl_timestamp_micro() - startTime;
}

int main()
{
	renderConfig = (render_config_t)
	{
		.maxDepth = UINT32_MAX,
		.samplesPerPixel = 10,
		.renderWidth = 1024,
		.renderHeight = 768
	};

	uint64_t startTime = cranpl_timestamp_micro();
	srand(0);

	background = stbi_loadf("background_4k.hdr", &backgroundWidth, &backgroundHeight, &backgroundStride, 0);

	ray_scene_t scene;
	generate_scene(&scene);

	int32_t imgWidth = renderConfig.renderWidth, imgHeight = renderConfig.renderHeight, imgStride = 4;
	int32_t halfImgWidth = imgWidth / 2, halfImgHeight = imgHeight / 2;

	// TODO: How do we want to express our camera?
	// Currently simply using the near triangle.
	float near = 1.0f, nearHeight = 1.0f, nearWidth = nearHeight * (float)imgWidth / (float)imgHeight;

	vec3 origin = { 0.0f, 0.0f, 0.0f };
	vec3 forward = { .x = 0.0f,.y = 0.0f,.z = 1.0f }, right = { .x = 1.0f,.y = 0.0f,.z = 0.0f }, up = { .x = 0.0f, .y = 1.0f, .z = 0.0f };

	float* __restrict hdrImage = malloc(imgWidth * imgHeight * imgStride * sizeof(float));

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

				float randX = xOff + xStep * (rand01f() * 0.5f - 0.5f);
				float randY = yOff + yStep * (rand01f() * 0.5f - 0.5f);

				// Construct our ray as a vector going from our origin to our near plane
				// V = F*n + R*ix*worldWidth/imgWidth + U*iy*worldHeight/imgHeight
				vec3 rayDir = vec3_add(vec3_mulf(forward, near), vec3_add(vec3_mulf(right, randX), vec3_mulf(up, randY)));
				sceneColor = vec3_add(sceneColor, vec3_mulf(cast_scene(&scene, origin, rayDir, 0), rcp((float)renderConfig.samplesPerPixel)));
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
		uint8_t* __restrict bitmap = malloc(imgWidth * imgHeight * imgStride);
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
		printf("\t\tSkybox Time: %f [%f%%]\n", micro_to_seconds(renderStats.skyboxTime), (float)renderStats.skyboxTime / (float)renderStats.renderTime);
		printf("\tImage Space Time: %f [%f%%]\n", micro_to_seconds(renderStats.imageSpaceTime), (float)renderStats.imageSpaceTime / (float)renderStats.totalTime);
		printf("\n");
		printf("Rays Fired: %" PRIu64 "\n", renderStats.rayCount);
		printf("\tCamera Rays Fired: %" PRIu64 " [%f%%]\n", renderStats.primaryRayCount, (float)renderStats.primaryRayCount / (float)renderStats.rayCount);
		printf("\tBounce Rays Fired: %" PRIu64 " [%f%%]\n", renderStats.rayCount - renderStats.primaryRayCount, (float)(renderStats.rayCount - renderStats.primaryRayCount) / (float)renderStats.rayCount);
	
		system("pause");
	}
	return 0;
}
