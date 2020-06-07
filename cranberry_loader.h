#pragma once

#include <stdint.h>

#ifdef _MSC_BUILD
#define cran_restrict __restrict
#else
#define cran_restrict restrict
#endif

typedef struct
{
	struct
	{
		float* cran_restrict data;
		uint32_t count;
	} vertices;

	struct
	{
		float* cran_restrict data;
		uint32_t count;
	} normals;

	struct
	{
		float* cran_restrict data;
		uint32_t count;
	} uvs;

	struct
	{
		uint32_t* cran_restrict vertexIndices;
		uint32_t* cran_restrict normalIndices;
		uint32_t* cran_restrict uvIndices;
		uint32_t count;
	} faces;

	struct
	{
		uint32_t* cran_restrict materialBoundaries;
		char** cran_restrict materialNames;
		uint32_t count;
	} materials;
} cranl_mesh_t;

enum
{
	cranl_flip_yz = 0x01,
	cranl_cm_to_m = 0x02
};

typedef struct
{
	void* instance;
	void*(*alloc)(void* allocator, uint64_t size);
	void(*free)(void* allocator, uint64_t size);
} cranl_allocator_t;

cranl_mesh_t cranl_obj_load(char const* cran_restrict filepath, uint32_t flags, cranl_allocator_t allocator);
void cranl_obj_free(cranl_mesh_t const* mesh, cranl_allocator_t allocator);
