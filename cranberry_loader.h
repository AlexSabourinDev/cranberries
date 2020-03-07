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
} cranl_mesh_t;

cranl_mesh_t cranl_obj_load(char const* cran_restrict filepath);
void cranl_obj_free(cranl_mesh_t const* mesh);