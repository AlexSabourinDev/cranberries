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
		uint32_t* cran_restrict groupOffsets;
		uint32_t count;
	} groups;

	struct
	{
		uint32_t* cran_restrict materialBoundaries;
		char** cran_restrict materialNames;
		uint32_t count;
	} materials;

	struct
	{
		char** cran_restrict names;
		uint32_t count;
	} materialLibraries;
} cranl_mesh_t;

typedef struct
{
	char* name;
	char* albedoMap; // map_Kd
	char* bumpMap; // map_bump
	char* specMap; // map_ks
	char* maskMap; // map_d
	float albedo[3]; // Kd
	float specular[3]; // Ks
	float emission[3]; // Ke
	float refractiveIndex; // Ni
	float specularAmount; // Ns
} cranl_material_t;

typedef struct
{
	cranl_material_t* materials;
	uint32_t count;
} cranl_material_lib_t;

enum
{
	cranl_flip_yz = 0x01,
	cranl_cm_to_m = 0x02
};

typedef struct
{
	void* instance;
	void*(*alloc)(void* allocator, uint64_t size);
	void(*free)(void* allocator, void* memory);
} cranl_allocator_t;

cranl_mesh_t cranl_obj_load(char const* cran_restrict filepath, uint32_t flags, cranl_allocator_t allocator);
void cranl_obj_free(cranl_mesh_t const* mesh, cranl_allocator_t allocator);

cranl_material_lib_t cranl_obj_mat_load(char const* cran_restrict filePath, cranl_allocator_t allocator);
void cranl_obj_mat_free(cranl_material_lib_t materialLibrary, cranl_allocator_t allocator);
