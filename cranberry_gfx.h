#ifndef __CRANBERRY_GFX
#define __CRANBERRY_GFX

#include <stdint.h>

typedef struct _crang_context_t crang_context_t;
typedef struct { uint32_t id; } crang_mesh_id_t;
typedef struct { uint32_t id; } crang_material_id_t;
typedef struct { uint32_t id; } crang_image_id_t;
typedef struct { uint32_t id; } crang_sampler_id_t;
static const crang_mesh_id_t crang_invalid_mesh = { 0 };
static const crang_material_id_t crang_invalid_material = { 0 };

typedef enum
{
	crang_material_deferred,
	crang_material_count,
} crang_material_e;

typedef struct
{
	struct
	{
		void* hinstance;
		void* hwindow;
	} win32;

	struct
	{
		struct
		{
			uint8_t* gbufferVShader;
			uint32_t gbufferVShaderSize;
			uint8_t* gbufferFShader;
			uint32_t gbufferFShaderSize;
			uint8_t* gbufferComputeShader;
			uint32_t gbufferComputeShaderSize;
		} deferred;
	} materials;
} crang_init_desc_t;

typedef struct
{
	// uv pos[3], normal[3]
	float pos[4];
	float normal[4];
} crang_vertex_t;

typedef struct
{
	struct
	{
		crang_vertex_t* data;
		uint32_t count; // number of vertices (sets of 3 floats)
	} vertices;

	struct
	{
		uint32_t* data;
		uint32_t count; // number of indices
	} indices;
} crang_mesh_desc_t;

// [Colums][Rows]
typedef struct { float f[4][4]; } crang_mat4_t;
typedef struct { float f[3][4]; } crang_mat4x3_t;
static const crang_mat4_t crang_mat4_identity =
{
	{
		{1.0f,0.0f,0.0f,0.0f},
		{0.0f,1.0f,0.0f,0.0f},
		{0.0f,0.0f,1.0f,0.0f},
		{0.0f,0.0f,0.0f,1.0f}
	}
};

#define crang_max_instances 1000
#define crang_max_batches 10
typedef struct
{
	crang_mat4_t viewProj;
	struct
	{
		crang_material_id_t material;
		struct
		{
			crang_mesh_id_t mesh;
			crang_mat4x3_t const* transforms;
			uint32_t count;
		} instances[crang_max_instances];
	} batches[crang_material_count][crang_max_batches];
} crang_view_t;

typedef struct
{
	float albedoTint[4];
	crang_image_id_t albedoImage;
} crang_deferred_desc_t;

typedef enum
{
	crang_image_format_rgba8,
	crang_image_format_count
} crang_image_format_e;

typedef struct
{
	crang_image_format_e format;
	uint32_t width;
	uint32_t height;
	uint8_t* data;
} crang_image_desc_t;

crang_context_t* crang_init(crang_init_desc_t* desc);

crang_mesh_id_t crang_create_mesh(crang_context_t* ctx, crang_mesh_desc_t const* desc);
crang_image_id_t crang_create_image(crang_context_t* ctx, crang_image_desc_t const* desc);

crang_material_id_t crang_create_mat_deferred(crang_context_t* ctx, crang_deferred_desc_t const* desc);

void crang_draw_view(crang_context_t* ctx, crang_view_t* view);

#endif // __CRANBERRY_GFX
