#define _CRT_SECURE_NO_WARNINGS
#include "cranberry_loader.h"
#include "cranberry_platform.h"

#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

cranl_mesh_t cranl_obj_load(char const* cran_restrict filepath, uint32_t flags, cranl_allocator_t allocator)
{
	cranpl_file_map_t fileMap = cranpl_map_file(filepath);

	char* cran_restrict fileStart = (char* cran_restrict)fileMap.fileData;
	char* cran_restrict fileEnd = (char* cran_restrict)fileStart + fileMap.fileSize;

	uint32_t vertexCount = 0;
	uint32_t normalCount = 0;
	uint32_t uvCount = 0;
	uint32_t faceCount = 0;
	uint32_t materialCount = 0;
	for (char const* cran_restrict fileIter = fileStart; fileIter != fileEnd; fileIter++)
	{
		if (fileIter == fileStart || fileIter[-1] == '\n')
		{
			switch (fileIter[0])
			{
			case 'v':
			case 'V':
				switch (fileIter[1])
				{
				case 't':
				case 'T':
					uvCount++;
					break;
				case 'n':
				case 'N':
					normalCount++;
					break;
				default:
					if (isblank(fileIter[1]))
					{
						vertexCount++;
					}
					break;
				}
				break;

			case 'f':
			case 'F':
				faceCount++;
				break;
			default:
				{
					if(memcmp(fileIter, "usemtl", strlen("usemtl")) == 0)
					{
						materialCount++;
					}
				}
				break;
			}
		}
	}

	float* cran_restrict vertices = (float* cran_restrict)allocator.alloc(allocator.instance, sizeof(float) * 3 * vertexCount);
	float* cran_restrict normals = (float* cran_restrict)allocator.alloc(allocator.instance, sizeof(float) * 3 * normalCount);
	float* cran_restrict uvs = (float* cran_restrict)allocator.alloc(allocator.instance, sizeof(float) * 2 * uvCount);
	uint32_t* cran_restrict vertexIndices = (uint32_t* cran_restrict)allocator.alloc(allocator.instance, sizeof(uint32_t) * faceCount * 3);
	uint32_t* cran_restrict normalIndices = (uint32_t* cran_restrict)allocator.alloc(allocator.instance, sizeof(uint32_t) * faceCount * 3);
	uint32_t* cran_restrict uvIndices = (uint32_t* cran_restrict)allocator.alloc(allocator.instance, sizeof(uint32_t) * faceCount * 3);
	uint32_t* cran_restrict materialBoundaries = (uint32_t* cran_restrict)allocator.alloc(allocator.instance, sizeof(uint32_t) * materialCount * 3);
	char** cran_restrict materialNames = (char** cran_restrict)allocator.alloc(allocator.instance, sizeof(char*) * materialCount * 3);

	uint32_t vertexIndex = 0;
	uint32_t normalIndex = 0;
	uint32_t uvIndex = 0;
	uint32_t faceIndex = 0;
	uint32_t materialIndex = 0;
	for (char* fileIter = fileStart; fileIter != fileEnd; fileIter++)
	{
		if (fileIter == fileStart || fileIter[-1] == '\n')
		{
			switch (fileIter[0])
			{
			case 'v':
			case 'V':
				switch (fileIter[1])
				{
				case 't':
				case 'T':
					fileIter += 2;
					for (uint32_t i = 0; i < 2; i++)
					{
						uvs[uvIndex++] = strtof(fileIter, &fileIter);
					}
					break;
				case 'n':
				case 'N':
					{
						fileIter += 2;

						uint32_t startingIndex = normalIndex;
						for (uint32_t i = 0; i < 3; i++)
						{
							normals[normalIndex++] = strtof(fileIter, &fileIter);
						}

						if (flags & cranl_flip_yz)
						{
							float temp = normals[startingIndex + 1];
							normals[startingIndex + 1] = normals[startingIndex + 2];
							normals[startingIndex + 2] = temp;
						}
					}
					break;
				default:
					if (isblank(fileIter[1]))
					{
						fileIter += 1;

						uint32_t startingIndex = vertexIndex;
						for (uint32_t i = 0; i < 3; i++)
						{
							vertices[vertexIndex++] = strtof(fileIter, &fileIter);
						}

						if (flags & cranl_flip_yz)
						{
							float temp = vertices[startingIndex + 1];
							vertices[startingIndex + 1] = vertices[startingIndex + 2];
							vertices[startingIndex + 2] = temp;
						}
					}
					break;
				}
				break;

			case 'f':
			case 'F':
				{
					fileIter += 1;

					uint32_t startingIndex = faceIndex;
					for (uint32_t i = 0; i < 3; i++)
					{
						int32_t vertex = strtol(fileIter, &fileIter, 10);
						int32_t uv = strtol(fileIter + 1, &fileIter, 10);
						int32_t normal = strtol(fileIter + 1, &fileIter, 10);

						vertexIndices[faceIndex] = vertex > 0 ? (vertex - 1) : vertexIndex / 3 + vertex;
						uvIndices[faceIndex] = uv > 0 ? (uv - 1) : uvIndex / 2 + uv;
						normalIndices[faceIndex] = normal > 0 ? (normal - 1) : normalIndex / 3 + normal;
						faceIndex++;
					}

					if (flags & cranl_flip_yz)
					{
						uint32_t temp = vertexIndices[startingIndex + 1];
						vertexIndices[startingIndex + 1] = vertexIndices[startingIndex + 2];
						vertexIndices[startingIndex + 2] = temp;

						temp = uvIndices[startingIndex + 1];
						uvIndices[startingIndex + 1] = uvIndices[startingIndex + 2];
						uvIndices[startingIndex + 2] = temp;

						temp = normalIndices[startingIndex + 1];
						normalIndices[startingIndex + 1] = normalIndices[startingIndex + 2];
						normalIndices[startingIndex + 2] = temp;
					}
				}
				break;
			default:
				{
					if(memcmp(fileIter, "usemtl", strlen("usemtl")) == 0)
					{
						char const* materialName = fileIter + strlen("usemtl") + 1;

						char const* materialNameEnd = materialName;
						for (; !isblank(materialNameEnd[0]); materialNameEnd++);
						assert(materialNameEnd != NULL);

						materialBoundaries[materialIndex] = faceIndex / 3;

						uint64_t nameLength = (materialNameEnd - materialName);
						materialNames[materialIndex] = malloc(nameLength + 1);
						memcpy(materialNames[materialIndex], materialName, nameLength);
						materialNames[materialIndex][nameLength] = '\0';

						materialIndex++;
					}
				}
				break;
			}
		}
	}

	cranpl_unmap_file(fileMap);

	assert(vertexCount == vertexIndex / 3);
	assert(uvCount == uvIndex / 2);
	assert(normalCount == normalIndex / 3);
	assert(faceCount == faceIndex / 3);
	return (cranl_mesh_t)
	{
		.vertices = 
		{
			.data = vertices,
			.count = vertexCount
		},
		.normals =
		{
			.data = normals,
			.count = normalCount
		},
		.uvs = 
		{
			.data = uvs,
			.count = uvCount
		},
		.faces = 
		{
			.vertexIndices = vertexIndices,
			.normalIndices = normalIndices,
			.uvIndices = uvIndices,
			.count = faceCount
		},
		.materials =
		{
			.materialBoundaries = materialBoundaries,
			.materialNames = materialNames,
			.count = materialCount
		}
	};
}

void cranl_obj_free(cranl_mesh_t const* mesh, cranl_allocator_t allocator)
{
	allocator.free(allocator.instance, mesh->vertices.count * sizeof(float) * 3);
	allocator.free(allocator.instance, mesh->normals.count * sizeof(float) * 3);
	allocator.free(allocator.instance, mesh->uvs.count * sizeof(float) * 2);
	allocator.free(allocator.instance, mesh->faces.count * sizeof(uint32_t) * 3);
	allocator.free(allocator.instance, mesh->materials.count * sizeof(uint32_t));

	for (uint32_t i = 0; i < mesh->materials.count; i++)
	{
		allocator.free(allocator.instance, strlen(mesh->materials.materialNames[i]) + 1);
	}
	allocator.free(allocator.instance, mesh->materials.count * sizeof(char*));
}
