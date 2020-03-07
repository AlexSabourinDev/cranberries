#include "cranberry_loader.h"
#include "cranberry_platform.h"

#include <stdlib.h>
#include <ctype.h>
#include <assert.h>

cranl_mesh_t cranl_obj_load(char const* cran_restrict filepath)
{
	cranpl_file_map_t fileMap = cranpl_map_file(filepath);

	char* cran_restrict fileStart = (char* cran_restrict)fileMap.fileData;
	char* cran_restrict fileEnd = (char* cran_restrict)fileStart + fileMap.fileSize;

	uint32_t vertexCount = 0;
	uint32_t normalCount = 0;
	uint32_t uvCount = 0;
	uint32_t faceCount = 0;
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
			}
		}
	}

	float* cran_restrict vertices = (float* cran_restrict)malloc(sizeof(float) * 3 * vertexCount);
	float* cran_restrict normals = (float* cran_restrict)malloc(sizeof(float) * 3 * normalCount);
	float* cran_restrict uvs = (float* cran_restrict)malloc(sizeof(float) * 2 * uvCount);
	uint32_t* cran_restrict vertexIndices = (uint32_t* cran_restrict)malloc(sizeof(uint32_t) * faceCount * 3);
	uint32_t* cran_restrict normalIndices = (uint32_t* cran_restrict)malloc(sizeof(uint32_t) * faceCount * 3);
	uint32_t* cran_restrict uvIndices = (uint32_t* cran_restrict)malloc(sizeof(uint32_t) * faceCount * 3);

	uint32_t vertexIndex = 0;
	uint32_t normalIndex = 0;
	uint32_t uvIndex = 0;

	uint32_t faceIndex = 0;
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
					fileIter += 2;
					for (uint32_t i = 0; i < 3; i++)
					{
						normals[normalIndex++] = strtof(fileIter, &fileIter);
					}
					break;
				default:
					if (isblank(fileIter[1]))
					{
						fileIter += 1;
						for (uint32_t i = 0; i < 3; i++)
						{
							vertices[vertexIndex++] = strtof(fileIter, &fileIter);
						}
					}
					break;
				}
				break;

			case 'f':
			case 'F':
				fileIter += 1;
				for (uint32_t i = 0; i < 3; i++)
				{
					int32_t vertex = strtol(fileIter, &fileIter, 10);
					int32_t uv = strtol(fileIter + 1, &fileIter, 10);
					int32_t normal = strtol(fileIter + 1, &fileIter, 10);

					vertexIndices[faceIndex] = vertex > 0 ? (vertex - 1) * 3 : vertexIndex + vertex * 3;
					uvIndices[faceIndex] = uv > 0 ? (uv - 1) * 2 : uvIndex + uv * 2;
					normalIndices[faceIndex] = normal > 0 ? (normal - 1) * 3 : normalIndex + normal * 3;
					faceIndex++;
				}
				break;
			}
		}
	}

	cranpl_unmap_file(fileMap);

	assert(vertexCount == vertexIndex);
	assert(uvCount == uvIndex);
	assert(normalCount == normalIndex);
	assert(faceCount == faceIndex);
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
		}
	};
}

void cranl_obj_free(cranl_mesh_t const* mesh)
{
	free(mesh->vertices.data);
	free(mesh->normals.data);
	free(mesh->uvs.data);
	free(mesh->faces.vertexIndices);
	free(mesh->faces.normalIndices);
	free(mesh->faces.uvIndices);
}
