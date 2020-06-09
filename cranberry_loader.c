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
	uint32_t materialLibCount = 0;
	// TODO: Skip comments
	for (char* cran_restrict fileIter = fileStart; fileIter != fileEnd; fileIter++)
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
					if (isspace(fileIter[1]))
					{
						vertexCount++;
					}
					break;
				}
				break;

			case 'f':
			case 'F':
				{
					uint32_t vertCount = 0;
					fileIter+=1;
					for (uint32_t i = 0; i < 4; i++)
					{
						int32_t v = strtol(fileIter, &fileIter, 10);
						if (v != 0)
						{
							strtol(fileIter + 1, &fileIter, 10);
							strtol(fileIter + 1, &fileIter, 10);
							vertCount++;
						}
						else
						{
							break;
						}
					}
					assert(vertCount == 3 || vertCount == 4);
					faceCount += vertCount == 3 ? 1 : 2;
				}
				break;
			default:
				{
					if(memcmp(fileIter, "usemtl", strlen("usemtl")) == 0)
					{
						materialCount++;
					}

					if(memcmp(fileIter, "mtllib", strlen("mtllib")) == 0)
					{
						materialLibCount++;
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
	uint32_t* cran_restrict materialBoundaries = (uint32_t* cran_restrict)allocator.alloc(allocator.instance, sizeof(uint32_t) * materialCount);
	char** cran_restrict materialNames = (char** cran_restrict)allocator.alloc(allocator.instance, sizeof(char*) * materialCount);
	char** cran_restrict materialLibNames = (char** cran_restrict)allocator.alloc(allocator.instance, sizeof(char*) * materialLibCount);

	uint32_t vertexIndex = 0;
	uint32_t normalIndex = 0;
	uint32_t uvIndex = 0;
	uint32_t faceIndex = 0;
	uint32_t materialIndex = 0;
	uint32_t materialLibIndex = 0;
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
					if (isspace(fileIter[1]))
					{
						fileIter += 1;

						uint32_t startingIndex = vertexIndex;
						for (uint32_t i = 0; i < 3; i++)
						{
							vertices[vertexIndex] = strtof(fileIter, &fileIter);
							if (flags & cranl_cm_to_m)
							{
								vertices[vertexIndex] /= 100.0f;
							}
							vertexIndex++;
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

					int32_t v[4];
					int32_t u[4];
					int32_t n[4];

					uint32_t vCount = 0;
					for (uint32_t i = 0; i < 4; i++)
					{
						v[i] = strtol(fileIter, &fileIter, 10);
						if (v[i] != 0)
						{
							u[i] = strtol(fileIter + 1, &fileIter, 10);
							n[i] = strtol(fileIter + 1, &fileIter, 10);
							vCount++;
						}
						else
						{
							break;
						}
					}
					assert(vCount == 3 || vCount == 4);

					uint32_t triangulationCount = vCount == 3 ? 1 : 2;
					for(uint32_t tr = 0; tr < triangulationCount; tr++)
					{
						int32_t triangleTable[2][3] =
						{
							{ 0, 1, 2},
							{ 0, 2, 3}
						};

						uint32_t startingIndex = faceIndex;
						for (uint32_t i = 0; i < 3; i++)
						{
							int32_t index = triangleTable[tr][i];

							vertexIndices[faceIndex] = v[index] > 0 ? (v[index] - 1) : vertexIndex / 3 + v[index];
							uvIndices[faceIndex] = u[index] > 0 ? (u[index] - 1) : uvIndex / 2 + u[index];
							normalIndices[faceIndex] = n[index] > 0 ? (n[index] - 1) : normalIndex / 3 + n[index];
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
				}
				break;
			default:
				{
					if(memcmp(fileIter, "usemtl", strlen("usemtl")) == 0)
					{
						char const* materialName = fileIter + strlen("usemtl") + 1;

						char const* materialNameEnd = materialName;
						for (; !isspace(materialNameEnd[0]); materialNameEnd++);
						assert(materialNameEnd != NULL);

						materialBoundaries[materialIndex] = faceIndex / 3;

						uint64_t nameLength = (materialNameEnd - materialName);
						materialNames[materialIndex] = allocator.alloc(allocator.instance, nameLength + 1);
						memcpy(materialNames[materialIndex], materialName, nameLength);
						materialNames[materialIndex][nameLength] = '\0';

						materialIndex++;
					}

					if(memcmp(fileIter, "mtllib", strlen("mtllib")) == 0)
					{
						char const* materialLibName = fileIter + strlen("mtllib") + 1;

						char const* materialLibNameEnd = materialLibName;
						for (; !isspace(materialLibNameEnd[0]); materialLibNameEnd++);
						assert(materialLibNameEnd != NULL);

						uint64_t nameLength = (materialLibNameEnd - materialLibName);
						materialLibNames[materialLibIndex] = allocator.alloc(allocator.instance, nameLength + 1);
						memcpy(materialLibNames[materialLibIndex], materialLibName, nameLength);
						materialLibNames[materialLibIndex][nameLength] = '\0';

						materialLibIndex++;
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
	assert(materialIndex == materialCount);
	assert(materialLibIndex == materialLibCount);
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
			.materialBoundaries = materialBoundaries, // TODO: If we deduplicate material names, these boundaries would need to reference their index
			.materialNames = materialNames, // TODO: Could deduplicate these
			.count = materialCount
		},
		.materialLibraries =
		{
			.names = materialLibNames,
			.count = materialLibCount
		}
	};
}

void cranl_obj_free(cranl_mesh_t const* mesh, cranl_allocator_t allocator)
{
	allocator.free(allocator.instance, mesh->vertices.data);
	allocator.free(allocator.instance, mesh->normals.data);
	allocator.free(allocator.instance, mesh->uvs.data);
	allocator.free(allocator.instance, mesh->faces.vertexIndices);
	allocator.free(allocator.instance, mesh->faces.normalIndices);
	allocator.free(allocator.instance, mesh->faces.uvIndices);
	allocator.free(allocator.instance, mesh->materials.materialBoundaries);

	for (uint32_t i = 0; i < mesh->materials.count; i++)
	{
		allocator.free(allocator.instance, mesh->materials.materialNames[i]);
	}
	allocator.free(allocator.instance, mesh->materials.materialNames);

	for (uint32_t i = 0; i < mesh->materialLibraries.count; i++)
	{
		allocator.free(allocator.instance, mesh->materialLibraries.names[i]);
	}
	allocator.free(allocator.instance, mesh->materialLibraries.names);
}

cranl_material_lib_t cranl_obj_mat_load(char const* cran_restrict filePath, cranl_allocator_t allocator)
{
	cranpl_file_map_t fileMap = cranpl_map_file(filePath);

	char* cran_restrict fileStart = (char* cran_restrict)fileMap.fileData;
	char* cran_restrict fileEnd = (char* cran_restrict)fileStart + fileMap.fileSize;

	uint32_t materialCount = 0;
	for (char* cran_restrict fileIter = fileStart; fileIter != fileEnd; fileIter++)
	{
		if (fileIter == fileStart || fileIter[-1] == '\n')
		{
			if(memcmp(fileIter, "newmtl", strlen("newmtl")) == 0)
			{
				materialCount++;
			}
		}
	}

	cranl_material_t* materials = allocator.alloc(allocator.instance, materialCount * sizeof(cranl_material_t));
	uint32_t materialIndex = 0;
	for (char* cran_restrict fileIter = fileStart; fileIter != fileEnd; fileIter++)
	{
		if (fileIter == fileStart || fileIter[-1] == '\n')
		{
			if(memcmp(fileIter, "newmtl", strlen("newmtl")) == 0)
			{
				char const* materialName = fileIter + strlen("newmtl") + 1;

				char const* materialNameEnd = materialName;
				for (; !isspace(materialNameEnd[0]); materialNameEnd++);
				assert(materialNameEnd != NULL);

				uint64_t nameLength = (materialNameEnd - materialName);
				materials[materialIndex].name = allocator.alloc(allocator.instance, nameLength + 1);
				memcpy(materials[materialIndex].name, materialName, nameLength);
				materials[materialIndex].name[nameLength] = '\0';

				for (fileIter += strlen("newmtl"); (fileIter + 1) != fileEnd && memcmp(fileIter+1, "newmtl", strlen("newmtl")) != 0; fileIter++)
				{
					switch (*fileIter)
					{
					case 'K':
						{
							if (*(fileIter + 1) == 'd' && isspace(*(fileIter - 1)))
							{
								float r=strtof(fileIter + 2, &fileIter);
								float g=strtof(fileIter + 1, &fileIter);
								float b=strtof(fileIter + 1, &fileIter);

								materials[materialIndex].albedo[0] = r;
								materials[materialIndex].albedo[1] = g;
								materials[materialIndex].albedo[2] = b;
							}
						}
					}
				}

				materialIndex++;
			}
		}
	}

	cranpl_unmap_file(fileMap);
	assert(materialIndex == materialCount);
	return (cranl_material_lib_t)
	{
		.materials = materials,
		.count = materialCount
	};
}

void cranl_obj_mat_free(cranl_material_lib_t materialLibrary, cranl_allocator_t allocator)
{
	for (uint32_t i = 0; i < materialLibrary.count; i++)
	{
		allocator.free(allocator.instance, materialLibrary.materials[i].name);
	}
	allocator.free(allocator.instance, materialLibrary.materials);
}
