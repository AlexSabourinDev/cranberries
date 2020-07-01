cd /D "%~dp0"
glslangValidator -V -o ../SPIR-V/gbuffer_frag.spv gbuffer_frag.glsl -H -S frag
glslangValidator -V -o ../SPIR-V/gbuffer_vert.spv gbuffer_vert.glsl -H -S vert
glslangValidator -V -o ../SPIR-V/gbuffer_compute.spv gbuffer_compute.glsl -H -S comp
pause