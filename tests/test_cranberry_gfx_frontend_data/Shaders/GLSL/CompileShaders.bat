cd /D "%~dp0"
glslangValidator -V -o ../SPIR-V/default.fspv default.frag -H
glslangValidator -V -o ../SPIR-V/default.vspv default.vert -H
pause