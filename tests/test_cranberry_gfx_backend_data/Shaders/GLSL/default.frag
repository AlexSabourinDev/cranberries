#version 450
#extension GL_ARB_separate_shader_objects : enable
#pragma shader_stage( fragment )

layout( location = 0 ) out vec4 out_Color;

void main()
{
	out_Color = vec4(1.0);
}