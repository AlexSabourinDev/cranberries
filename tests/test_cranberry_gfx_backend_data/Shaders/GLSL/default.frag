#version 450
#extension GL_ARB_separate_shader_objects : enable
#pragma shader_stage( fragment )

layout (set=1, binding = 1 ) uniform sampler2D in_Texture;

layout( location = 0 ) out vec4 out_Color;

void main()
{
	out_Color = texture(in_Texture, vec2(0.0));
}