#version 150 compatibility

#include "/lib/distort.glsl"

out vec2 texCoord;
out vec4 tintColor;

void main() {
    texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    tintColor = gl_Color;
    gl_Position = ftransform();
    gl_Position.xyz = distortShadowClipPos(gl_Position.xyz);
}
