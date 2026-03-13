#version 150 compatibility

uniform mat4 gbufferModelViewInverse;

out vec2 texCoord;
out vec2 lightmapCoord;
out vec4 tintColor;
out vec3 worldNormal;

void main() {
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    gl_Position = gl_ProjectionMatrix * viewPos;
    texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lightmapCoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    tintColor = gl_Color;
    vec3 viewNormal = normalize(gl_NormalMatrix * gl_Normal);
    worldNormal = normalize(mat3(gbufferModelViewInverse) * viewNormal);
}
