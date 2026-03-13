#version 150 compatibility

uniform sampler2D gtexture;

in vec2 texCoord;
in vec4 tintColor;

void main() {
    vec4 albedo = texture2D(gtexture, texCoord) * tintColor;
    if (albedo.a < 0.1) {
        discard;
    }
    gl_FragData[0] = albedo;
}
