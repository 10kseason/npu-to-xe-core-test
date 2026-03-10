#version 150 compatibility

uniform sampler2D gtexture;
uniform sampler2D lightmap;
uniform sampler2D npuAssist;
uniform float viewWidth;
uniform float viewHeight;

in vec2 texCoord;
in vec2 lightmapCoord;
in vec4 tintColor;
in vec3 worldNormal;

vec2 saturate2(vec2 value) {
    return clamp(value, vec2(0.0), vec2(1.0));
}

vec3 toLinear(vec3 color) {
    return pow(clamp(color, vec3(0.0), vec3(1.0)), vec3(2.2));
}

void main() {
    vec4 albedo = texture2D(gtexture, texCoord) * tintColor;
    if (albedo.a < 0.02) {
        discard;
    }

    vec2 screenUv = gl_FragCoord.xy / vec2(max(viewWidth, 1.0), max(viewHeight, 1.0));
    vec4 assist = texture2D(npuAssist, saturate2(screenUv));
    vec2 bakedLight = texture2D(lightmap, lightmapCoord).rg;

    vec3 waterTint = mix(vec3(0.03, 0.14, 0.24), vec3(0.08, 0.26, 0.40), bakedLight.y);
    waterTint *= mix(0.92, 1.14, assist.b);
    float opacity = clamp(0.28 + albedo.a * 0.42, 0.24, 0.82);
    vec3 waterColor = toLinear(albedo.rgb) * waterTint;

    vec2 waterFlow = normalize((assist.rg * 2.0 - 1.0) + worldNormal.xz * 0.45) * 0.5 + 0.5;
    float reflectivity = clamp(0.58 + assist.b * 0.28 + bakedLight.y * 0.10, 0.45, 0.98);

    /* RENDERTARGETS: 0,4 */
    gl_FragData[0] = vec4(waterColor, opacity);
    gl_FragData[1] = vec4(waterFlow, reflectivity, 1.0);
}
