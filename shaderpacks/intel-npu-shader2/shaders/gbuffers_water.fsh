#version 150 compatibility

#include "/lib/npu_policy.glsl"

uniform sampler2D gtexture;
uniform sampler2D lightmap;
uniform sampler2D npuAssist;
uniform float viewWidth;
uniform float viewHeight;

in vec2 texCoord;
in vec2 lightmapCoord;
in vec4 tintColor;
in vec3 worldNormal;
flat in float blockId;

const float WATER_BLOCK_ID = 100.0;

vec3 toLinear(vec3 color) {
    return pow(clamp(color, vec3(0.0), vec3(1.0)), vec3(2.2));
}

void main() {
    vec4 albedo = texture2D(gtexture, texCoord) * tintColor;
    if (albedo.a < 0.02) {
        discard;
    }

    vec2 bakedLight = texture2D(lightmap, lightmapCoord).rg;
    vec3 encodedNormal = normalize(worldNormal) * 0.5 + 0.5;
    vec3 baseColor = toLinear(albedo.rgb);
    bool isWaterBlock = abs(blockId - WATER_BLOCK_ID) < 0.5;

    vec3 outColor = baseColor;
    float opacity = albedo.a;
    vec4 auxWater = vec4(0.5, 0.5, 0.0, 0.0);

    if (isWaterBlock) {
        vec2 screenUv = clamp(
            gl_FragCoord.xy / vec2(max(viewWidth, 1.0), max(viewHeight, 1.0)),
            vec2(0.0),
            vec2(1.0)
        );
        NpuPolicy policy = sampleNpuPolicy(npuAssist, screenUv);
        vec3 waterTint = mix(vec3(0.02, 0.10, 0.18), vec3(0.05, 0.20, 0.32), bakedLight.y);
        waterTint *= mix(0.92, 1.10, policy.energy);
        opacity = clamp(0.20 + albedo.a * 0.30, 0.16, 0.58);
        outColor = baseColor * waterTint;

        vec2 waterFlow = normalize(policy.trajectory * 1.16 + worldNormal.xz * 0.18) * 0.5 + 0.5;
        float reflectivity = clamp(0.20 + policy.energy * 0.38 + policy.budget * 0.22 + bakedLight.y * 0.05, 0.14, 0.78);
        float roughness = clamp(0.70 - policy.budget * 0.34 + (1.0 - policy.energy) * 0.05 + (1.0 - abs(worldNormal.y)) * 0.08, 0.18, 0.78);
        auxWater = vec4(waterFlow, reflectivity, roughness);
    }

    /* RENDERTARGETS: 0,1,2,4 */
    gl_FragData[0] = vec4(outColor, opacity);
    gl_FragData[1] = vec4(encodedNormal, 1.0);
    gl_FragData[2] = vec4(clamp(bakedLight, 0.0, 1.0), 0.0, 1.0);
    gl_FragData[3] = auxWater;
}
