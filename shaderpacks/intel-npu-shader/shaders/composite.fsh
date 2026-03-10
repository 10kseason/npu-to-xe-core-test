#version 150 compatibility

#include "/lib/distort.glsl"

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D depthtex0;
uniform sampler2D shadowtex0;
uniform sampler2D npuAssist;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowModelView;

uniform vec3 shadowLightPosition;
uniform vec3 sunPosition;
uniform vec3 upPosition;
uniform float viewWidth;
uniform float viewHeight;

in vec2 texCoord;

// Keep true shadow-map work on the GPU.
// This stage depends on the current frame's depth, matrices, and shadowtex0.
// The NPU should only feed low-frequency controls such as softness, tint, or
// sample-budget hints into this pass through npuAssist.
float saturate1(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 saturate2(vec2 value) {
    return clamp(value, vec2(0.0), vec2(1.0));
}

vec3 saturate3(vec3 value) {
    return clamp(value, vec3(0.0), vec3(1.0));
}

vec3 projectAndDivide(mat4 projectionMatrix, vec3 position) {
    vec4 homPos = projectionMatrix * vec4(position, 1.0);
    return homPos.xyz / max(homPos.w, 0.0001);
}

vec3 decodeNormal(vec3 encoded) {
    return normalize(encoded * 2.0 - 1.0);
}

float sampleShadowMap(vec3 shadowCoord, float texelRadius) {
    if (shadowCoord.x <= 0.0 || shadowCoord.x >= 1.0 || shadowCoord.y <= 0.0 || shadowCoord.y >= 1.0) {
        return 1.0;
    }
    if (shadowCoord.z <= 0.0 || shadowCoord.z >= 1.0) {
        return 1.0;
    }

    float visibility = 0.0;
    vec2 texel = vec2(texelRadius / float(shadowMapResolution));
    float compareDepth = shadowCoord.z;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texel;
            float shadowDepth = texture2D(shadowtex0, shadowCoord.xy + offset).r;
            visibility += shadowDepth + 0.0015 >= compareDepth ? 1.0 : 0.0;
        }
    }
    return visibility / 9.0;
}

float sampleScreenSunShadow(vec2 uv, float depth, vec2 sunScreenDir, float radiusPixels) {
    if (length(sunScreenDir) < 0.0001) {
        return 1.0;
    }

    vec2 dir = normalize(-sunScreenDir);
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float occlusion = 0.0;
    for (int index = 1; index <= 6; ++index) {
        float travel = float(index) * radiusPixels;
        vec2 sampleUv = saturate2(uv + dir * pixel * travel);
        float sampleDepth = texture2D(depthtex0, sampleUv).r;
        float depthGap = depth - sampleDepth;
        occlusion += smoothstep(0.0015, 0.02 + float(index) * 0.004, depthGap);
    }
    return 1.0 - occlusion / 6.0;
}

void main() {
    vec2 uv = texCoord;
    vec3 albedo = texture2D(colortex0, uv).rgb;
    float depth = texture2D(depthtex0, uv).r;

    if (depth >= 0.999999) {
        gl_FragData[0] = vec4(albedo, 1.0);
        return;
    }

    vec3 worldNormal = decodeNormal(texture2D(colortex1, uv).rgb);
    vec2 bakedLight = texture2D(colortex2, uv).rg;
    // npuAssist is the right place to feed coarse lighting policy into the pass:
    // alpha -> PCF softness, blue -> extra horizon / atmosphere energy, rg -> spare channels.
    // If you want to move more work to the NPU later, add low-resolution budgets here first.
    vec4 assist = texture2D(npuAssist, saturate2(uv));

    // Reconstructing positions from the live frame must stay on the GPU.
    // It uses current matrices and depth values that the NPU does not see directly.
    vec3 ndcPos = vec3(uv, depth) * 2.0 - 1.0;
    vec3 viewPos = projectAndDivide(gbufferProjectionInverse, ndcPos);
    vec3 feetPlayerPos = (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
    vec3 shadowViewPos = (shadowModelView * vec4(feetPlayerPos, 1.0)).xyz;
    vec4 shadowClip = shadowProjection * vec4(shadowViewPos, 1.0);
    shadowClip.z -= 0.001;
    shadowClip.xyz = distortShadowClipPos(shadowClip.xyz);
    vec3 shadowCoord = shadowClip.xyz / max(shadowClip.w, 0.0001);
    shadowCoord = shadowCoord * 0.5 + 0.5;

    vec3 worldLightDir = normalize(mat3(gbufferModelViewInverse) * normalize(shadowLightPosition));
    vec3 sunDir = normalize(sunPosition);
    vec3 upDir = normalize(upPosition);

    float sunHeight = saturate1(dot(sunDir, upDir) * 0.5 + 0.5);
    float ndl = saturate1(dot(worldNormal, worldLightDir));
    // This is a good NPU hand-off seam: the NPU can predict "how soft should the sun shadow be
    // in this region?" while the actual PCF compare still happens here on the GPU.
    float pcfRadius = mix(0.75, 2.25, assist.a);
    float shadowVisibility = sampleShadowMap(shadowCoord, pcfRadius);
    float screenShadow = sampleScreenSunShadow(uv, depth, shadowLightPosition.xy, mix(8.0, 28.0, assist.a));
    float shadowDensity = mix(0.58, 1.0, assist.r);
    float shadowStrength = mix(0.45, 0.90, sunHeight) * shadowDensity;
    float shadowTerm = min(shadowVisibility, screenShadow);
    float sunShadow = mix(1.0, shadowTerm, shadowStrength * ndl);

    float blockLight = bakedLight.r;
    float skyLight = bakedLight.g;
    float ambientAmount = 0.08 + skyLight * 0.30 + blockLight * 0.28;
    vec3 ambientColor = mix(vec3(0.18, 0.22, 0.32), vec3(0.42, 0.48, 0.60), sunHeight);
    vec3 sunColor = mix(vec3(1.25, 0.72, 0.48), vec3(1.35, 1.24, 1.14), sunHeight);

    float horizonGlow = pow(max(0.0, 1.0 - sunHeight), 1.75);
    float assistEnergy = assist.b;
    // Another NPU-friendly seam: ambient color scales, sun warmth, and horizon energy are cheap
    // control fields. The actual shadow lookup, depth reconstruction, and view-dependent fresnel are not.
    vec3 hdr = albedo * ambientColor * ambientAmount;
    hdr += albedo * sunColor * ndl * sunShadow * (0.55 + skyLight * 1.35);
    hdr += albedo * (0.10 + assistEnergy * 0.22) * horizonGlow;
    hdr += albedo * blockLight * blockLight * (1.25 + assist.a * 0.35);

    float fresnel = pow(1.0 - saturate1(dot(worldNormal, normalize(mat3(gbufferModelViewInverse) * normalize(-viewPos)))), 3.0);
    hdr += sunColor * fresnel * ndl * sunShadow * (0.04 + assistEnergy * 0.08);

    /* RENDERTARGETS: 0 */
    gl_FragData[0] = vec4(max(hdr, vec3(0.0)), 1.0);
}
