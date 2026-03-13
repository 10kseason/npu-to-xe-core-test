#version 150 compatibility

#include "/lib/distort.glsl"
#include "/lib/npu_policy.glsl"

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

float luma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

int budgetTapCount(float budget, int minTaps, int maxTaps) {
    float clampedBudget = saturate1(budget);
    return int(floor(mix(float(minTaps), float(maxTaps) + 0.999, clampedBudget)));
}

float sampleDepthDiscontinuity(vec2 uv, float centerDepth) {
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float maxGap = 0.0;
    maxGap = max(maxGap, abs(centerDepth - texture2D(depthtex0, saturate2(uv + vec2(pixel.x, 0.0))).r));
    maxGap = max(maxGap, abs(centerDepth - texture2D(depthtex0, saturate2(uv - vec2(pixel.x, 0.0))).r));
    maxGap = max(maxGap, abs(centerDepth - texture2D(depthtex0, saturate2(uv + vec2(0.0, pixel.y))).r));
    maxGap = max(maxGap, abs(centerDepth - texture2D(depthtex0, saturate2(uv - vec2(0.0, pixel.y))).r));
    float farFactor = smoothstep(0.60, 0.985, centerDepth);
    float minThreshold = mix(0.0012, 0.00018, farFactor);
    float maxThreshold = mix(0.022, 0.0048, farFactor);
    return smoothstep(minThreshold, maxThreshold, maxGap);
}

float sampleShadowMap(vec3 shadowCoord, float texelRadius, int kernelRadius) {
    if (shadowCoord.x <= 0.0 || shadowCoord.x >= 1.0 || shadowCoord.y <= 0.0 || shadowCoord.y >= 1.0) {
        return 1.0;
    }
    if (shadowCoord.z <= 0.0 || shadowCoord.z >= 1.0) {
        return 1.0;
    }

    float visibility = 0.0;
    float sampleCount = 0.0;
    vec2 texel = vec2(texelRadius / float(shadowMapResolution));
    float compareDepth = shadowCoord.z;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            if (abs(x) > kernelRadius || abs(y) > kernelRadius) {
                continue;
            }
            vec2 offset = vec2(float(x), float(y)) * texel;
            float shadowDepth = texture2D(shadowtex0, shadowCoord.xy + offset).r;
            visibility += shadowDepth + 0.0015 >= compareDepth ? 1.0 : 0.0;
            sampleCount += 1.0;
        }
    }
    return visibility / max(sampleCount, 1.0);
}

float sampleScreenSunShadow(vec2 uv, float depth, vec2 sunScreenDir, float radiusPixels, int sampleCount) {
    if (length(sunScreenDir) < 0.0001) {
        return 1.0;
    }

    vec2 dir = normalize(-sunScreenDir);
    vec2 pixel = vec2(1.0 / max(viewWidth, 1.0), 1.0 / max(viewHeight, 1.0));
    float occlusion = 0.0;
    for (int index = 1; index <= 6; ++index) {
        if (index > sampleCount) {
            break;
        }
        float travel = float(index) * radiusPixels;
        vec2 sampleUv = saturate2(uv + dir * pixel * travel);
        float sampleDepth = texture2D(depthtex0, sampleUv).r;
        float depthGap = depth - sampleDepth;
        occlusion += smoothstep(0.0015, 0.02 + float(index) * 0.004, depthGap);
    }
    return 1.0 - occlusion / max(float(sampleCount), 1.0);
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
    vec4 lightData = texture2D(colortex2, uv);
    vec2 bakedLight = lightData.rg;
    float entityMask = lightData.b;
    // shader2 trusts the low-resolution NPU policy more directly and keeps only
    // the live depth and shadow comparisons on the GPU.
    NpuPolicy policy = mergeScreenNpuPolicy(npuAssist, uv, depth, luma(albedo), bakedLight);
    float farFieldSuppression = smoothstep(0.72, 0.985, depth);
    float depthDiscontinuity = sampleDepthDiscontinuity(uv, depth);
    float assistFade = saturate1(entityMask * 0.75 + farFieldSuppression * 0.48 + depthDiscontinuity * 0.36);
    float effectiveBudget = mix(policy.budget, min(policy.budget, 0.30), assistFade);

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
    float pcfRadius = mix(0.92, 3.00, effectiveBudget);
    int shadowKernelRadius = effectiveBudget >= 0.64 ? 1 : 0;
    int screenShadowSamples = budgetTapCount(effectiveBudget, 1, 3);
    float shadowVisibility = sampleShadowMap(shadowCoord, pcfRadius, shadowKernelRadius);
    float screenShadowRaw = sampleScreenSunShadow(
        uv,
        depth,
        shadowLightPosition.xy,
        mix(5.0, 14.0, effectiveBudget) * mix(1.0, 0.26, farFieldSuppression),
        screenShadowSamples
    );
    float screenShadow = mix(1.0, screenShadowRaw, 1.0 - assistFade * 0.82);
    float assistEnergy = mix(policy.energy, min(policy.energy, 0.38), entityMask);
    assistEnergy = mix(assistEnergy, min(assistEnergy, 0.22), farFieldSuppression * 0.65 + depthDiscontinuity * 0.30);
    float shadowDensity = mix(0.86, 1.12, effectiveBudget * 0.50 + assistEnergy * 0.24);
    float shadowContrast = mix(1.65, 1.25, sunHeight);
    float shadowTerm = pow(saturate1(min(shadowVisibility, screenShadow)), shadowContrast);
    float shadowStrength = saturate1(mix(0.80, 1.00, sunHeight) * shadowDensity * (0.42 + ndl * 0.58));
    float sunShadow = mix(1.0, shadowTerm, shadowStrength);

    float blockLight = bakedLight.r;
    float skyLight = bakedLight.g;
    float ambientAmount = 0.08 + skyLight * 0.30 + blockLight * 0.28;
    vec3 ambientColor = mix(vec3(0.18, 0.22, 0.32), vec3(0.42, 0.48, 0.60), sunHeight);
    vec3 sunColor = mix(vec3(1.25, 0.72, 0.48), vec3(1.35, 1.24, 1.14), sunHeight);

    float horizonGlow = pow(max(0.0, 1.0 - sunHeight), 1.75);
    // Another NPU-friendly seam: ambient color scales, sun warmth, and horizon energy are cheap
    // control fields. The actual shadow lookup, depth reconstruction, and view-dependent fresnel are not.
    float shadowAmbient = mix(0.62, 1.0, sunShadow);
    vec3 hdr = albedo * ambientColor * ambientAmount * shadowAmbient;
    hdr += albedo * sunColor * ndl * sunShadow * (0.55 + skyLight * 1.35);
    hdr += albedo * (0.05 + assistEnergy * 0.22 + effectiveBudget * 0.08) * horizonGlow * (1.0 - assistFade * 0.60);
    hdr += albedo * blockLight * blockLight * (1.00 + effectiveBudget * 0.54 + assistEnergy * 0.08) * (1.0 - assistFade * 0.36);

    vec3 viewDirWorld = normalize(mat3(gbufferModelViewInverse) * normalize(-viewPos));
    float fresnel = pow(1.0 - saturate1(dot(worldNormal, viewDirWorld)), 3.0);
    hdr += sunColor * fresnel * ndl * sunShadow * (0.03 + assistEnergy * 0.05) * (1.0 - assistFade * 0.44);

    // Feed the final GI stage with a stronger indirect-light seed in darker regions.
    // The NPU predicts low-frequency GI direction/energy, while the GPU keeps the
    // live normal/view/depth work and builds the local shadowed color basis here.
    float shadowLift = (1.0 - sunShadow);
    vec3 shadowBounceSeed = albedo * mix(ambientColor, sunColor, 0.18) * shadowLift * (0.05 + assistEnergy * 0.20 + effectiveBudget * 0.16) * (1.0 - assistFade * 0.74);
    hdr += shadowBounceSeed;

    /* RENDERTARGETS: 0 */
    gl_FragData[0] = vec4(max(hdr, vec3(0.0)), 1.0);
}
