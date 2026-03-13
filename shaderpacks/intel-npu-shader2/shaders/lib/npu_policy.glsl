struct NpuPolicy {
    vec4 rawAssist;
    vec2 trajectory;
    float energy;
    float budget;
    float curvature;
};

float npuClamp1(float value) {
    return clamp(value, 0.0, 1.0);
}

vec2 npuClamp2(vec2 value) {
    return clamp(value, vec2(0.0), vec2(1.0));
}

vec2 npuSafeNormalize2(vec2 value) {
    float lengthSquared = dot(value, value);
    if (lengthSquared <= 0.000001) {
        return vec2(0.0, -1.0);
    }
    return value * inversesqrt(lengthSquared);
}

float npuDecodeBudget(float encodedAlpha) {
    return npuClamp1((encodedAlpha - 0.08) / 0.56);
}

vec4 npuSampleAssistFiltered(sampler2D npuAssist, vec2 uv) {
    vec2 clampedUv = npuClamp2(uv);
    vec2 texel = 1.0 / vec2(textureSize(npuAssist, 0));
    vec4 center = texture2D(npuAssist, clampedUv) * 0.42;
    vec4 north = texture2D(npuAssist, npuClamp2(clampedUv + vec2(0.0, texel.y))) * 0.15;
    vec4 south = texture2D(npuAssist, npuClamp2(clampedUv - vec2(0.0, texel.y))) * 0.15;
    vec4 east = texture2D(npuAssist, npuClamp2(clampedUv + vec2(texel.x, 0.0))) * 0.14;
    vec4 west = texture2D(npuAssist, npuClamp2(clampedUv - vec2(texel.x, 0.0))) * 0.14;
    return center + north + south + east + west;
}

NpuPolicy sampleNpuPolicy(sampler2D npuAssist, vec2 uv) {
    vec4 assist = npuSampleAssistFiltered(npuAssist, uv);
    vec2 rawTrajectory = assist.rg * 2.0 - 1.0;
    float trajectoryAmount = smoothstep(0.10, 0.62, length(rawTrajectory));
    NpuPolicy policy;
    policy.rawAssist = assist;
    policy.trajectory = npuSafeNormalize2(mix(vec2(0.0, -1.0), npuSafeNormalize2(rawTrajectory), trajectoryAmount));
    policy.energy = npuClamp1(0.5 + (assist.b - 0.5) * 0.78);
    policy.budget = npuClamp1(0.5 + (npuDecodeBudget(assist.a) - 0.5) * 0.72);
    policy.curvature = (assist.r - assist.g) * 0.72;
    return policy;
}

// shader2 keeps this helper for compatibility, but deliberately avoids heavy
// GPU-side reinterpretation of the low-resolution NPU policy.
NpuPolicy mergeScreenNpuPolicy(sampler2D npuAssist, vec2 uv, float depth, float baseSceneLuma, vec2 bakedLight) {
    NpuPolicy policy = sampleNpuPolicy(npuAssist, uv);
    float horizonWeight = smoothstep(0.08, 0.92, 1.0 - abs(uv.y * 2.0 - 1.0));
    float darkReceiver = smoothstep(0.25, 0.88, 1.0 - baseSceneLuma);
    float shallowDepth = 1.0 - smoothstep(0.10, 0.96, depth);

    policy.energy = npuClamp1(policy.energy * (0.92 + bakedLight.y * 0.02) + horizonWeight * 0.01 + darkReceiver * 0.02 + bakedLight.x * 0.008);
    policy.budget = npuClamp1(policy.budget * (0.92 + horizonWeight * 0.02) + shallowDepth * 0.015 + darkReceiver * 0.01);
    policy.trajectory = npuSafeNormalize2(mix(policy.trajectory, vec2(0.0, -1.0), shallowDepth * 0.10 + (1.0 - bakedLight.y) * 0.05));
    return policy;
}
