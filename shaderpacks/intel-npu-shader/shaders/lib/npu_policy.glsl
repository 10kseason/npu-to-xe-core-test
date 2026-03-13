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

NpuPolicy sampleNpuPolicy(sampler2D npuAssist, vec2 uv) {
    vec4 assist = texture2D(npuAssist, npuClamp2(uv));
    NpuPolicy policy;
    policy.rawAssist = assist;
    policy.trajectory = npuSafeNormalize2(assist.rg * 2.0 - 1.0);
    policy.energy = npuClamp1(assist.b);
    policy.budget = npuDecodeBudget(assist.a);
    policy.curvature = (assist.r - assist.g) * 0.85;
    return policy;
}

NpuPolicy mergeScreenNpuPolicy(sampler2D npuAssist, vec2 uv, float depth, float baseSceneLuma, vec2 bakedLight) {
    NpuPolicy policy = sampleNpuPolicy(npuAssist, uv);
    float depthWeight = smoothstep(0.12, 0.98, depth);
    float shadowReceiver = smoothstep(0.18, 0.86, 1.0 - baseSceneLuma);
    float horizonWeight = smoothstep(0.05, 0.95, 1.0 - abs(uv.y * 2.0 - 1.0));

    policy.energy = npuClamp1(
        policy.energy * (0.88 + bakedLight.y * 0.04)
        + shadowReceiver * 0.08
        + bakedLight.x * 0.04
        + horizonWeight * 0.03
    );
    policy.budget = npuClamp1(
        policy.budget * (0.86 + horizonWeight * 0.04)
        + shadowReceiver * 0.07
        + (1.0 - bakedLight.y) * 0.05
        + depthWeight * 0.02
    );
    policy.trajectory = npuSafeNormalize2(
        mix(policy.trajectory, vec2(0.0, -1.0), (1.0 - depthWeight) * 0.08 + (1.0 - bakedLight.y) * 0.04)
    );
    return policy;
}
