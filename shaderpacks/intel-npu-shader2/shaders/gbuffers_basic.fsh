#version 150 compatibility

in vec4 tintColor;
in vec3 worldNormal;

vec3 toLinear(vec3 color) {
    return pow(clamp(color, vec3(0.0), vec3(1.0)), vec3(2.2));
}

void main() {
    vec3 encodedNormal = normalize(worldNormal) * 0.5 + 0.5;

    /* RENDERTARGETS: 0,1,2,4 */
    gl_FragData[0] = vec4(toLinear(tintColor.rgb), 1.0);
    gl_FragData[1] = vec4(encodedNormal, 1.0);
    gl_FragData[2] = vec4(0.0, 0.0, 0.0, 1.0);
    gl_FragData[3] = vec4(0.5, 0.5, 0.0, 1.0);
}
