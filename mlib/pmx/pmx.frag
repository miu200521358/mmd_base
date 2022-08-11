# version 440

uniform mat4 BoneMatrix;
// uniform sampler2D textureSampler;

in vec4 vertexColor;
in vec3 vertexSpecular;
in vec2 vertexUv;
out vec4  outColor;

void main() {
    // outColor = vec4(1, 0, 0, 1);
    outColor = vertexColor;
    // テクスチャ適用
    // outColor *= texture(textureSampler, vertexUv).rgba;
    // スペキュラ適用
    outColor.rgb += vertexSpecular;
}