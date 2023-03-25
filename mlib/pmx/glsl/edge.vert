# version 440

in layout(location = %d) vec3 position;
in layout(location = %d) vec3 normal;
in layout(location = %d) vec2 uv;
in layout(location = %d) vec2 extendUv;
in layout(location = %d) float vertexEdge;
in layout(location = %d) vec4 boneIdxs;
in layout(location = %d) vec4 boneWeights;

// ボーン変形行列を格納するテクスチャ
uniform sampler2D boneMatrixTexture;

uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform float edgeSize;

void main() {
    // 各頂点で使用されるボーン変形行列を計算する
    mat4 boneTransformMatrix = mat4(0.0);
    for(int i = 0; i < 4; i++) {
        float boneWeight = boneWeights[i];
        if (boneWeight <= 0.0) {
            continue;
        }
        int boneIndex = int(boneIdxs[i]);

        vec4 row0 = texelFetch(boneMatrixTexture, ivec2(0, boneIndex), 0);
        vec4 row1 = texelFetch(boneMatrixTexture, ivec2(1, boneIndex), 0);
        vec4 row2 = texelFetch(boneMatrixTexture, ivec2(2, boneIndex), 0);
        vec4 row3 = texelFetch(boneMatrixTexture, ivec2(3, boneIndex), 0);
        mat4 boneMatrix = mat4(row0, row1, row2, row3);
        boneTransformMatrix += boneMatrix * boneWeight;
    }

    // エッジサイズｘ頂点エッジ倍率ｘモーフ倍率＋モーフバイアス
    float edgeWight = edgeSize * vertexEdge;

    // 頂点位置
    gl_Position = modelViewProjectionMatrix * boneTransformMatrix * (vec4(position + (normal * edgeWight * 0.02), 1.0));
}
