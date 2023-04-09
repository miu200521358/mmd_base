# version 440

in layout(location = 0) vec3 position;
in layout(location = 1) int  index;

// ボーン変形行列を格納するテクスチャ
uniform sampler2D boneMatrixTexture;

uniform mat4 modelViewProjectionMatrix;

void main() {
    vec4 position4 = vec4(position, 1.0);

    // 各頂点で使用されるボーン変形行列を計算する
    mat4 transformMatrix = mat4(0.0);
    {
        vec4 row0 = texelFetch(boneMatrixTexture, ivec2(0, index), 0);
        vec4 row1 = texelFetch(boneMatrixTexture, ivec2(1, index), 0);
        vec4 row2 = texelFetch(boneMatrixTexture, ivec2(2, index), 0);
        vec4 row3 = texelFetch(boneMatrixTexture, ivec2(3, index), 0);
        transformMatrix = mat4(row0, row1, row2, row3);
    }
    gl_Position = modelViewProjectionMatrix * transformMatrix * position4;
}