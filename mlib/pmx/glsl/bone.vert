# version 440

in layout(location = 0) vec3 position;

// ボーン変形行列を格納するテクスチャ
uniform sampler2D boneMatrixTexture;

void main() {
    vec4 position4 = vec4(position, 1.0);
    // // 各頂点で使用されるボーン変形行列を計算する
    // mat4 childTransformMatrix = mat4(0.0);
    // {
    //     vec4 row0 = texelFetch(boneMatrixTexture, ivec2(0, childIndex), 0);
    //     vec4 row1 = texelFetch(boneMatrixTexture, ivec2(1, childIndex), 0);
    //     vec4 row2 = texelFetch(boneMatrixTexture, ivec2(2, childIndex), 0);
    //     vec4 row3 = texelFetch(boneMatrixTexture, ivec2(3, childIndex), 0);
    //     childTransformMatrix = mat4(row0, row1, row2, row3);
    // }
    // vec4 glChildPosition = modelViewProjectionMatrix * (childTransformMatrix * childPosition);

    // // 各頂点で使用されるボーン変形行列を計算する
    // mat4 parentTransformMatrix = mat4(0.0);
    // {
    //     vec4 row0 = texelFetch(boneMatrixTexture, ivec2(0, parentIndex), 0);
    //     vec4 row1 = texelFetch(boneMatrixTexture, ivec2(1, parentIndex), 0);
    //     vec4 row2 = texelFetch(boneMatrixTexture, ivec2(2, parentIndex), 0);
    //     vec4 row3 = texelFetch(boneMatrixTexture, ivec2(3, parentIndex), 0);
    //     parentTransformMatrix = mat4(row0, row1, row2, row3);
    // }
    // vec4 glParentPosition = modelViewProjectionMatrix * (parentTransformMatrix * parentPosition);

    // // 直線の方向ベクトルを計算
    // vec3 direction = normalize(glChildPosition - glParentPosition);

    // 頂点の位置を直線に対して正規化した値でスケーリングすることで、
    // 直線上の点の位置を求める
    gl_Position = position4;
}