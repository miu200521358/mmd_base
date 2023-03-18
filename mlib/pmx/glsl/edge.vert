# version 440

in layout(location = %d) vec3 position;
in layout(location = %d) vec3 normal;
in layout(location = %d) vec2 uv;
in layout(location = %d) vec2 extendUv;
in layout(location = %d) float vertexEdge;
in layout(location = %d) vec4 boneIdxs;
in layout(location = %d) vec4 boneWeights;

uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform mat4 boneMatrixes[%d];
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform float edgeSize;

void main() {
    vec4 vertexGLPosition = vec4(-position.x, position.y, position.z, 1.0);
    vec4 vertexGLNormal = vec4(-normal.x, normal.y, normal.z, 1.0);

    // 各頂点で使用されるボーン変形行列を計算する
    mat4 boneTransformMatrix = mat4(0.0);
    for(int i = 0; i < 4; i++) {
        boneTransformMatrix += boneMatrixes[int(boneIdxs[i])] * boneWeights[i];
    }

    // エッジサイズｘ頂点エッジ倍率ｘモーフ倍率＋モーフバイアス
    float edgeWight = edgeSize * vertexEdge;

    // 頂点法線
    vec3 vetexNormal = normalize(boneTransformMatrix * normalize(vertexGLNormal)).xyz;

    // 頂点位置
    gl_Position = modelViewProjectionMatrix * boneTransformMatrix * (vec4(vertexGLPosition.xyz + vertexGLNormal.xyz * edgeWight * 0.01, 1.0));
}
