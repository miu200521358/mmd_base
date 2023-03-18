# version 440

in layout(location = %d) vec3 position;
in layout(location = %d) vec3 normal;
in layout(location = %d) vec2 uv;
in layout(location = %d) vec2 extendUv;
in layout(location = %d) float vertexEdge;
in layout(location = %d) vec4 boneIdxs;
in layout(location = %d) vec4 boneWeights;

uniform vec3 cameraPos;
uniform mat4 boneMatrixes[%d];
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform vec4 diffuse;
uniform vec3 ambient;
uniform vec4 specular;

uniform int useToon;
uniform int useSphere;
uniform int sphereMode;
uniform vec3 lightDirection;

out vec4 vertexColor;
out vec3 vertexSpecular;
out vec2 vertexUv;
out vec3 vetexNormal;
out vec2 sphereUv;
out vec3 eye;

void main() {
    vec4 vertexGLNormal = vec4(-normal.x, normal.y, normal.z, 1.0);

    // 各頂点で使用されるボーン変形行列を計算する
    mat4 boneTransformMatrix = mat4(0.0);
    for(int i = 0; i < 4; i++) {
        boneTransformMatrix += boneMatrixes[int(boneIdxs[i])] * boneWeights[i];
    }

    // OpenGL的座標
    vec4 vertexGLPosition = vec4(position.x, position.y, position.z, 1.0);
    vec4 transformPosition = boneTransformMatrix * vertexGLPosition;
    transformPosition.x *= -1;

    // 頂点位置
    gl_Position = modelViewProjectionMatrix * transformPosition;

    // 頂点法線
    vetexNormal = normalize(boneTransformMatrix * normalize(vertexGLNormal)).xyz;

    // 頂点色設定
    vertexColor = clamp(diffuse, 0.0, 1.0);

    if (useToon == 0) {
        // ディフューズ色＋アンビエント色 計算
        float lightNormal = clamp(dot( vetexNormal, -lightDirection ), 0.0, 1.0);
        vertexColor.rgb += diffuse.rgb * lightNormal;
    }

    // テクスチャ描画位置
    vertexUv = uv;

    if (useSphere == 1) {
        // Sphereマップ計算
        if (sphereMode == 3) {
            // PMXサブテクスチャ座標
            sphereUv = extendUv;
        }
        else {
	        // スフィアマップテクスチャ座標
            vec3 normalWv = mat3(modelViewMatrix) * vetexNormal;
	        sphereUv.x = normalWv.x * 0.5f + 0.5f;
	        sphereUv.y = 1 - (normalWv.y * -0.5f + 0.5f);
        }
    }

    // カメラとの相対位置
    vec3 eye = cameraPos - (boneTransformMatrix * vertexGLPosition).xyz;

    // スペキュラ色計算
    vec3 HalfVector = normalize( normalize(eye) + -lightDirection );
    vertexSpecular = pow( max(0, dot( HalfVector, vetexNormal )), specular.w ) * specular.rgb;
}
