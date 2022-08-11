# version 440

in layout(location = %d) vec3 position;
in layout(location = %d) vec3 normal;
in layout(location = %d) vec2 uv;
in layout(location = %d) vec2 extendUv;

uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform vec4 diffuse;
uniform vec3 ambient;
uniform vec4 specular;

uniform int useToon;
uniform int useSphere;
uniform int sphereMode;

out vec4 vertexColor;
out vec3 vertexSpecular;
out vec2 vertexUv;
out vec3 vetexNormal;
out vec3 lightDirection;
out vec2 sphereUv;

void main() {
    mat4 modelViewMatrix = viewMatrix * modelMatrix;

    // カメラ視点のワールドビュー射影変換
    gl_Position = modelViewProjectionMatrix * vec4(-position.x, position.y, position.z, 1.0);

    // カメラとの相対位置
    vec3 eye = cameraPos - (mat3(modelMatrix) * position);

    // 頂点法線
    vetexNormal = (mat3(modelMatrix) * normalize(normal));

    // 照明位置
    vec3 lightDirection = normalize(lightPos);

    // 頂点色設定
    vertexColor = clamp(vec4(ambient, diffuse.a), 0.0, 1.0);

    if (useToon == 0) {
        // ディフューズ色＋アンビエント色 計算
        float factor = clamp(dot(vetexNormal, -lightDirection), 0.0f, 1.0f);
        vertexColor.rgb += diffuse.rgb * factor;
    }
    vertexColor.a = diffuse.a;
    // saturate
    vertexColor = clamp(vertexColor, 0.0, 1.0);

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
	        sphereUv.y = normalWv.y * -0.5f + 0.5f;
        }
    }

    // スペキュラ色計算
    vec3 HalfVector = normalize( normalize(eye) + -lightDirection );
    vertexSpecular = clamp(pow( max(0, dot( HalfVector, vetexNormal )), specular.a ) * specular.rgb, 0.0f, 1.0f);
}
