# version 440

in layout(location = %d) vec3 position;
in layout(location = %d) vec3 normal;
in layout(location = %d) vec2 uv;
in layout(location = %d) vec2 extendUv;
in layout(location = %d) float vertexEdge;

uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform mat4 BoneMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform vec4 diffuse;
uniform vec3 ambient;
uniform vec4 specular;

uniform int useSphere;
uniform int sphereMode;

out vec4 vertexColor;
out vec3 vertexSpecular;
out vec2 vertexUv;
out vec3 vetexNormal;
out vec3 lightDirection;
out vec2 sphereUv;

void main() {
    // 頂点位置
    gl_Position = modelViewProjectionMatrix * vec4(position.xyz, 1.0);

    // 頂点法線
    vec3 vetexNormal = (BoneMatrix * normalize(vec4(normal.xyz, 1.0))).rgb;

    // 照明位置
    vec3 lightDirection = -normalize(lightPos);

    // 頂点色設定
    vertexColor = clamp(vec4(ambient, diffuse.w), 0.0, 1.0);
    // 色傾向
    float factor = clamp(dot(vetexNormal, lightDirection), 0.0f, 1.0f);

    // ディフューズ色＋アンビエント色 計算
    // TODO TOONでないとき、の条件付与
    vertexColor.rgb += diffuse.rgb * factor;

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

    // カメラとの相対位置
    vec3 eye = cameraPos - gl_Position.rgb;

    // スペキュラ色計算
    vec3 HalfVector = normalize( normalize(eye) + lightDirection );
    vertexSpecular = clamp(pow( max(0, dot( HalfVector, vetexNormal )), specular.w ) * specular.rgb, 0.0f, 1.0f);
}
