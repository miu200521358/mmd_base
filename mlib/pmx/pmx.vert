# version 440

in layout(location = %d) vec3 position;
in layout(location = %d) vec3 normal;
in layout(location = %d) vec2 uv;
in layout(location = %d) vec2 extendUv;

uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform mat4 BoneMatrix;
uniform mat4 ViewMatrix;

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
    vec4 pvec = BoneMatrix * vec4(position, 1.0);
    gl_Position = vec4( pvec.x, pvec.y, pvec.z, pvec.w );  // 座標系による反転を行う、カリングも反転

    // 頂点法線
    vec3 N = (BoneMatrix * normalize(vec4(normal, 1.0))).rgb;
    vec3 vetexNormal = vec3(N[0], N[1], N[2]); // 座標系による反転

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
            vec3 normalWv = mat3(ViewMatrix) * vetexNormal;
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
