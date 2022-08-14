# version 440

in layout(location = %d) vec3 position;
in layout(location = %d) vec3 normal;
in layout(location = %d) vec2 uv;
in layout(location = %d) vec2 extendUv;
in layout(location = %d) float vertexEdge;

uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

uniform float edgeSize;

void main() {
    // エッジサイズｘ頂点エッジ倍率ｘモーフ倍率＋モーフバイアス
    float edgeWight = 1 + 0.002 * edgeSize * vertexEdge;

    // 頂点位置(座標系による反転)
    gl_Position = modelViewProjectionMatrix * (vec4(position.xyz * edgeWight, 1.0));
}
