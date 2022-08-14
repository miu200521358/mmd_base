# version 440

uniform mat4 modelMatrix;

uniform int useTexture;
uniform sampler2D textureSampler;
uniform vec4 diffuse;
uniform vec3 ambient;

uniform int useToon;
uniform sampler2D toonSampler;

uniform int useSphere;
uniform int sphereMode;
uniform sampler2D sphereSampler;

uniform vec3 lightDirection;

in vec4 vertexColor;
in vec3 vertexSpecular;
in vec2 vertexUv;
in vec3 vetexNormal;
in vec2 sphereUv;

out vec4  outColor;

void main() {
    outColor = vertexColor;
    
    if (useTexture == 1) {
        // テクスチャ適用
        outColor *= texture(textureSampler, vertexUv);
    }

    if (useSphere == 1) {
        // Sphere適用
        vec4 texColor = texture(sphereSampler, sphereUv);
        if (sphereMode == 2) {
            // スフィア加算
            outColor.rgb += texColor.rgb;
        }
        else {
            // スフィア乗算
            outColor.rgb *= texColor.rgb;
        }
        texColor.a *= texColor.a;
    }

    if (useToon == 1) {
        // Toon適用
        float lightNormal = dot( vetexNormal, -lightDirection );
        outColor *= texture(toonSampler, vec2(0, lightNormal));
    }

    // スペキュラ適用
    outColor.rgb += vertexSpecular;
}