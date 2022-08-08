import OpenGL.GL as gl
from mlib.base.base import BaseModel

MESH_BONE_LIMIT = 20


class MShader(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self.program = None
        if not self.compile():
            raise IOError

    def compile(self) -> bool:
        vertex_shader_src = (
            """
            #version 110
            // uniform     mat4    ProjectionMatrix;
            mat4                ProjectionMatrix = gl_ModelViewProjectionMatrix;
            attribute   vec3    Vertex;
            attribute   vec2    InputUV;
            varying     vec2    ShareUV;
            uniform     vec4    InputColor;
            varying     vec4    ShareColor;
            attribute   vec4    BoneWeights;
            attribute   vec4    BoneIndices;
            uniform     mat4    BoneMatrix[%d];

            void main( void )
            {
                mat4            skinTransform;
                vec4            pvec;

                skinTransform  = BoneWeights[ 0 ] * BoneMatrix[ int( BoneIndices[ 0 ] ) ];
                skinTransform += BoneWeights[ 1 ] * BoneMatrix[ int( BoneIndices[ 1 ] ) ];
                skinTransform += BoneWeights[ 2 ] * BoneMatrix[ int( BoneIndices[ 2 ] ) ];
                skinTransform += BoneWeights[ 3 ] * BoneMatrix[ int( BoneIndices[ 3 ] ) ];

                pvec = ProjectionMatrix * skinTransform * vec4( Vertex, 1.00 );
                gl_Position = vec4( -pvec[ 0 ], pvec[ 1 ], pvec[ 2 ], pvec[ 3 ] );  // 座標系による反転を行う、カリングも反転
                ShareUV = InputUV;
                ShareColor = InputColor;
            }
        """
            % MESH_BONE_LIMIT
        )

        fragment_shader_src = """
            #version 110
            uniform     sampler2D   Texture01;
            varying     vec2        ShareUV;
            varying     vec4        ShareColor;
            uniform     float       IsTexture;
            uniform     float       Alpha;

            void main( void )
            {
                vec4    tex_color = texture2D( Texture01, ShareUV );
                // tex_color += ( 1.00 - IsTexture ) * ShareColor;
                tex_color.a = tex_color.a * Alpha;
                //gl_FragColor = tex_color * ( 1.00 - ShareColor.a ) + ShareColor * ShareColor.a;
                gl_FragColor = tex_color * ( 1.00 - ShareColor.a ) + ShareColor * ShareColor.a;
            }
        """
        self.program = gl.glCreateProgram()
        if not self.create_shader(gl.GL_VERTEX_SHADER, vertex_shader_src):
            # FIXME
            print("GL_VERTEX_SHADER ERROR")
            return False
        if not self.create_shader(gl.GL_FRAGMENT_SHADER, fragment_shader_src):
            print("GL_FRAGMENT_SHADER ERROR")
            return False
        else:
            if not self.link():
                print("SHADER LINK ERROR")
                return False
            self.create_variable()
            return True

    def create_shader(self, shader_type: gl.Constant, src: str) -> bool:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, src)
        gl.glCompileShader(shader)
        r = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if r == 0:
            return False
        gl.glAttachShader(self.program, shader)
        gl.glDeleteShader(shader)
        return True

    def link(self) -> bool:
        if self.program:
            gl.glLinkProgram(self.program)
        r = gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
        if r == 0:
            gl.glDeleteProgram(self.program)
            return False
        else:
            gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
            gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
            return True

    def create_variable(self) -> None:
        self.shader_on()
        self.glsl_id_vertex: int = gl.glGetAttribLocation(self.program, "Vertex")
        self.glsl_id_uv: int = gl.glGetAttribLocation(self.program, "InputUV")
        self.glsl_id_color: int = gl.glGetUniformLocation(self.program, "InputColor")
        self.glsl_id_texture01: int = gl.glGetUniformLocation(self.program, "Texture01")
        self.glsl_id_alpha: int = gl.glGetUniformLocation(self.program, "Alpha")
        self.glsl_id_is_texture: int = gl.glGetUniformLocation(
            self.program, "IsTexture"
        )
        self.glsl_id_bone_indexes: int = gl.glGetAttribLocation(
            self.program, "BoneIndices"
        )
        self.glsl_id_bone_weights: int = gl.glGetAttribLocation(
            self.program, "BoneWeights"
        )
        self.glsl_id_bone_matrix: int = gl.glGetUniformLocation(
            self.program, "BoneMatrix"
        )
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1i(self.glsl_id_texture01, 0)
        self.shader_off()

    def shader_on(self) -> bool:
        if self.program:
            gl.glUseProgram(self.program)
        else:
            return False
        return True

    def shader_off(self) -> None:
        gl.glUseProgram(0)
