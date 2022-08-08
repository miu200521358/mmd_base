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
        vertex_shader_src = """
            # version 330
            in layout(location = 0) vec3 positions;
            in layout(location = 1) vec3 colors;

            out vec3 newColor;

            void main() {
                gl_Position = vec4(positions, 1.0);
                newColor = colors;
            }
            """

        fragment_shader_src = """
            # version 330

            in vec3 newColor;
            out vec4  outColor;

            void main() {
                outColor = vec4(newColor, 1.0);
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
        # self.glsl_id_uv: int = gl.glGetAttribLocation(self.program, "InputUV")
        # self.glsl_id_color: int = gl.glGetUniformLocation(self.program, "InputColor")
        # self.glsl_id_texture01: int = gl.glGetUniformLocation(self.program, "Texture01")
        # self.glsl_id_alpha: int = gl.glGetUniformLocation(self.program, "Alpha")
        # self.glsl_id_is_texture: int = gl.glGetUniformLocation(
        #     self.program, "IsTexture"
        # )
        # self.glsl_id_bone_indexes: int = gl.glGetAttribLocation(
        #     self.program, "BoneIndices"
        # )
        # self.glsl_id_bone_weights: int = gl.glGetAttribLocation(
        #     self.program, "BoneWeights"
        # )

        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(0)

        gl.glVertexAttribPointer(
            1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(12)
        )
        gl.glEnableVertexAttribArray(1)
        gl.glBindVertexArray(0)

        self.glsl_id_bone_matrix: int = gl.glGetUniformLocation(
            self.program, "BoneMatrix"
        )
        # gl.glActiveTexture(gl.GL_TEXTURE0)
        # gl.glUniform1i(self.glsl_id_texture01, 0)
        self.shader_off()

    def shader_on(self) -> bool:
        if self.program:
            gl.glUseProgram(self.program)
        else:
            return False
        return True

    def shader_off(self) -> None:
        gl.glUseProgram(0)
