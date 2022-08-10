import os
from enum import IntEnum
from pathlib import Path

import OpenGL.GL as gl
import OpenGL.GLU as glu


class VsLayout(IntEnum):
    POSITION_ID = 0
    NORMAL_ID = 1
    UV_ID = 2
    FACE_ID = 3

    COLOR_DIFFUSE_ID = 4
    COLOR_AMBIENT_ID = 5
    COLOR_SPECULAR_ID = 6


class MShader:
    def __init__(self, width: int, height: int) -> None:

        vertex_shader_src = Path(
            os.path.join(os.path.dirname(__file__), "pmx.vert")
        ).read_text(encoding="utf-8")
        vertex_shader_src = vertex_shader_src % (
            VsLayout.POSITION_ID.value,
            VsLayout.NORMAL_ID.value,
            VsLayout.UV_ID.value,
        )

        fragments_shader_src = Path(
            os.path.join(os.path.dirname(__file__), "pmx.frag")
        ).read_text(encoding="utf-8")

        self.vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(self.vertex_shader, vertex_shader_src)
        gl.glCompileShader(self.vertex_shader)
        gl.glGetShaderiv(self.vertex_shader, gl.GL_COMPILE_STATUS)

        self.fragments_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(self.fragments_shader, fragments_shader_src)
        gl.glCompileShader(self.fragments_shader)
        gl.glGetShaderiv(self.fragments_shader, gl.GL_COMPILE_STATUS)

        # プログラムオブジェクト作成しアタッチ
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, self.vertex_shader)
        gl.glAttachShader(self.program, self.fragments_shader)

        gl.glLinkProgram(self.program)
        gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)

        # -----------------------

        gl.glClearColor(0.7, 0.7, 0.7, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)  # enable shading

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        camera_length = 160.0

        # set perspective
        glu.gluPerspective(30.0, float(width) / float(height), 0.10, camera_length)

        # modeling transform
        gl.glMatrixMode(gl.GL_MODELVIEW)

        # light color
        light_ambient = [0.25, 0.25, 0.25]
        light_diffuse = [1.0, 1.0, 1.0]
        light_specular = [1.0, 1.0, 1.0]

        # light position
        light_position = [-0.5, -1.0, 0.5]

        # light setting
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, light_specular)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHTING)

        # -----------------
        gl.glUseProgram(self.program)

        # ボーンデフォーム行列
        self.bone_matrix_uniform = gl.glGetUniformLocation(self.program, "BoneMatrix")

        # ライトの位置
        self.light_vec_uniform = gl.glGetUniformLocation(self.program, "lightPos")
        gl.glUniform3f(self.light_vec_uniform, *light_position)

        # カメラの位置
        self.camera_vec_uniform = gl.glGetUniformLocation(self.program, "cameraPos")
        gl.glUniform3f(self.camera_vec_uniform, 0, 0, camera_length)

        # マテリアル設定
        self.diffuse_uniform = gl.glGetUniformLocation(self.program, "diffuse")
        self.ambient_uniform = gl.glGetUniformLocation(self.program, "ambient")
        self.specular_uniform = gl.glGetUniformLocation(self.program, "specular")

        # --------

        # # テクスチャの設定
        # gl.glActiveTexture(gl.GL_TEXTURE0)
        # gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        # self.texture_uniform = gl.glGetUniformLocation(self.program, "texture")

        # -----------------------
        gl.glUseProgram(0)

        gl.glDetachShader(self.program, self.vertex_shader)
        gl.glDetachShader(self.program, self.fragments_shader)

        gl.glDeleteShader(self.vertex_shader)
        gl.glDeleteShader(self.fragments_shader)
