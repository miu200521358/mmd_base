import os
from enum import IntEnum
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from mlib.math import MMatrix4x4, MQuaternion, MVector3D


class VsLayout(IntEnum):
    POSITION_ID = 0
    NORMAL_ID = 1
    UV_ID = 2
    EXTEND_UV_ID = 3


class MShader:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.vertical_degrees = 45.0
        self.aspect_ratio = float(self.width) / float(self.height)
        self.near_plane = 0.01
        self.far_plane = 500000
        self.look_at_center = MVector3D(0.0, 0.0, 0.0)
        self.look_at_up = MVector3D(0.0, 1.0, 0.0)

        self.program = gl.glCreateProgram()
        self.compile()
        self.use()

        self.initialize()

        self.unuse()

    def __del__(self) -> None:
        gl.glDeleteProgram(self.program)

    def load_shader(self, src: str, shader_type: int) -> int:
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, src)
        gl.glCompileShader(shader)
        error = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if error != gl.GL_TRUE:
            info = gl.glGetShaderInfoLog(shader)
            gl.glDeleteShader(shader)
            raise Exception(info)
        return shader

    def compile(self) -> None:
        vertex_shader_src = Path(
            os.path.join(os.path.dirname(__file__), "pmx.vert")
        ).read_text(encoding="utf-8")
        vertex_shader_src = vertex_shader_src % (
            VsLayout.POSITION_ID.value,
            VsLayout.NORMAL_ID.value,
            VsLayout.UV_ID.value,
            VsLayout.EXTEND_UV_ID.value,
        )

        fragments_shader_src = Path(
            os.path.join(os.path.dirname(__file__), "pmx.frag")
        ).read_text(encoding="utf-8")

        vs = self.load_shader(vertex_shader_src, gl.GL_VERTEX_SHADER)
        if not vs:
            return
        fs = self.load_shader(fragments_shader_src, gl.GL_FRAGMENT_SHADER)
        if not fs:
            return
        gl.glAttachShader(self.program, vs)
        gl.glAttachShader(self.program, fs)
        gl.glLinkProgram(self.program)
        error = gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        if error != gl.GL_TRUE:
            try:
                info = gl.glGetShaderInfoLog(self.program)
            except Exception as e:
                print(e)
            raise Exception(info)

    def initialize(self):
        # light color
        light_ambient = MVector3D(0.25, 0.25, 0.25)
        light_diffuse = MVector3D(1.0, 1.0, 1.0)
        light_specular = MVector3D(1.0, 1.0, 1.0)

        # light position
        light_position = MVector3D(-0.5, -1.0, 0.5)

        # light setting
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient.vector)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse.vector)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, light_specular.vector)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position.vector)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHTING)

        gl.glEnable(gl.GL_DEPTH_TEST)  # enable shading

        # ライトの位置
        self.light_vec_uniform = gl.glGetUniformLocation(self.program, "lightPos")
        gl.glUniform3f(self.light_vec_uniform, *light_position.vector)

        # --------

        self.model_matrix = MMatrix4x4(identity=True)

        # ボーンデフォーム行列
        self.model_matrix_uniform = gl.glGetUniformLocation(self.program, "modelMatrix")

        # --------

        # カメラの位置
        self.camera_position = MVector3D(0, 10, 100)
        # カメラの回転
        self.camera_rotation = MQuaternion()

        # カメラの位置
        self.camera_vec_uniform = gl.glGetUniformLocation(self.program, "cameraPos")

        # プロジェクション行列
        # self.projection_matrix = MMatrix4x4(identity=True)
        self.projection_matrix = MMatrix4x4.perspective(
            self.vertical_degrees, self.aspect_ratio, self.near_plane, self.far_plane
        )
        self.projection_matrix_uniform = gl.glGetUniformLocation(
            self.program, "projectionMatrix"
        )
        gl.glUniformMatrix4fv(
            self.projection_matrix_uniform,
            1,
            gl.GL_FALSE,
            self.projection_matrix.vector,
        )

        # --------

        # ビュー行列
        self.view_matrix = MMatrix4x4(identity=True)
        self.view_matrix_uniform = gl.glGetUniformLocation(self.program, "viewMatrix")

        # --------

        # モデルビュー射影行列（MVP）
        self.model_view_projection_matrix = MMatrix4x4(identity=True)
        self.model_view_projection_matrix_uniform = gl.glGetUniformLocation(
            self.program, "modelViewProjectionMatrix"
        )

        # --------

        # マテリアル設定
        self.diffuse_uniform = gl.glGetUniformLocation(self.program, "diffuse")
        self.ambient_uniform = gl.glGetUniformLocation(self.program, "ambient")
        self.specular_uniform = gl.glGetUniformLocation(self.program, "specular")

        # --------

        # テクスチャの設定
        self.use_texture_uniform = gl.glGetUniformLocation(self.program, "useTexture")
        self.texture_uniform = gl.glGetUniformLocation(self.program, "textureSampler")

        # Toonの設定
        self.use_toon_uniform = gl.glGetUniformLocation(self.program, "useToon")
        self.toon_uniform = gl.glGetUniformLocation(self.program, "toonSampler")

        # Sphereの設定
        self.use_sphere_uniform = gl.glGetUniformLocation(self.program, "useSphere")
        self.sphere_mode_uniform = gl.glGetUniformLocation(self.program, "sphereMode")
        self.sphere_uniform = gl.glGetUniformLocation(self.program, "sphereSampler")

    def update_camera_uniform(self):
        # カメラ位置
        camera_mat = MMatrix4x4(identity=True)
        camera_mat.rotate(self.camera_rotation)
        camera_mat.translate(self.camera_position)
        camera_pos = camera_mat * MVector3D()

        gl.glUniform3f(self.camera_vec_uniform, *camera_pos.vector)

        # # 視点位置の決定
        # gl.glMatrixMode(gl.GL_MODELVIEW)
        # gl.glLoadIdentity()
        # glu.gluLookAt(
        #     *camera_pos.vector, *self.look_at_eye.vector, *self.look_at_up.vector
        # )

        # モデル行列
        gl.glUniformMatrix4fv(
            self.model_matrix_uniform, 1, gl.GL_FALSE, self.model_matrix.vector
        )

        # ビュー行列
        self.view_matrix = MMatrix4x4.look_at(
            camera_pos, self.look_at_center, self.look_at_up
        )
        # self.view_matrix[3, 0] = -self.camera_position.x
        # self.view_matrix[3, 1] = -self.camera_position.y
        # self.view_matrix[3, 2] = -self.camera_position.z
        self.view_matrix.translate(-self.camera_position)

        gl.glUniformMatrix4fv(
            self.view_matrix_uniform,
            1,
            gl.GL_FALSE,
            self.view_matrix.vector,
        )

        # MVP
        gl.glUniformMatrix4fv(
            self.model_view_projection_matrix_uniform,
            1,
            gl.GL_FALSE,
            (self.model_matrix * self.view_matrix * self.projection_matrix).vector,
        )

    def resize(self, width: int, height: int):
        self.use()

        self.width = width
        self.height = height
        self.aspect_ratio = float(self.width) / float(self.height)

        # ビューポートの設定
        gl.glViewport(0, 0, self.width, self.height)

        # # 視野領域の決定
        # gl.glMatrixMode(gl.GL_PROJECTION)
        # gl.glLoadIdentity()
        # glu.gluPerspective(
        #     self.vertical_degrees, self.aspect_ratio, self.near_plane, self.far_plane
        # )

        self.update_camera_uniform()

        self.unuse()

    def use(self):
        gl.glUseProgram(self.program)

    def unuse(self):
        gl.glUseProgram(0)
