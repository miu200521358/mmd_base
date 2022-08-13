import os
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
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
        self.vertical_degrees = 30.0
        self.aspect_ratio = float(self.width) / float(self.height)
        self.near_plane = 0.01
        self.far_plane = 10000
        self.look_at_center = MVector3D(0.0, 0.0, 0.0)
        self.look_at_up = MVector3D(0.0, 1.0, 0.0)

        # カメラの位置
        self.camera_position = MVector3D(0, 0, -3)
        # カメラの回転
        self.camera_rotation = MQuaternion()

        # モデル描画シェーダー ------------------
        self.model_program = gl.glCreateProgram()

        model_vertex_shader_src = Path(
            os.path.join(os.path.dirname(__file__), "glsl", "model.vert")
        ).read_text(encoding="utf-8")
        model_vertex_shader_src = model_vertex_shader_src % (
            VsLayout.POSITION_ID.value,
            VsLayout.NORMAL_ID.value,
            VsLayout.UV_ID.value,
            VsLayout.EXTEND_UV_ID.value,
        )

        model_fragments_shader_src = Path(
            os.path.join(os.path.dirname(__file__), "glsl", "model.frag")
        ).read_text(encoding="utf-8")

        self.compile(
            self.model_program, model_vertex_shader_src, model_fragments_shader_src
        )

        self.use()
        self.initialize(self.model_program)
        self.unuse()

        # TODO エッジ描画シェーダー

    def __del__(self) -> None:
        gl.glDeleteProgram(self.model_program)

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

    def compile(
        self, program: Any, vertex_shader_src: str, fragments_shader_src: str
    ) -> None:
        vs = self.load_shader(vertex_shader_src, gl.GL_VERTEX_SHADER)
        if not vs:
            return
        fs = self.load_shader(fragments_shader_src, gl.GL_FRAGMENT_SHADER)
        if not fs:
            return
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)
        error = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        if error != gl.GL_TRUE:
            info = gl.glGetShaderInfoLog(program)
            raise Exception(info)

    def initialize(self, program: Any):
        # light color
        light_diffuse = MVector3D(0.7, 0.7, 0.7)
        light_ambient = MVector3D(0.3, 0.3, 0.3)
        light_specular = MVector3D(1.0, 1.0, 1.0)

        # light position
        light_position = MVector3D(3, -2, 5)

        # light setting
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, light_ambient.vector)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, light_diffuse.vector)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, light_specular.vector)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position.vector)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHTING)

        gl.glEnable(gl.GL_DEPTH_TEST)  # enable shading
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # ボーンデフォーム行列
        self.bone_matrix_uniform = gl.glGetUniformLocation(program, "BoneMatrix")

        # モデルビュー行列
        self.model_view_matrix_uniform = gl.glGetUniformLocation(
            program, "modelViewMatrix"
        )

        # MVP行列
        self.model_view_projection_matrix_uniform = gl.glGetUniformLocation(
            program, "modelViewProjectionMatrix"
        )

        # ライトの位置
        self.light_vec_uniform = gl.glGetUniformLocation(program, "lightPos")
        gl.glUniform3f(self.light_vec_uniform, *light_position.vector)

        # カメラの位置
        self.camera_vec_uniform = gl.glGetUniformLocation(program, "cameraPos")

        # --------

        # マテリアル設定
        self.diffuse_uniform = gl.glGetUniformLocation(program, "diffuse")
        self.ambient_uniform = gl.glGetUniformLocation(program, "ambient")
        self.specular_uniform = gl.glGetUniformLocation(program, "specular")

        # --------

        # テクスチャの設定
        self.use_texture_uniform = gl.glGetUniformLocation(program, "useTexture")
        self.texture_uniform = gl.glGetUniformLocation(program, "textureSampler")

        # Toonの設定
        self.use_toon_uniform = gl.glGetUniformLocation(program, "useToon")
        self.toon_uniform = gl.glGetUniformLocation(program, "toonSampler")

        # Sphereの設定
        self.use_sphere_uniform = gl.glGetUniformLocation(program, "useSphere")
        self.sphere_mode_uniform = gl.glGetUniformLocation(program, "sphereMode")
        self.sphere_uniform = gl.glGetUniformLocation(program, "sphereSampler")

        self.fit(self.width, self.height)

    def update_camera(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.4, 0.4, 0.4, 1)

        # カメラ位置
        camera_mat = MMatrix4x4(identity=True)
        camera_mat.rotate(self.camera_rotation)
        camera_mat.translate(self.camera_position)
        camera_pos: MVector3D = camera_mat * MVector3D()

        gl.glUniform3f(self.camera_vec_uniform, *camera_pos.vector)

        # 視点位置の決定
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(
            *camera_pos.vector, *self.look_at_center.vector, *self.look_at_up.vector
        )

        model_view_matrix = np.array(
            gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX), dtype=np.float32
        )
        gl.glUniformMatrix4fv(
            self.model_view_matrix_uniform,
            1,
            gl.GL_FALSE,
            model_view_matrix,
        )

        gl.glUniformMatrix4fv(
            self.model_view_projection_matrix_uniform,
            1,
            gl.GL_FALSE,
            np.matmul(model_view_matrix, self.projection_matrix),
        )

    def fit(self, width: int, height: int):
        self.use()

        self.width = width
        self.height = height
        self.aspect_ratio = float(self.width) / float(self.height)

        # ビューポートの設定
        gl.glViewport(0, 0, self.width, self.height)

        # 視野領域の決定
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(
            self.vertical_degrees, self.aspect_ratio, self.near_plane, self.far_plane
        )

        self.projection_matrix = np.array(
            gl.glGetFloatv(gl.GL_PROJECTION_MATRIX), dtype=np.float32
        )

        self.update_camera()

        self.unuse()

    def use(self):
        gl.glUseProgram(self.model_program)

    def unuse(self):
        gl.glUseProgram(0)
