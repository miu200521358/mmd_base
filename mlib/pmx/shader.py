import os
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu

from mlib.base.math import MMatrix4x4, MQuaternion, MVector3D, MVector4D


class VsLayout(IntEnum):
    POSITION_ID = 0
    NORMAL_ID = 1
    UV_ID = 2
    EXTEND_UV_ID = 3
    EDGE_ID = 4
    BONE_ID = 5
    WEIGHT_ID = 6
    MORPH_POS_ID = 7
    MORPH_UV_ID = 8
    MORPH_UV1_ID = 9


class Msaa:
    """
    MSAA(アンチエイリアス)
    https://blog.techlab-xe.net/opengl%E3%81%A7msaa/
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.msaa_samples = 4

        # MSAA用のフレームバッファオブジェクトを作成する
        self.msaa_buffer = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.msaa_buffer)

        # カラーバッファと深度バッファをMSAAで使うテクスチャに割り当てる
        self.msaa_color_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.msaa_color_buffer)
        gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, self.msaa_samples, gl.GL_RGBA, self.width, self.height)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER,
            self.msaa_color_buffer,
        )

        self.msaa_depth_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.msaa_depth_buffer)
        gl.glRenderbufferStorageMultisample(
            gl.GL_RENDERBUFFER,
            self.msaa_samples,
            gl.GL_DEPTH_COMPONENT,
            self.width,
            self.height,
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self.msaa_depth_buffer,
        )

        # 描画先テクスチャのバインドを解除しておく
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def bind(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # フレームバッファオブジェクトをバインドする
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.msaa_buffer)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def unbind(self):
        # フレームバッファオブジェクトの内容を画面に描画する
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.msaa_buffer)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        gl.glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
            gl.GL_COLOR_BUFFER_BIT,
            gl.GL_NEAREST,
        )


class MShader:
    INITIAL_VERTICAL_DEGREES = 40.0
    INITIAL_CAMERA_POSITION_Y = 11.0
    INITIAL_CAMERA_POSITION_Z = -40.0
    INITIAL_LOOK_AT_CENTER_Y = INITIAL_CAMERA_POSITION_Y * 1.1
    INITIAL_CAMERA_POSITION_X = 40.0

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.vertical_degrees = self.INITIAL_VERTICAL_DEGREES
        self.aspect_ratio = float(self.width) / float(self.height)
        self.near_plane = 1
        self.far_plane = 100
        self.look_at_center = MVector3D(0.0, self.INITIAL_LOOK_AT_CENTER_Y, 0.0)
        self.look_at_up = MVector3D(0.0, 1.0, 0.0)

        # カメラの位置
        self.camera_position = MVector3D(
            0.0,
            self.INITIAL_CAMERA_POSITION_Y,
            self.INITIAL_CAMERA_POSITION_Z,
        )
        # カメラの回転
        self.camera_rotation = MQuaternion()

        # light position
        self.light_position = MVector3D(-20, self.INITIAL_CAMERA_POSITION_Y * 2, self.INITIAL_CAMERA_POSITION_Z * 2)
        self.light_direction = (self.light_position * MVector3D(-1, -1, -1)).normalized()

        self.bone_matrix_texture_uniform: dict[bool, Any] = {}
        self.bone_matrix_texture_id: dict[bool, Any] = {}
        self.model_view_matrix_uniform: dict[bool, Any] = {}
        self.model_view_projection_matrix_uniform: dict[bool, Any] = {}
        self.light_direction_uniform: dict[bool, Any] = {}
        self.camera_vec_uniform: dict[bool, Any] = {}
        self.diffuse_uniform: dict[bool, Any] = {}
        self.ambient_uniform: dict[bool, Any] = {}
        self.specular_uniform: dict[bool, Any] = {}
        self.edge_color_uniform: dict[bool, Any] = {}
        self.edge_size_uniform: dict[bool, Any] = {}
        self.use_texture_uniform: dict[bool, Any] = {}
        self.texture_uniform: dict[bool, Any] = {}
        self.texture_factor_uniform: dict[bool, Any] = {}
        self.use_toon_uniform: dict[bool, Any] = {}
        self.toon_uniform: dict[bool, Any] = {}
        self.toon_factor_uniform: dict[bool, Any] = {}
        self.use_sphere_uniform: dict[bool, Any] = {}
        self.sphere_mode_uniform: dict[bool, Any] = {}
        self.sphere_uniform: dict[bool, Any] = {}
        self.sphere_factor_uniform: dict[bool, Any] = {}

        # モデル描画シェーダー ------------------
        self.model_program = gl.glCreateProgram()
        self.compile(self.model_program, "model.vert", "model.frag")

        # 初期化
        self.use()
        self.initialize(self.model_program)
        self.unuse()

        # エッジ描画シェーダー ------------------
        self.edge_program = gl.glCreateProgram()
        self.compile(self.edge_program, "edge.vert", "edge.frag")

        # 初期化
        self.use(edge=True)
        self.initialize(self.edge_program, edge=True)
        self.unuse()

        # フィット（両方）
        self.fit(self.width, self.height)

    def __del__(self) -> None:
        gl.glDeleteProgram(self.model_program)
        gl.glDeleteProgram(self.edge_program)

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

    def compile(self, program: Any, vertex_shader_name: str, fragments_shader_name: str) -> None:
        vertex_shader_src = Path(os.path.join(os.path.dirname(__file__), "glsl", vertex_shader_name)).read_text(encoding="utf-8")
        vertex_shader_src = vertex_shader_src % (
            VsLayout.POSITION_ID.value,
            VsLayout.NORMAL_ID.value,
            VsLayout.UV_ID.value,
            VsLayout.EXTEND_UV_ID.value,
            VsLayout.EDGE_ID.value,
            VsLayout.BONE_ID.value,
            VsLayout.WEIGHT_ID.value,
            VsLayout.MORPH_POS_ID.value,
            VsLayout.MORPH_UV_ID.value,
            VsLayout.MORPH_UV1_ID.value,
        )

        fragments_shader_src = Path(os.path.join(os.path.dirname(__file__), "glsl", fragments_shader_name)).read_text(encoding="utf-8")

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

    def initialize(self, program: Any, edge=False):
        # light color
        # MMD Light Diffuse は必ず0
        self.light_diffuse = MVector3D()
        # MMDの照明色そのまま
        self.light_ambient = MVector3D(154 / 255, 154 / 255, 154 / 255)
        self.light_specular = self.light_ambient
        # light_diffuse == MMDのambient
        self.light_ambient4 = MVector4D(
            self.light_ambient.x,
            self.light_ambient.y,
            self.light_ambient.z,
            1,
        )

        # モデルビュー行列
        self.model_view_matrix_uniform[edge] = gl.glGetUniformLocation(program, "modelViewMatrix")

        # MVP行列
        self.model_view_projection_matrix_uniform[edge] = gl.glGetUniformLocation(program, "modelViewProjectionMatrix")

        # ボーン変形行列用テクスチャ

        # テクスチャを作成する
        self.bone_matrix_texture_id[edge] = gl.glGenTextures(1)
        self.bone_matrix_texture_uniform[edge] = gl.glGetUniformLocation(program, "boneMatrixTexture")

        self.msaa = Msaa(self.width, self.height)

        if not edge:
            # モデルシェーダーへの割り当て

            self.light_direction_uniform[edge] = gl.glGetUniformLocation(program, "lightDirection")
            gl.glUniform3f(self.light_direction_uniform[edge], *self.light_direction.vector)

            # カメラの位置
            self.camera_vec_uniform[edge] = gl.glGetUniformLocation(program, "cameraPos")

            # --------

            # マテリアル設定
            self.diffuse_uniform[edge] = gl.glGetUniformLocation(program, "diffuse")
            self.ambient_uniform[edge] = gl.glGetUniformLocation(program, "ambient")
            self.specular_uniform[edge] = gl.glGetUniformLocation(program, "specular")

            # --------

            # テクスチャの設定
            self.use_texture_uniform[edge] = gl.glGetUniformLocation(program, "useTexture")
            self.texture_uniform[edge] = gl.glGetUniformLocation(program, "textureSampler")
            self.texture_factor_uniform[edge] = gl.glGetUniformLocation(program, "textureFactor")

            # Toonの設定
            self.use_toon_uniform[edge] = gl.glGetUniformLocation(program, "useToon")
            self.toon_uniform[edge] = gl.glGetUniformLocation(program, "toonSampler")
            self.toon_factor_uniform[edge] = gl.glGetUniformLocation(program, "toonFactor")

            # Sphereの設定
            self.use_sphere_uniform[edge] = gl.glGetUniformLocation(program, "useSphere")
            self.sphere_mode_uniform[edge] = gl.glGetUniformLocation(program, "sphereMode")
            self.sphere_uniform[edge] = gl.glGetUniformLocation(program, "sphereSampler")
            self.sphere_factor_uniform[edge] = gl.glGetUniformLocation(program, "sphereFactor")

            # --------
        else:
            # エッジシェーダーへの割り当て

            # エッジ設定
            self.edge_color_uniform[edge] = gl.glGetUniformLocation(program, "edgeColor")
            self.edge_size_uniform[edge] = gl.glGetUniformLocation(program, "edgeSize")

    def update_camera(self, edge=False) -> None:
        # 視野領域の決定
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(
            self.vertical_degrees,
            self.aspect_ratio,
            self.near_plane,
            self.far_plane,
        )

        self.projection_matrix = np.array(gl.glGetFloatv(gl.GL_PROJECTION_MATRIX), dtype=np.float32)

        # カメラ位置
        camera_mat = MMatrix4x4()
        camera_mat.rotate(self.camera_rotation)
        camera_mat.translate(self.camera_position)
        camera_pos: MVector3D = camera_mat * MVector3D()

        if not edge:
            gl.glUniform3f(self.camera_vec_uniform[edge], *camera_pos.vector)

        # 視点位置の決定
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(*camera_pos.vector, *self.look_at_center.vector, *self.look_at_up.vector)

        model_view_matrix = np.array(gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX), dtype=np.float32)
        gl.glUniformMatrix4fv(
            self.model_view_matrix_uniform[edge],
            1,
            gl.GL_FALSE,
            model_view_matrix,
        )

        gl.glUniformMatrix4fv(
            self.model_view_projection_matrix_uniform[edge],
            1,
            gl.GL_FALSE,
            np.matmul(model_view_matrix, self.projection_matrix),
        )

        # 隠面消去
        # https://learnopengl.com/Advanced-OpenGL/Depth-testing
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def fit(self, width: int, height: int):
        self.width = width
        self.height = height
        self.aspect_ratio = float(self.width) / float(self.height)

        for is_edge in [False, True]:
            self.use(edge=is_edge)

            # ビューポートの設定
            gl.glViewport(0, 0, self.width, self.height)

            self.update_camera(edge=is_edge)

            self.unuse()

    def use(self, edge=False):
        if edge:
            gl.glUseProgram(self.edge_program)
        else:
            gl.glUseProgram(self.model_program)

    def unuse(self):
        gl.glUseProgram(0)
