from typing import Optional

import numpy as np
import OpenGL.GL as gl
from mlib.base.base import BaseModel
from mlib.base.part import BaseIndexModel
from mlib.math import MMatrix4x4, MVector3D
from mlib.pmx.part import Material, Texture
from mlib.pmx.shader import MShader


class MShaderInfo(BaseModel):
    def __init__(
        self,
        shader: MShader,
        vertex_position_list: list[float],
        vertex_uv_list: list[float],
        face_list: list[int],
        material: Material,
        texture: Optional[Texture],
        sphere_texture: Optional[Texture],
    ):
        self.shader = shader
        self.vertex_poses = np.array(vertex_position_list, dtype=np.float32).flatten()
        self.vertex_uvs = np.array(vertex_uv_list, dtype=np.float32).flatten()
        self.face_indexes = np.array(face_list, dtype=np.uint16)
        self.face_size = len(face_list) // 3

        self.material = material
        self.texture = texture
        self.sphere_texture = sphere_texture

        self.glsl_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.glsl_vao)

        # glsl_vbo_vertex = gl.glGenBuffers(1)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, glsl_vbo_vertex)
        # gl.glBufferData(
        #     gl.GL_ARRAY_BUFFER,
        #     self.vertex_poses.nbytes,
        #     self.vertex_poses,
        #     gl.GL_STATIC_DRAW,
        # )

        # gl.glEnableVertexAttribArray(shader.glsl_id_vertex)

        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        # glsl_vbo_uv = gl.glGenBuffers(1)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, glsl_vbo_uv)
        # gl.glBufferData(
        #     gl.GL_ARRAY_BUFFER,
        #     self.vertex_uvs.nbytes,
        #     self.vertex_uvs,
        #     gl.GL_STATIC_DRAW,
        # )
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # gl.glEnableVertexAttribArray(shader.glsl_id_uv)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, glsl_vbo_vertex)
        # gl.glVertexAttribPointer(
        #     shader.glsl_id_vertex, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None
        # )
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, glsl_vbo_uv)
        # gl.glVertexAttribPointer(
        #     shader.glsl_id_uv, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None
        # )
        # glsl_vbo_face = gl.glGenBuffers(1)
        # gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, glsl_vbo_face)
        # gl.glBufferData(
        #     gl.GL_ELEMENT_ARRAY_BUFFER,
        #     self.face_indexes.nbytes,
        #     self.face_indexes,
        #     gl.GL_STATIC_DRAW,
        # )
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        # gl.glBindVertexArray(0)
        # self.glsl_vao = glsl_vao

        mat = MMatrix4x4(identity=True)
        mat.translate(MVector3D(10, 10, 10))

        self.metrices: np.ndarray = mat.vector

    def draw(self):
        # self.shader.shader_on()
        # gl.glUniform1f(self.shader.glsl_id_is_texture, 0.00)
        # TODO
        # if self.texture:
        #     gl.glUniform1f(self.shader.glsl_id_is_texture, 1.00)
        #     self.texture.draw()
        # gl.glUniform4f(
        #     self.shader.glsl_id_color,
        #     self.material.diffuse_color.x,
        #     self.material.diffuse_color.y,
        #     self.material.diffuse_color.z,
        #     self.material.diffuse_color.w,
        # )
        # # gl.glUniform1f(self.shader.glsl_id_alpha, self.material.diffuse_color.w)
        # gl.glUniform1f(self.shader.glsl_id_alpha, 1)
        # # TODO
        # try:
        #     gl.glUniformMatrix4fv(
        #         self.shader.glsl_id_bone_matrix,
        #         1,
        #         gl.GL_FALSE,
        #         self.metrices,
        #     )
        # except Exception as e:
        #     from OpenGL.GLU import gluErrorString

        #     print(gluErrorString(e))
        # gl.glBindVertexArray(self.glsl_vao)
        # gl.glDrawElements(gl.GL_TRIANGLES, self.face_size, gl.GL_UNSIGNED_SHORT, None)
        # gl.glBindVertexArray(0)
        gl.glBindVertexArray(self.glsl_vao)
        # self.shader.shader_off()


class Mesh(BaseIndexModel):
    """
    メッシュデータ（描画用）
    """

    def __init__(
        self,
        shader: MShader,
        vertex_pose_list: list[float],
        vertex_uv_list: list[float],
        face_indexe_list: list[int],
        material: Material,
        texture: Optional[Texture],
        sphere_texture: Optional[Texture],
    ):
        super().__init__()
        self.shader_info = MShaderInfo(
            shader,
            vertex_pose_list,
            vertex_uv_list,
            face_indexe_list,
            material,
            texture,
            sphere_texture,
        )

    def draw(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        # TODO
        # if self.both_side_flag:
        #     gl.glDisable(gl.GL_CULL_FACE)
        # else:
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glFrontFace(gl.GL_CCW)
        gl.glCullFace(gl.GL_FRONT)
        # gl.glCullFace(gl.GL_BACK)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self.shader_info.draw()
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_CULL_FACE)
