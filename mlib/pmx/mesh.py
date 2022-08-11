from typing import Optional

import numpy as np
import OpenGL.GL as gl
from mlib.base.part import BaseIndexModel
from mlib.pmx.part import Material, Texture
from mlib.pmx.shader import MShader


class VBO:
    def __init__(self, data: np.ndarray) -> None:
        self.vbo = gl.glGenBuffers(1)
        self.set_vertex_attribute(data)

    def __del__(self) -> None:
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])

    def bind(self) -> None:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

    def unbind(self) -> None:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def set_vertex_attribute(self, data: np.ndarray) -> None:
        self.component_count = data.shape[1]
        self.bind()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)

    def set_slot(self, slot: int) -> None:
        self.bind()
        gl.glEnableVertexAttribArray(slot)
        gl.glVertexAttribPointer(
            slot,
            self.component_count,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            0,
            gl.ctypes.c_void_p(0),
        )


class IBO:
    def __init__(self, data: np.ndarray) -> None:
        self.ibo = gl.glGenBuffers(1)
        self.data = data
        self.set_indices()

    def __del__(self) -> None:
        if self.ibo:
            gl.glDeleteBuffers(1, [self.ibo])

    def bind(self) -> None:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ibo)

    def unbind(self) -> None:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def set_indices(self) -> None:
        self.bind()
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, self.data.nbytes, self.data, gl.GL_STATIC_DRAW
        )


class Mesh(BaseIndexModel):
    """
    メッシュデータ（描画用）
    """

    def __init__(
        self,
        material: Material,
        texture: Optional[Texture],
        toon_texture: Optional[Texture],
        sphere_texture: Optional[Texture],
        prev_vertex_count: int,
    ):
        super().__init__()
        self.material = material
        self.texture = texture
        self.toon_texture = toon_texture
        self.sphere_texture = sphere_texture
        self.prev_vertex_count = prev_vertex_count

    def draw(
        self,
        shader: MShader,
        ibo: IBO,
    ):
        # ------------------
        # 材質色設定
        gl.glUniform4f(shader.diffuse_uniform, *self.material.diffuse_color.vector)
        gl.glUniform3f(shader.ambient_uniform, *self.material.ambient_color.vector)
        gl.glUniform4f(
            shader.specular_uniform,
            *self.material.specular_color.vector,
            self.material.specular_factor
        )

        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.material.vertices_count,
            gl.GL_UNSIGNED_INT,
            gl.ctypes.c_void_p(self.prev_vertex_count),
        )
