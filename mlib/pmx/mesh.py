from typing import Optional

import numpy as np
import OpenGL.GL as gl
from mlib.base.part import BaseIndexModel
from mlib.math import MMatrix4x4
from mlib.pmx.part import Material, Texture
from mlib.pmx.shader import MShader, VsLayout


class VAO:
    """
    VAO（Vertex Array Object） ･･･ 頂点情報と状態を保持するオブジェクト
    """

    def __init__(self) -> None:
        self.vao = gl.glGenVertexArrays(1)

    def bind(self) -> None:
        gl.glBindVertexArray(self.vao)

    def unbind(self) -> None:
        gl.glBindVertexArray(0)


class VBO:
    """
    VBO（Vertex Buffer Object）･･･ 頂点バッファオブジェクト
    """

    def __init__(self, data: np.ndarray, components: dict[int, dict[str, int]]) -> None:
        self.vbo = gl.glGenBuffers(1)
        self.dsize = np.dtype(data.dtype).itemsize
        self.components = components
        stride = sum([v["size"] for v in self.components.values()])
        for v in self.components.values():
            v["stride"] = stride * self.dsize
            v["pointer"] = v["offset"] * self.dsize
        self.set_vertex_attribute(data)

    def __del__(self) -> None:
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])

    def bind(self) -> None:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

    def unbind(self) -> None:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def set_vertex_attribute(self, data: np.ndarray) -> None:
        self.bind()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)

    def set_slot(self, slot: VsLayout) -> None:
        self.bind()
        gl.glEnableVertexAttribArray(slot.value)
        gl.glVertexAttribPointer(
            slot.value,
            self.components[slot.value]["size"],
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.components[slot.value]["stride"],
            gl.ctypes.c_void_p(self.components[slot.value]["pointer"]),
        )


class IBO:
    """
    IBO（Index Buffer Object） ･･･ インデックスバッファオブジェクト
    """

    def __init__(self, data: np.ndarray) -> None:
        self.ibo = gl.glGenBuffers(1)
        self.dtype = (
            gl.GL_UNSIGNED_BYTE
            if data.dtype == np.uint8
            else gl.GL_UNSIGNED_SHORT
            if data.dtype == np.uint16
            else gl.GL_UNSIGNED_INT
        )
        self.dsize = np.dtype(data.dtype).itemsize

        self.set_indices(data)

    def __del__(self) -> None:
        if self.ibo:
            gl.glDeleteBuffers(1, [self.ibo])

    def bind(self) -> None:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ibo)

    def unbind(self) -> None:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def set_indices(self, data: np.ndarray) -> None:
        self.bind()
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            data.nbytes,
            data,
            gl.GL_STATIC_DRAW,
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
        prev_vertices_count: int,
        face_dtype: type,
    ):
        super().__init__()
        self.material = material
        self.texture = texture
        self.toon_texture = toon_texture
        self.sphere_texture = sphere_texture
        self.prev_vertices_count = prev_vertices_count
        self.prev_vertices_pointer = prev_vertices_count * np.dtype(face_dtype).itemsize

    def draw(
        self,
        shader: MShader,
        ibo: IBO,
    ):
        # ボーンデフォーム設定
        gl.glUniformMatrix4fv(
            shader.bone_matrix_uniform, 1, gl.GL_FALSE, MMatrix4x4(identity=True).vector
        )

        # ------------------
        # 材質色設定
        gl.glUniform4f(shader.diffuse_uniform, *self.material.diffuse_color.vector)
        gl.glUniform3f(shader.ambient_uniform, *self.material.ambient_color.vector)
        gl.glUniform4f(
            shader.specular_uniform,
            *self.material.specular_color.vector,
            self.material.specular_factor
        )

        # テクスチャ使用有無
        gl.glUniform1i(shader.use_texture_uniform, self.texture is not None)
        if self.texture:
            self.texture.bind()
            gl.glUniform1i(shader.texture_uniform, self.texture.texture_type.value)

        # Toon使用有無
        gl.glUniform1i(shader.use_toon_uniform, self.toon_texture is not None)
        if self.toon_texture:
            self.toon_texture.bind()
            gl.glUniform1i(shader.toon_uniform, self.toon_texture.texture_type.value)

        # Sphere使用有無
        gl.glUniform1i(shader.use_sphere_uniform, self.sphere_texture is not None)
        if self.sphere_texture:
            self.sphere_texture.bind()
            gl.glUniform1i(shader.sphere_mode_uniform, self.material.sphere_mode)
            gl.glUniform1i(
                shader.sphere_uniform, self.sphere_texture.texture_type.value
            )

        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.material.vertices_count,
            ibo.dtype,
            gl.ctypes.c_void_p(self.prev_vertices_pointer),
        )
