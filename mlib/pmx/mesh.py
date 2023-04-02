from typing import Optional

import numpy as np
import OpenGL.GL as gl

from mlib.base.math import MVector4D
from mlib.base.part import BaseIndexModel
from mlib.pmx.pmx_part import DrawFlg, Material, Texture
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
        self.data = data
        stride = sum([v["size"] for v in self.components.values()])
        for v in self.components.values():
            v["stride"] = stride * self.dsize
            v["pointer"] = v["offset"] * self.dsize
        self.set_vertex_attribute()

    def __del__(self) -> None:
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])

    def bind(self) -> None:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

    def unbind(self) -> None:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def set_vertex_attribute(self) -> None:
        self.bind()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data.nbytes, self.data, gl.GL_STATIC_DRAW)

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
        self.dtype = gl.GL_UNSIGNED_BYTE if data.dtype == np.uint8 else gl.GL_UNSIGNED_SHORT if data.dtype == np.uint16 else gl.GL_UNSIGNED_INT
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

    def draw_model(
        self,
        mats: np.ndarray,
        shader: MShader,
        ibo: IBO,
    ):
        if DrawFlg.DOUBLE_SIDED_DRAWING in self.material.draw_flg:
            # 両面描画
            # カリングOFF
            gl.glDisable(gl.GL_CULL_FACE)
        else:
            # 片面描画
            # カリングON
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)

        # ボーンデフォームテクスチャ設定
        self.bind_bone_matrixes(mats, shader, False)

        # ------------------
        # 材質色設定
        # full.fx の AmbientColor相当
        gl.glUniform4f(
            shader.diffuse_uniform[False],
            *(
                self.material.diffuse_color * shader.light_ambient4
                + MVector4D(
                    self.material.ambient_color.x,
                    self.material.ambient_color.y,
                    self.material.ambient_color.z,
                    self.material.diffuse_color.w,
                )
            ).vector
        )
        # TODO 材質モーフの色を入れる
        gl.glUniform3f(shader.ambient_uniform[False], *(self.material.diffuse_color.xyz * shader.light_ambient).vector)
        gl.glUniform4f(shader.specular_uniform[False], *(self.material.specular_color * shader.light_specular).vector, self.material.specular_factor)

        # テクスチャ使用有無
        gl.glUniform1i(shader.use_texture_uniform[False], self.texture is not None and self.texture.valid)
        if self.texture and self.texture.valid:
            self.texture.bind()
            gl.glUniform1i(shader.texture_uniform[False], self.texture.texture_type.value)

        # Toon使用有無
        gl.glUniform1i(shader.use_toon_uniform[False], self.toon_texture is not None and self.toon_texture.valid)
        if self.toon_texture and self.toon_texture.valid:
            self.toon_texture.bind()
            gl.glUniform1i(shader.toon_uniform[False], self.toon_texture.texture_type.value)

        # Sphere使用有無
        gl.glUniform1i(shader.use_sphere_uniform[False], self.sphere_texture is not None and self.sphere_texture.valid)
        if self.sphere_texture and self.sphere_texture.valid:
            self.sphere_texture.bind()
            gl.glUniform1i(shader.sphere_mode_uniform[False], self.material.sphere_mode)
            gl.glUniform1i(shader.sphere_uniform[False], self.sphere_texture.texture_type.value)

        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.material.vertices_count,
            ibo.dtype,
            gl.ctypes.c_void_p(self.prev_vertices_pointer),
        )

        if self.texture and self.texture.valid:
            self.texture.unbind()

        if self.toon_texture and self.toon_texture.valid:
            self.toon_texture.unbind()

        if self.sphere_texture and self.sphere_texture.valid:
            self.sphere_texture.unbind()

        self.unbind_bone_matrixes()

    def draw_edge(
        self,
        mats: np.ndarray,
        shader: MShader,
        ibo: IBO,
    ):
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_FRONT)

        # ボーンデフォームテクスチャ設定
        self.bind_bone_matrixes(mats, shader, True)

        # ------------------
        # エッジ設定
        gl.glUniform4f(shader.edge_color_uniform[True], *self.material.edge_color.vector)
        gl.glUniform1f(shader.edge_size_uniform[True], self.material.edge_size)

        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.material.vertices_count,
            ibo.dtype,
            gl.ctypes.c_void_p(self.prev_vertices_pointer),
        )

        self.unbind_bone_matrixes()

    def bind_bone_matrixes(
        self,
        mats: np.ndarray,
        shader: MShader,
        edge: bool,
    ):
        # テクスチャをアクティブにする
        gl.glActiveTexture(gl.GL_TEXTURE3)

        # テクスチャをバインドする
        gl.glBindTexture(gl.GL_TEXTURE_2D, shader.bone_matrix_texture_id[edge])

        # テクスチャのパラメーターの設定
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # テクスチャをシェーダーに渡す
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            mats.shape[1],
            mats.shape[0],
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            mats.flatten(),
        )

        gl.glUniform1i(shader.bone_matrix_texture_uniform[edge], 3)

    def unbind_bone_matrixes(
        self,
    ):
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
