from typing import Optional

import OpenGL.GL as gl
from mlib.base.part import BaseIndexModel
from mlib.pmx.part import Material, Texture
from mlib.pmx.shader import MShader


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
        prev_face_count: int,
    ):
        super().__init__()
        self.material = material
        self.texture = texture
        self.toon_texture = toon_texture
        self.sphere_texture = sphere_texture
        self.prev_face_count = prev_face_count

    def draw(
        self,
        shader: MShader,
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

        # ------------------

        # 該当色で描画
        gl.glDrawElements(
            gl.GL_TRIANGLES,
            self.material.vertices_count // 3,
            gl.GL_UNSIGNED_INT,
            self.prev_face_count,
        )

        # -------------------
