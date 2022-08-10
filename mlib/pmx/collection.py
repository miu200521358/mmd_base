import os
from ctypes import c_float, sizeof
from glob import glob
from typing import Optional

import numpy as np
import OpenGL.GL as gl
from mlib.base.collection import (
    BaseHashModel,
    BaseIndexDictModel,
    BaseIndexListModel,
    BaseIndexNameListModel,
)
from mlib.math import MVector3D
from mlib.pmx.mesh import Mesh
from mlib.pmx.part import (
    Bone,
    DisplaySlot,
    Face,
    Joint,
    Material,
    Morph,
    RigidBody,
    Texture,
    ToonSharing,
    Vertex,
)
from mlib.pmx.shader import MShader, VsLayout


class Vertices(BaseIndexListModel[Vertex]):
    """
    頂点リスト
    """

    def __init__(self):
        super().__init__()


class Faces(BaseIndexListModel[Face]):
    """
    面リスト
    """

    def __init__(self):
        super().__init__()


class Textures(BaseIndexListModel[Texture]):
    """
    テクスチャリスト
    """

    def __init__(self):
        super().__init__()


class ToonTextures(BaseIndexDictModel[Texture]):
    """
    共有テクスチャ辞書
    """

    def __init__(self):
        super().__init__()


class Materials(BaseIndexNameListModel[Material]):
    """
    材質リスト
    """

    def __init__(self):
        super().__init__()


class Bones(BaseIndexNameListModel[Bone]):
    """
    ボーンリスト
    """

    def __init__(self):
        super().__init__()

    def get_max_layer(self) -> int:
        """
        最大変形階層を取得

        Returns
        -------
        int
            最大変形階層
        """
        return max([b.layer for b in self.data])

    def get_bones_by_layer(self, layer: int) -> list[Bone]:
        return [b for b in self.data if b.layer == layer]


class Morphs(BaseIndexNameListModel[Morph]):
    """
    モーフリスト
    """

    def __init__(self):
        super().__init__()


class DisplaySlots(BaseIndexNameListModel[DisplaySlot]):
    """
    表示枠リスト
    """

    def __init__(
        self,
    ):
        super().__init__()


class RigidBodies(BaseIndexNameListModel[RigidBody]):
    """
    剛体リスト
    """

    def __init__(self):
        super().__init__()


class Joints(BaseIndexNameListModel[Joint]):
    """
    ジョイントリスト
    """

    def __init__(self):
        super().__init__()


class PmxModel(BaseHashModel):
    """
    Pmxモデルデータ

    Parameters
    ----------
    path : str, optional
        パス, by default ""
    signature : str, optional
        signature, by default ""
    version : float, optional
        バージョン, by default 0.0
    extended_uv_count : int, optional
        追加UV数, by default 0
    vertex_count : int, optional
        頂点数, by default 0
    texture_count : int, optional
        テクスチャ数, by default 0
    material_count : int, optional
        材質数, by default 0
    bone_count : int, optional
        ボーン数, by default 0
    morph_count : int, optional
        モーフ数, by default 0
    rigidbody_count : int, optional
        剛体数, by default 0
    name : str, optional
        モデル名, by default ""
    english_name : str, optional
        モデル名英, by default ""
    comment : str, optional
        コメント, by default ""
    english_comment : str, optional
        コメント英, by default ""
    json_data : dict, optional
        JSONデータ（vroidデータ用）, by default {}
    """

    def __init__(
        self,
        path: str = None,
    ):
        super().__init__(path=path or "")
        self.signature: str = ""
        self.version: float = 0.0
        self.extended_uv_count: int = 0
        self.vertex_count: int = 0
        self.texture_count: int = 0
        self.material_count: int = 0
        self.bone_count: int = 0
        self.morph_count: int = 0
        self.rigidbody_count: int = 0
        self.name: str = ""
        self.english_name: str = ""
        self.comment: str = ""
        self.english_comment: str = ""
        self.json_data: dict = {}
        self.vertices = Vertices()
        self.faces = Faces()
        self.textures = Textures()
        self.toon_textures = ToonTextures()
        self.materials = Materials()
        self.bones = Bones()
        self.morphs = Morphs()
        self.display_slots = DisplaySlots()
        self.rigidbodies = RigidBodies()
        self.joints = Joints()
        self.for_draw = False
        self.meshs = None

    def get_name(self) -> str:
        return self.name

    def init_draw(self, shader):
        if self.for_draw:
            # 既にフラグが立ってたら描画初期化済み
            return

        # 描画初期化
        self.for_draw = True
        # 共有Toon読み込み
        for tidx, tpath in enumerate(glob("resources/share_toon/*.bmp")):
            self.toon_textures[tidx] = Texture(os.path.abspath(tpath))

        self.meshs = Meshs(shader, self)

    def update(self):
        if not self.for_draw:
            return
        self.meshs.update()

    def draw(self):
        if not self.for_draw:
            return
        self.meshs.draw()


class Meshs(BaseIndexListModel[Mesh]):
    """
    メッシュリスト
    """

    VERTEX_BINDING_POINT = 0

    def __init__(self, shader: MShader, model: PmxModel):
        super().__init__()

        self.shader = shader

        # 頂点情報
        self.vertices = np.array(
            [
                np.array(
                    [
                        *(v.position + MVector3D(0, 0, 0)).vector,
                        *v.normal.vector,
                        *v.uv.vector,
                    ],
                    dtype=np.float32,
                )
                for v in model.vertices
            ],
            dtype=np.float32,
        )
        # 面情報
        self.faces = np.array(
            [np.array(f.vertices, dtype=np.int8) for f in model.faces], dtype=np.int8
        )

        prev_face_count = 0
        for material in model.materials:
            texture: Optional[Texture] = None
            if material.texture_index >= 0:
                texture = model.textures[material.texture_index]
                texture.init_draw(model.path, material.texture_index)

            toon_texture: Optional[Texture] = None
            if ToonSharing.SHARING == material.toon_sharing_flg:
                # 共有Toon
                toon_texture = model.toon_textures[material.toon_texture_index]
                toon_texture.init_draw(
                    model.path, material.toon_texture_index, is_individual=False
                )
            elif (
                ToonSharing.INDIVIDUAL == material.toon_sharing_flg
                and material.toon_texture_index >= 0
            ):
                # 個別Toon
                toon_texture = model.textures[material.toon_texture_index]
                # 共有Toonのを優先させるため、INDEXをずらす
                toon_texture.init_draw(model.path, material.toon_texture_index + 10)

            sphere_texture: Optional[Texture] = None
            if material.sphere_texture_index >= 0:
                sphere_texture = model.textures[material.sphere_texture_index]
                sphere_texture.init_draw(model.path, material.texture_index)

            self.data.append(
                Mesh(material, texture, toon_texture, sphere_texture, prev_face_count)
            )

            prev_face_count += material.vertices_count // 3

        # ---------------------
        # VAO
        self.vao_vertices = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_vertices)

        # ---------------------
        # VBO: 頂点

        self.vbo_vertices = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_vertices)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self.vertices.nbytes,
            self.vertices,
            gl.GL_STATIC_DRAW,
        )
        gl.glVertexAttribPointer(
            VsLayout.POSITION_ID,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            0,
            gl.ctypes.c_void_p(0),
        )
        gl.glVertexAttribPointer(
            VsLayout.NORMAL_ID,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            0,
            gl.ctypes.c_void_p(0),
        )
        gl.glVertexAttribPointer(
            VsLayout.UV_ID,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            0,
            gl.ctypes.c_void_p(0),
        )

        # ---------------------
        # EBO: 面

        self.ebo_faces = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_faces)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            self.faces.nbytes,
            self.faces,
            gl.GL_STATIC_DRAW,
        )

        # ---------------------
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def update(self):
        pass

    def draw(self):
        for m in self.data:
            gl.glUseProgram(self.shader.program)
            gl.glBindVertexArray(self.vao_vertices)

            gl.glEnableVertexAttribArray(VsLayout.POSITION_ID)
            gl.glEnableVertexAttribArray(VsLayout.NORMAL_ID)
            gl.glEnableVertexAttribArray(VsLayout.UV_ID)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_vertices)
            gl.glVertexAttribPointer(
                VsLayout.POSITION_ID,
                3,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                0,
                gl.ctypes.c_void_p(0),
            )
            gl.glVertexAttribPointer(
                VsLayout.NORMAL_ID,
                3,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                0,
                gl.ctypes.c_void_p(0),
            )
            gl.glVertexAttribPointer(
                VsLayout.UV_ID,
                2,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                0,
                gl.ctypes.c_void_p(0),
            )

            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_faces)

            m.draw(
                self.shader,
            )

            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glDisableVertexAttribArray(VsLayout.POSITION_ID)
            gl.glDisableVertexAttribArray(VsLayout.NORMAL_ID)
            gl.glDisableVertexAttribArray(VsLayout.UV_ID)

            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
