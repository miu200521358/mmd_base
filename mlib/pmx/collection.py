import os
from glob import glob
from typing import Optional

import numpy as np
from mlib.base.collection import (BaseHashModel, BaseIndexDictModel,
                                  BaseIndexListModel, BaseIndexNameListModel)
from mlib.math import MVector3D
from mlib.pmx.mesh import IBO, VAO, VBO, Mesh
from mlib.pmx.part import (Bone, DisplaySlot, Face, Joint, Material, Morph,
                           RigidBody, Texture, TextureType, ToonSharing,
                           Vertex)
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
    def __init__(self, shader: MShader, model: PmxModel):
        super().__init__()

        self.shader = shader

        # 頂点情報
        self.vertices = np.array(
            [
                np.array(
                    [
                        *(v.position + MVector3D(0, -10, 0)).vector / 15,
                        *v.normal.vector,
                        v.uv.x,
                        1 - v.uv.y,
                        *(
                            v.extended_uvs[0].vector
                            if len(v.extended_uvs) > 0
                            else [0, 0]
                        ),
                    ],
                    dtype=np.float32,
                )
                for v in model.vertices
            ],
            dtype=np.float32,
        )

        face_dtype: type = (
            np.uint8
            if model.vertex_count == 1
            else np.uint16
            if model.vertex_count == 2
            else np.uint32
        )

        # 面情報
        self.faces: np.ndarray = np.array(
            [
                np.array(
                    [f.vertices[0], f.vertices[1], f.vertices[2]], dtype=face_dtype
                )
                for f in model.faces
            ],
            dtype=face_dtype,
        )

        prev_vertices_count = 0
        for material in model.materials:
            texture: Optional[Texture] = None
            if material.texture_index >= 0:
                texture = model.textures[material.texture_index]
                texture.init_draw(model.path, TextureType.TEXTURE)

            toon_texture: Optional[Texture] = None
            if ToonSharing.SHARING == material.toon_sharing_flg:
                # 共有Toon
                toon_texture = model.toon_textures[material.toon_texture_index]
                toon_texture.init_draw(
                    model.path, TextureType.TOON, is_individual=False
                )
            elif (
                ToonSharing.INDIVIDUAL == material.toon_sharing_flg
                and material.toon_texture_index >= 0
            ):
                # 個別Toon
                toon_texture = model.textures[material.toon_texture_index]
                toon_texture.init_draw(model.path, TextureType.TOON)

            sphere_texture: Optional[Texture] = None
            if material.sphere_texture_index >= 0:
                sphere_texture = model.textures[material.sphere_texture_index]
                sphere_texture.init_draw(model.path, TextureType.SPHERE)

            self.append(
                Mesh(
                    material,
                    texture,
                    toon_texture,
                    sphere_texture,
                    prev_vertices_count,
                    face_dtype,
                )
            )

            prev_vertices_count += material.vertices_count

        # ---------------------

        self.vao = VAO()
        self.vbo_vertices = VBO(
            self.vertices,
            {
                VsLayout.POSITION_ID.value: {"size": 3, "offset": 0},
                VsLayout.NORMAL_ID.value: {"size": 3, "offset": 3},
                VsLayout.UV_ID.value: {"size": 2, "offset": 6},
                VsLayout.EXTEND_UV_ID.value: {"size": 2, "offset": 8},
            },
        )

        self.ibo_faces = IBO(self.faces)

    def update(self):
        pass

    def draw(self):
        for mesh in self.data:
            self.vao.bind()
            self.vbo_vertices.set_slot(VsLayout.POSITION_ID)
            self.vbo_vertices.set_slot(VsLayout.NORMAL_ID)
            self.vbo_vertices.set_slot(VsLayout.UV_ID)
            self.vbo_vertices.set_slot(VsLayout.EXTEND_UV_ID)
            self.ibo_faces.bind()

            # 描画
            mesh.draw(self.shader, self.ibo_faces)

            self.ibo_faces.unbind()
            self.vbo_vertices.unbind()
            self.vao.unbind()
